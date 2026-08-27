
from typing import List, Union, Optional, Dict, Tuple
import copy
from collections import defaultdict
import torch
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin
from dataclasses import dataclass, field
import PIL
import re

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import process_image, collate_fn
import tracerigor.env
from tracerigor.env import REGISTERED_ENV
from tracerigor.rollout.multimodal_utils import get_multimodal_handler, detect_model_type
from tracerigor.server.client import BatchEnvClient
from tracerigor.rollout.utils.mask_utils import compute_loss_mask
from tracerigor.utils.response_utils import extract_reasoning_content

def find_actual_tag_ids(ids: torch.Tensor, tokenizer, tag="</think>", max_window=8):
    """
    Returns a list of (start_idx, end_idx_exclusive, ids_slice)
    where tokenizer.decode(ids[start:end]) == tag for some window size <= max_window.
    Also prints non-canonical segmentations where the decoded text equals `tag`
    but the ID slice does NOT equal the canonical tokenizer.encode(tag) pattern.
    """
    ids = ids.cpu()
    L = ids.size(0)
    matches = []

    # NEW: canonical segmentation for this tag in isolation
    canonical_ids = tokenizer.encode(tag, add_special_tokens=False)

    for i in range(L):
        for w in range(1, max_window + 1):
            j = i + w
            if j > L:
                break
            slice_ids = ids[i:j].tolist()
            text = tokenizer.decode(slice_ids)
            if text == tag:
                matches.append((i, j, slice_ids))
                # NEW: log non-canonical segmentations explicitly
                if slice_ids != canonical_ids:
                    print(
                        f"[DEBUG-tag] Non-canonical segmentation for {repr(tag)} "
                        f"at tokens[{i}:{j}]: ids={slice_ids}, canonical={canonical_ids}"
                    )
                # If you want to also see canonical hits, uncomment:
                # else:
                #     print(
                #         f"[DEBUG-tag] Canonical segmentation for {repr(tag)} "
                #         f"at tokens[{i}:{j}]: ids={slice_ids}"
                #     )
    return matches

class QwenVLRolloutManagerService():
    """generate_batch_for_rollout takes place first and then generate_rollout_batch_for_update starts"""
    def __init__(self,
                 actor_rollout_wg,
                 config,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 split="train",
                 ):
        self.split=split
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.recorder= None # defaultdict(list) env_id:record
        self.envs = None # dict env_id:env_config_instance
        self.system_prompts = None # dict env_id:str
        self.env_states = None # dict
        self.batch_idx_to_env_id = None # dict
        self.rollout_analysis_cache = None # defaultdict(list) env_id: per-turn rollout artifacts
        self.env_client = BatchEnvClient(base_url=self.config.base_url,timeout=self.config.timeout,max_workers=self.config.max_workers)

        # Detect model type and get appropriate handler
        self.model_type = detect_model_type(processor)
        self.multimodal_handler = get_multimodal_handler(self.model_type)


    @torch.no_grad()
    def _handle_special_tokens(self, llm_raw_response: str, prep_for_loss_mask: bool) -> str:
        """
        1. Filter out special tokens: <image> and special tokens marking environment observation in the llm generated response
        2. prep_for_loss_mask: if true, add special tokens to the beginning and end of the response if compute_loss_mask is True
        """
        llm_raw_response = llm_raw_response.replace('<image>', '')  # <image> is not in llm_raw_response
        if prep_for_loss_mask:
            # filtering special tokens for llm_raw_response, then adding them to the beginning and end of the response for loss mask computation
            sptk_b = self.config.special_token_for_loss_mask[0]
            sptk_e = self.config.special_token_for_loss_mask[1]
            llm_raw_response = llm_raw_response.replace(sptk_b, '')  # sptk_b is not in llm_raw_response
            llm_raw_response = llm_raw_response.replace(sptk_e, '')  # sptk_e is not in llm_raw_response
            llm_raw_response = sptk_b + llm_raw_response + sptk_e
        return llm_raw_response

    @torch.no_grad()
    def _handle_multi_modal_data(
            self,
            prompt_template: str,
            row_dict: Dict,
            image_data: List[PIL.Image.Image],
            do_embedding: bool = True,
        ) -> str:
        """Handle multi-modal data in the prompt template using model-specific handlers.

        - For do_embedding=False(vllm), replace <image> with model-specific tokens -> raw_prompt
        - For do_embedding=True, replace <image> with model-specific embedded tokens -> prompt_template
        """
        if self.multimodal_handler is None:
            raise ValueError("No multimodal handler available. Processor might be None.")

        return self.multimodal_handler(
            prompt_template=prompt_template,
            row_dict=row_dict,
            image_data=image_data,
            processor=self.processor,
            do_embedding=do_embedding
        )

    def _latest_user_history_index(self, history_len: int, is_final: bool) -> Optional[int]:
        if history_len <= 0:
            return None
        latest_idx = history_len - 2 if is_final else history_len - 1
        return latest_idx if latest_idx >= 0 else None

    def _prepare_user_turn_for_prompt(
        self,
        record: Dict,
        history_index: int,
        history_len: int,
        is_final: bool,
        image_placeholder: str,
    ) -> Tuple[str, List[PIL.Image.Image]]:
        obs_text = record['obs_str']
        image_data = list(record.get('image_data', []))

        if not image_data:
            return obs_text, []

        if not self.config.get("latest_image_only_in_history", False):
            return obs_text, image_data

        latest_user_index = self._latest_user_history_index(history_len, is_final)
        if latest_user_index is not None and history_index == latest_user_index:
            return obs_text, image_data

        historical_text = obs_text.replace(image_placeholder, "[historical image omitted]")
        historical_text = re.sub(r'[ \t]+\n', '\n', historical_text)
        historical_text = re.sub(r'\n{3,}', '\n\n', historical_text).strip()
        if not historical_text:
            historical_text = "[historical image omitted]"
        return historical_text, []

    @torch.no_grad()
    def _compute_loss_mask(self, input_ids, attention_mask):
        # Get token IDs for special tokens and pad token
        sptk_b = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[0])  # sptk_b token id is 151648
        sptk_e = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[1])  # sptk_e token id is 151649
        pad_token_id = self.tokenizer.pad_token_id  # pad_token_id is 151643

        return compute_loss_mask(input_ids, attention_mask, sptk_b, sptk_e, pad_token_id)

    # @torch.no_grad()
    # def _build_span_mask(
    #     self,
    #     input_ids_1d: torch.Tensor,
    #     start_tokens: List[int],
    #     end_tokens: List[int],
    # ) -> torch.Tensor:
    #     """
    #     Build a 1D mask over input_ids_1d that is 1 inside every
    #     <start_tokens> ... <end_tokens> span (inclusive), 0 elsewhere.

    #     - input_ids_1d: shape (seq_len,)
    #     - start_tokens / end_tokens: token-id sequences for delimiters
    #     """
    #     mask = torch.zeros_like(input_ids_1d, dtype=torch.float32)
    #     if input_ids_1d.numel() == 0:
    #         return mask

    #     ids = input_ids_1d.tolist()
    #     n = len(ids)
    #     ns = len(start_tokens)
    #     ne = len(end_tokens)

    #     i = 0
    #     while i <= n - ns:
    #         # match start pattern
    #         if ids[i : i + ns] == start_tokens:
    #             # find the next end pattern
    #             j = i + ns
    #             end_j = None
    #             while j <= n - ne:
    #                 if ids[j : j + ne] == end_tokens:
    #                     end_j = j + ne  # exclusive
    #                     break
    #                 j += 1
    #             if end_j is None:
    #                 end_j = n
    #             # mark [i, end_j)
    #             mask[i:end_j] = 1.0
    #             i = end_j
    #         else:
    #             i += 1
    #     return mask

    ## a robust version of _build_span_mask that handles truncated-end behavior
    # @torch.no_grad()
    # def _build_span_mask(
    #     self,
    #     input_ids_1d: torch.Tensor,
    #     start_tokens: List[int],
    #     end_tokens: List[int],
    # ) -> torch.Tensor:
    #     """
    #     Build a 1D mask over input_ids_1d that is 1 inside every
    #     <start_tokens> ... <end_tokens> span (exclusive of the delimiters),
    #     0 elsewhere.

    #     - input_ids_1d: shape (seq_len,)
    #     - start_tokens / end_tokens: token-id sequences for delimiters

    #     Truncated-end behavior: if a start is found but no end, span runs
    #     until the end of the sequence, just like the truncated-end logic
    #     in compute_loss_mask.
    #     """
    #     mask = torch.zeros_like(input_ids_1d, dtype=torch.float32)
    #     if input_ids_1d.numel() == 0:
    #         return mask

    #     ids = input_ids_1d.tolist()
    #     n = len(ids)
    #     ns = len(start_tokens)
    #     ne = len(end_tokens)

    #     i = 0
    #     while i <= n - ns:
    #         # match start pattern
    #         if ids[i : i + ns] == start_tokens:
    #             # span starts *after* the start tokens
    #             span_start = i + ns
    #             j = span_start
    #             end_found = False
    #             # search for end pattern
    #             while j <= n - ne:
    #                 if ids[j : j + ne] == end_tokens:
    #                     span_end = j  # exclusive of end tokens
    #                     end_found = True
    #                     break
    #                 j += 1
    #             if not end_found:
    #                 # truncated-end case: extend to end of sequence
    #                 span_end = n

    #             if span_start < span_end:
    #                 mask[span_start:span_end] = 1.0

    #             # skip to *after* the end tokens if found,
    #             # otherwise we are at end already
    #             i = span_end + (ne if end_found else 0)
    #         else:
    #             i += 1

    #     return mask

    ## a more robust version of _build_span_mask that handles truncated-start and truncated-end behavior
    # def _build_span_mask(
    #     self,
    #     input_ids_1d: torch.Tensor,
    #     start_tokens: list[int],
    #     end_tokens: list[int],
    # ) -> torch.Tensor:
    #     """
    #     Build a mask that is 1.0 for tokens between <start_tokens> and <end_tokens>,
    #     including the boundary tokens themselves, robust to truncation.

    #     Handles:
    #     - multiple <think>...</think> spans
    #     - truncated start (seeing </think> before any <think>)
    #     - truncated end (seeing <think> without a later </think>)
    #     """
    #     device = input_ids_1d.device
    #     L = input_ids_1d.shape[0]
    #     mask = torch.zeros(L, dtype=torch.float32, device=device)
    #     if L == 0:
    #         return mask

    #     ids = input_ids_1d.tolist()
    #     ns = len(start_tokens)
    #     ne = len(end_tokens)

    #     start_positions = []
    #     end_positions_last = []

    #     # find all start positions
    #     if ns > 0 and L >= ns:
    #         for i in range(0, L - ns + 1):
    #             if ids[i:i + ns] == start_tokens:
    #                 start_positions.append(i)

    #     # find all end positions (we use index of the *last* end-token)
    #     if ne > 0 and L >= ne:
    #         for i in range(0, L - ne + 1):
    #             if ids[i:i + ne] == end_tokens:
    #                 end_positions_last.append(i + ne - 1)

    #     if len(start_positions) == 0 and len(end_positions_last) == 0:
    #         return mask

    #     s_indices = torch.tensor(start_positions, device=device, dtype=torch.long)
    #     e_indices = torch.tensor(end_positions_last, device=device, dtype=torch.long)

    #     # truncated start: first end before first start
    #     if e_indices.numel() > 0 and (s_indices.numel() == 0 or e_indices[0] < s_indices[0]):
    #         s_indices = torch.cat([torch.tensor([-1], device=device), s_indices])

    #     # truncated end: last start has no matching end
    #     if s_indices.numel() > e_indices.numel():
    #         e_indices = torch.cat(
    #             [e_indices, torch.tensor([L - 1], device=device)]
    #         )

    #     for s_idx, e_idx in zip(s_indices, e_indices):
    #         s = int(s_idx.item())
    #         e = int(e_idx.item())
    #         if s < 0 and e < L:
    #             # truncated start: span started before this chunk
    #             mask[: e + 1] = 1.0
    #         elif s >= 0 and e < L:
    #             # normal span
    #             mask[s : e + 1] = 1.0
    #         elif s >= 0 and e >= L:
    #             # truncated end: span continues beyond this chunk
    #             mask[s:] = 1.0
    #         elif s < 0 and e >= L:
    #             # entire chunk is inside a span
    #             mask[:] = 1.0

    #     return mask

    def _find_pattern_positions(self, ids_sub: torch.Tensor, pattern_ids: torch.Tensor) -> List[int]:
        """Return list of start indices where `pattern_ids` appears in `ids_sub`."""
        L = ids_sub.size(0)
        P = pattern_ids.size(0)
        if P == 0 or L < P:
            return []
        positions = []
        # simple loop; ids_sub length is O(1e3–1e4), so this is fine
        for i in range(L - P + 1):
            if torch.all(ids_sub[i : i + P] == pattern_ids):
                positions.append(i)
        return positions

    # Supported reasoning tag pairs — tried in order.
    _REASONING_TAG_PAIRS = [
        ("<think>", "</think>"),
        ("<reflection>", "</reflection>"),
    ]

    def _build_think_mask_for_response(
        self,
        ids_span: torch.Tensor,
        include_tags: bool = True,
        debug: bool = False,
    ) -> Tuple[torch.BoolTensor, bool]:
        """
        Build a think mask for a *single* assistant response span (no prompts).

        Format-agnostic: tries <think>...</think> first, then
        <reflection>...</reflection>, so the mask works for both free_think
        and reflact formats.

        Args:
            ids_span: (T,) token ids for this response only.
            include_tags: if True, include tag tokens themselves in the mask.
            debug: print debug info.

        Returns:
            local_mask: (T,) bool tensor, True where token is part of reasoning span.
            ok: whether a well-formed reasoning span was found.
        """
        device = ids_span.device
        T = ids_span.size(0)
        local_mask = torch.zeros(T, dtype=torch.bool, device=device)

        # Decode full response text
        text = self.tokenizer.decode(ids_span.tolist(), skip_special_tokens=False)

        # Auto-detect which reasoning tag pair is present
        start_tag = end_tag = None
        for s, e in self._REASONING_TAG_PAIRS:
            if s in text and e in text:
                start_tag, end_tag = s, e
                break

        if start_tag is None:
            if debug:
                print("[think-mask][resp] no reasoning tags found in response text")
            return local_mask, False

        if debug:
            print(f"[think-mask][resp] detected tags: {start_tag} / {end_tag}")

        # --- 1) ID-based matching first ---
        start_ids = self.tokenizer.encode(start_tag, add_special_tokens=False)
        end_ids   = self.tokenizer.encode(end_tag, add_special_tokens=False)

        if debug:
            print(f"[think-mask][resp] start_ids={start_ids}, end_ids={end_ids}")

        if start_ids and end_ids:
            start_ids_t = torch.tensor(start_ids, dtype=ids_span.dtype, device=device)
            end_ids_t   = torch.tensor(end_ids,   dtype=ids_span.dtype, device=device)

            start_pos_rel = self._find_pattern_positions(ids_span, start_ids_t)
            end_pos_rel   = self._find_pattern_positions(ids_span, end_ids_t)

            if debug:
                print(f"[think-mask][resp] ID-starts={start_pos_rel}")
                print(f"[think-mask][resp] ID-ends={end_pos_rel}")

            if start_pos_rel and end_pos_rel:
                first_start = start_pos_rel[0]
                last_end_start = end_pos_rel[-1]

                if include_tags:
                    inner_start_tok = first_start                      # '<think>'
                    inner_end_tok_excl = last_end_start + len(end_ids) # past '</think>'
                else:
                    inner_start_tok = first_start + len(start_ids)     # after '<think>'
                    inner_end_tok_excl = last_end_start                # before '</think>'

                inner_start_tok = max(inner_start_tok, 0)
                inner_end_tok_excl = min(inner_end_tok_excl, T)

                if inner_start_tok < inner_end_tok_excl:
                    local_mask[inner_start_tok:inner_end_tok_excl] = True
                    if debug:
                        print(f"[think-mask][resp] ID-based inner span="
                              f"[{inner_start_tok}, {inner_end_tok_excl})")
                    return local_mask, True
                else:
                    if debug:
                        print("[think-mask][resp] ID-based inner span invalid; "
                              "falling back to char-level.")

        # --- 2) Char-level fallback (uses detected start_tag / end_tag) ---
        s_char = text.find(start_tag)
        e_char_start = text.rfind(end_tag)

        if s_char == -1 or e_char_start == -1 or e_char_start <= s_char:
            if debug:
                print(f"[think-mask][resp] malformed {start_tag}...{end_tag} span in text; "
                      "returning zeros.")
            return local_mask, False

        if include_tags:
            inner_start_char = s_char
            inner_end_char = e_char_start + len(end_tag)
        else:
            inner_start_char = s_char + len(start_tag)
            inner_end_char = e_char_start

        if debug:
            print(f"[think-mask][resp] char span=[{inner_start_char}, {inner_end_char})")

        ids_cpu = ids_span.detach().cpu()
        pieces = [
            self.tokenizer.decode([int(tid)], skip_special_tokens=False)
            for tid in ids_cpu.tolist()
        ]
        offsets = [0]
        acc = 0
        for p in pieces:
            acc += len(p)
            offsets.append(acc)

        for i in range(T):
            tok_s = offsets[i]
            tok_e = offsets[i + 1]
            if tok_e <= inner_start_char or tok_s >= inner_end_char:
                continue
            local_mask[i] = True

        if debug:
            print(f"[think-mask][resp] char-level marked "
                  f"{int(local_mask.sum().item())} tokens")

        return local_mask, True

    def _build_subtag_masks_for_response(
        self,
        ids_span: torch.Tensor,
        include_tags: bool = True,
        debug: bool = False,
    ) -> tuple[dict[str, torch.BoolTensor], bool]:
        """
        Build masks for <think> and its subtags within a single assistant response.

        Args:
            ids_span: (T,) token ids for this response only.
            include_tags: if True, include the tag tokens themselves in the masks.
            debug: print debug info.

        Returns:
            (masks, ok) where:
                masks = {
                    "think":       (T,) bool,
                    "observation": (T,) bool,
                    "reasoning":   (T,) bool,
                    "prediction":  (T,) bool,
                }
            'ok' is True if a well-formed <think>...</think> span was found.
            Subtag masks are always subsets of the think mask.
        """
        device = ids_span.device
        T = ids_span.size(0)

        # 1) reuse existing logic to locate <think> span
        think_mask, ok = self._build_think_mask_for_response(
            ids_span,
            include_tags=include_tags,
            debug=debug,
        )

        obs_mask = torch.zeros(T, dtype=torch.bool, device=device)
        reason_mask = torch.zeros(T, dtype=torch.bool, device=device)
        pred_mask = torch.zeros(T, dtype=torch.bool, device=device)

        if not ok or not think_mask.any():
            return {
                "think": think_mask,
                "observation": obs_mask,
                "reasoning": reason_mask,
                "prediction": pred_mask,
            }, ok

        # 2) Decode once and build char offsets per token
        text = self.tokenizer.decode(ids_span.tolist(), skip_special_tokens=False)
        ids_cpu = ids_span.detach().cpu()
        pieces = [
            self.tokenizer.decode([int(tid)], skip_special_tokens=False)
            for tid in ids_cpu.tolist()
        ]
        offsets = [0]
        acc = 0
        for p in pieces:
            acc += len(p)
            offsets.append(acc)
        # offsets[i]..offsets[i+1] is approximate char span for token i

        def mark_tag(start_tag: str, end_tag: str) -> torch.BoolTensor:
            mask = torch.zeros(T, dtype=torch.bool, device=device)
            s_char = text.find(start_tag)
            if s_char == -1:
                return mask
            e_char_start = text.find(end_tag, s_char + len(start_tag))
            if e_char_start == -1 or e_char_start <= s_char:
                return mask

            if include_tags:
                inner_start = s_char
                inner_end = e_char_start + len(end_tag)
            else:
                inner_start = s_char + len(start_tag)
                inner_end = e_char_start

            for i in range(T):
                tok_s = offsets[i]
                tok_e = offsets[i + 1]
                if tok_e <= inner_start or tok_s >= inner_end:
                    continue
                mask[i] = True
            return mask

        obs_mask = mark_tag("<observation>", "</observation>")
        reason_mask = mark_tag("<reasoning>", "</reasoning>")
        pred_mask = mark_tag("<prediction>", "</prediction>")

        # ensure subtags are subsets of think
        obs_mask &= think_mask
        reason_mask &= think_mask
        pred_mask &= think_mask

        return {
            "think": think_mask,
            "observation": obs_mask,
            "reasoning": reason_mask,
            "prediction": pred_mask,
        }, ok

    def _build_think_subtag_mask_from_loss_1d(
        self,
        ids_resp_1d: torch.Tensor,                # (L,)
        loss_mask_resp_1d: torch.Tensor,          # (L,)
        eor_mask_resp_1d: torch.Tensor,           # (L,)
        think_verifier_scores: Optional[List[float]] = None,
        include_tags: bool = True,
        debug: bool = False,
        score_eps: float = 1e-8,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Build masks for <think> and its subtags over a single sequence using
        loss_mask + end_of_response_position_mask.

        Returns:
            think_mask, obs_mask, reason_mask, pred_mask
            (all (L,) bool tensors).
        """
        device = ids_resp_1d.device
        L = ids_resp_1d.size(0)

        think_mask  = torch.zeros(L, dtype=torch.bool, device=device)
        obs_mask    = torch.zeros(L, dtype=torch.bool, device=device)
        reason_mask = torch.zeros(L, dtype=torch.bool, device=device)
        pred_mask   = torch.zeros(L, dtype=torch.bool, device=device)

        loss_b = loss_mask_resp_1d.to(torch.bool)
        eor_b  = eor_mask_resp_1d.to(torch.bool)

        resp_end_indices = torch.nonzero(eor_b, as_tuple=True)[0].tolist()
        if debug:
            print(f"[think-mask][1d] num_responses={len(resp_end_indices)}")

        if not resp_end_indices:
            return think_mask, obs_mask, reason_mask, pred_mask

        turn_idx = 0
        verifier_active = think_verifier_scores is not None

        for end_pos in resp_end_indices:
            # walk backward through contiguous loss_mask==1
            start_pos = end_pos
            while start_pos > 0 and loss_b[start_pos - 1]:
                start_pos -= 1

            if not loss_b[start_pos : end_pos + 1].any():
                if debug:
                    print(
                        f"[think-mask][1d] end_pos={end_pos}: "
                        f"no loss tokens in span [{start_pos}, {end_pos}]"
                    )
                continue

            span_len = end_pos - start_pos + 1
            ids_span = ids_resp_1d[start_pos : end_pos + 1]

            score = None
            if verifier_active and turn_idx < len(think_verifier_scores):
                score = float(think_verifier_scores[turn_idx])

            local_think  = torch.zeros(span_len, dtype=torch.bool, device=device)
            local_obs    = torch.zeros(span_len, dtype=torch.bool, device=device)
            local_reason = torch.zeros(span_len, dtype=torch.bool, device=device)
            local_pred   = torch.zeros(span_len, dtype=torch.bool, device=device)

            # if not verifier_active:
            #     # NEW: verifier OFF → skip parsing entirely (all local masks stay zero)
            #     if debug:
            #         print(
            #             f"[think-mask][1d] turn={turn_idx}: "
            #             f"verifier inactive → skip <think>/<obs>/<reason>/<pred> parsing."
            #         )
            #     ok = False
            # elif (score is not None) and abs(score) <= score_eps:
            if (score is not None) and abs(score) <= score_eps:
                # cheap full-span think; subtags remain 0 (score≈0 anyway)
                local_think[:] = True
                ok = True
                if debug:
                    print(
                        f"[think-mask][1d] turn={turn_idx}, score≈0 → full-span think."
                    )
            else:
                sub_masks, ok = self._build_subtag_masks_for_response(
                    ids_span,
                    include_tags=include_tags,
                    debug=debug,
                )
                local_think  = sub_masks["think"]
                local_obs    = sub_masks["observation"]
                local_reason = sub_masks["reasoning"]
                local_pred   = sub_masks["prediction"]

                # For non-zero scores, require a think span; otherwise full-span fallback.
                if (score is not None) and (score > score_eps) and not ok:
                    if debug:
                        print(
                            f"[think-mask][1d][WARN] turn={turn_idx}, score={score:.2f} "
                            f"but no well-formed <think> span; falling back to full-span."
                        )
                    local_think = torch.ones(span_len, dtype=torch.bool, device=device)
                    # subtags stay zero—we don't know the structure

            if local_think.any():
                think_mask[start_pos : end_pos + 1] |= local_think
            if local_obs.any():
                obs_mask[start_pos : end_pos + 1] |= local_obs
            if local_reason.any():
                reason_mask[start_pos : end_pos + 1] |= local_reason
            if local_pred.any():
                pred_mask[start_pos : end_pos + 1] |= local_pred

            if debug:
                print(
                    f"[think-mask][1d] turn={turn_idx}, span=[{start_pos}, {end_pos}], "
                    f"len={span_len}, think_tokens={int(local_think.sum().item())}, score={score}"
                )

            turn_idx += 1

        return think_mask, obs_mask, reason_mask, pred_mask


    def _build_think_mask_from_loss_1d(
        self,
        ids_resp_1d: torch.Tensor,                # (L,)
        loss_mask_resp_1d: torch.Tensor,          # (L,)
        eor_mask_resp_1d: torch.Tensor,           # (L,)
        think_verifier_scores: Optional[List[float]] = None,
        include_tags: bool = True,
        debug: bool = False,
        score_eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Build a think mask for a *single* sequence using loss_mask + end_of_response_position_mask.

        - Splits the sequence into response spans using eor_mask + contiguous loss_mask.
        - For each response turn:
            * if score≈0 and scores are provided: full response span is 'think' (cheap, harmless).
            * else: extract <think>...</think> via ID-first, char-fallback.

        Args:
            ids_resp_1d: tokens for full conversation (prompt+responses), after compute_loss_mask.
            loss_mask_resp_1d: 1 where we train (assistant response tokens).
            eor_mask_resp_1d: 1 at last token of each response.
            think_verifier_scores: list of float scores per turn, or None if verifier disabled.
            include_tags: include '<think>' and '</think>' tokens in the mask.
            debug: print debug info.
            score_eps: small threshold for "score≈0".

        Returns:
            think_mask: (L,) bool tensor, True where token is part of some <think>...</think>.
        """
        device = ids_resp_1d.device
        L = ids_resp_1d.size(0)

        think_mask = torch.zeros(L, dtype=torch.bool, device=device)

        loss_b = loss_mask_resp_1d.to(torch.bool)
        eor_b = eor_mask_resp_1d.to(torch.bool)

        resp_end_indices = torch.nonzero(eor_b, as_tuple=True)[0].tolist()
        if debug:
            print(f"[think-mask][1d] num_responses={len(resp_end_indices)}")

        if not resp_end_indices:
            return think_mask

        turn_idx = 0

        for end_pos in resp_end_indices:
            # walk backward through contiguous loss_mask==1
            start_pos = end_pos
            while start_pos > 0 and loss_b[start_pos - 1]:
                start_pos -= 1

            if not loss_b[start_pos : end_pos + 1].any():
                if debug:
                    print(f"[think-mask][1d] end_pos={end_pos}: "
                          f"no loss tokens in span [{start_pos}, {end_pos}]")
                continue

            span_len = end_pos - start_pos + 1
            ids_span = ids_resp_1d[start_pos : end_pos + 1]

            # optional per-turn score
            score = None
            if think_verifier_scores is not None and turn_idx < len(think_verifier_scores):
                score = float(think_verifier_scores[turn_idx])

            if score is not None and abs(score) <= score_eps:
                # zero-score turn: cheap full-span mask (gradients will be zero anyway)
                local_mask = torch.ones(span_len, dtype=torch.bool, device=device)
                ok = True
                if debug:
                    print(f"[think-mask][1d] turn={turn_idx}, score≈0 → full-span think.")
            else:
                local_mask, ok = self._build_think_mask_for_response(
                    ids_span,
                    include_tags=include_tags,
                    debug=debug,
                )

                # For non-zero scores, require a think span; otherwise fall back & warn.
                if score is not None and (score > score_eps) and not ok:
                    if debug:
                        print(
                            f"[think-mask][1d][WARN] turn={turn_idx}, score={score:.2f} "
                            f"but no well-formed <think> span; falling back to full-span."
                        )
                    local_mask = torch.ones(span_len, dtype=torch.bool, device=device)

                # If score is None and ok==False, we just leave local_mask as zeros.
                # (No shaping; mask is only used for analysis / optional signals.)

            # Write local_mask into global think_mask
            if local_mask.any():
                think_mask[start_pos : end_pos + 1] |= local_mask

            if debug:
                print(
                    f"[think-mask][1d] turn={turn_idx}, span=[{start_pos}, {end_pos}], "
                    f"len={span_len}, think_tokens={int(local_mask.sum().item())}, score={score}"
                )

            turn_idx += 1

        return think_mask

    # Another version of _build_think_mask_from_loss_1d in which we skip parsing completely when think_verifier_scores is None
    def _build_think_mask_from_loss_1d_skip_when_no_scores(
        self,
        ids_resp_1d: torch.Tensor,                # (L,)
        loss_mask_resp_1d: torch.Tensor,          # (L,)
        eor_mask_resp_1d: torch.Tensor,           # (L,)
        think_verifier_scores: Optional[List[float]] = None,
        include_tags: bool = True,
        debug: bool = False,
        score_eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Build a think mask for a *single* sequence using loss_mask + end_of_response_position_mask.

        - Splits the sequence into response spans using eor_mask + contiguous loss_mask.
        - For each response turn:
            * if score≈0 and scores are provided: full response span is 'think' (cheap, harmless).
            * else: extract <think>...</think> via ID-first, char-fallback.

        Args:
            ids_resp_1d: tokens for full conversation (prompt+responses), after compute_loss_mask.
            loss_mask_resp_1d: 1 where we train (assistant response tokens).
            eor_mask_resp_1d: 1 at last token of each response.
            think_verifier_scores: list of float scores per turn, or None if verifier disabled.
            include_tags: include '<think>' and '</think>' tokens in the mask.
            debug: print debug info.
            score_eps: small threshold for "score≈0".

        Returns:
            think_mask: (L,) bool tensor, True where token is part of some <think>...</think>.
        """
        device = ids_resp_1d.device
        L = ids_resp_1d.size(0)

        think_mask = torch.zeros(L, dtype=torch.bool, device=device)

        loss_b = loss_mask_resp_1d.to(torch.bool)
        eor_b = eor_mask_resp_1d.to(torch.bool)

        resp_end_indices = torch.nonzero(eor_b, as_tuple=True)[0].tolist()
        if debug:
            print(f"[think-mask][1d] num_responses={len(resp_end_indices)}")

        if not resp_end_indices:
            return think_mask

        # NEW: single flag to tell if verifier is active
        verifier_active = think_verifier_scores is not None

        turn_idx = 0

        for end_pos in resp_end_indices:
            # walk backward through contiguous loss_mask==1
            start_pos = end_pos
            while start_pos > 0 and loss_b[start_pos - 1]:
                start_pos -= 1

            if not loss_b[start_pos : end_pos + 1].any():
                if debug:
                    print(
                        f"[think-mask][1d] end_pos={end_pos}: "
                        f"no loss tokens in span [{start_pos}, {end_pos}]"
                    )
                continue

            span_len = end_pos - start_pos + 1
            ids_span = ids_resp_1d[start_pos : end_pos + 1]

            # local mask defaults to all-zeros
            local_mask = torch.zeros(span_len, dtype=torch.bool, device=device)
            ok = False

            # optional per-turn score
            score = None
            if verifier_active and turn_idx < len(think_verifier_scores):
                score = float(think_verifier_scores[turn_idx])

            # NEW: if verifier is OFF, skip expensive parsing entirely
            if not verifier_active:
                if debug:
                    print(
                        f"[think-mask][1d] turn={turn_idx}: "
                        f"verifier inactive → skip think-span parsing (mask stays zero)."
                    )
                # local_mask remains zeros; ok stays False
            elif score is not None and abs(score) <= score_eps:
                # zero-score turn: cheap full-span mask (gradients will be zero anyway)
                local_mask = torch.ones(span_len, dtype=torch.bool, device=device)
                ok = True
                if debug:
                    print(
                        f"[think-mask][1d] turn={turn_idx}, score≈0 "
                        f"→ full-span think."
                    )
            else:
                local_mask, ok = self._build_think_mask_for_response(
                    ids_span,
                    include_tags=include_tags,
                    debug=debug,
                )

                # For non-zero scores, require a think span; otherwise fall back & warn.
                if score is not None and (score > score_eps) and not ok:
                    if debug:
                        print(
                            f"[think-mask][1d][WARN] turn={turn_idx}, score={score:.2f} "
                            f"but no well-formed <think> span; falling back to full-span."
                        )
                    local_mask = torch.ones(span_len, dtype=torch.bool, device=device)

                # If score is None and ok==False, we just leave local_mask as zeros.
                # (No shaping; mask is only used for analysis / optional signals.)

            # Write local_mask into global think_mask
            if local_mask.any():
                think_mask[start_pos : end_pos + 1] |= local_mask

            if debug:
                print(
                    f"[think-mask][1d] turn={turn_idx}, span=[{start_pos}, {end_pos}], "
                    f"len={span_len}, think_tokens={int(local_mask.sum().item())}, score={score}"
                )

            turn_idx += 1

        return think_mask


    def _build_span_mask(
        self,
        ids_resp_1d: torch.Tensor,
        tokenizer,
        loss_mask_1d: Optional[torch.Tensor] = None,
        start_str: str = "<think>",
        end_str: str = "</think>",
        debug: bool = True,
    ) -> torch.Tensor:
        """
        Build a mask for tokens inside <think>...</think> within the assistant's response region.

        Args:
            ids_resp_1d: (L,) tensor of token ids.
            tokenizer: HF tokenizer.
            loss_mask_1d: (L,) tensor; 1 where the assistant response tokens live (from compute_loss_mask).
            start_str: tag marking start of think region.
            end_str: tag marking end of think region.
            debug: if True, prints debug info.

        Returns:
            think_mask: (L,) float tensor on same device as ids_resp_1d.
        """
        device = ids_resp_1d.device
        L = ids_resp_1d.size(0)

        # 0) Init mask
        think_mask = torch.zeros(L, dtype=torch.float32, device=device)

        # 1) Determine the assistant response span using loss_mask
        if loss_mask_1d is not None:
            lm = loss_mask_1d.bool()
            active = torch.nonzero(lm, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                # No response tokens → nothing to mark
                if debug:
                    print("[think-mask] Empty loss_mask; returning all-zero think_mask.")
                return think_mask
            resp_start = int(active[0].item())
            resp_end   = int(active[-1].item())
        else:
            # If no loss_mask, treat the whole sequence as candidate region
            resp_start, resp_end = 0, L - 1

        if resp_start >= resp_end:
            if debug:
                print(f"[think-mask] Degenerate resp span [{resp_start}, {resp_end}]; returning zeros.")
            return think_mask

        ids_sub = ids_resp_1d[resp_start : resp_end + 1]  # (S,)

        if debug:
            print(f"[think-mask] L={L}, response span=[{resp_start}, {resp_end}], S={ids_sub.size(0)}")

        # 2) Try ID-based matching first
        #    Note: encode() gives canonical segmentation, which may or may not match generated segmentation.
        start_ids = tokenizer.encode(start_str, add_special_tokens=False)
        end_ids   = tokenizer.encode(end_str, add_special_tokens=False)

        if debug:
            print(f"[think-mask] start_ids={start_ids}, end_ids={end_ids}")
            # Decode just to sanity-check
            if start_ids:
                print(f"[think-mask] decode(start_ids)={tokenizer.decode(start_ids)}")
            if end_ids:
                print(f"[think-mask] decode(end_ids)={tokenizer.decode(end_ids)}")

        if start_ids and end_ids:
            start_ids_t = torch.tensor(start_ids, dtype=ids_resp_1d.dtype, device=device)
            end_ids_t   = torch.tensor(end_ids,   dtype=ids_resp_1d.dtype, device=device)

            start_pos_rel = self._find_pattern_positions(ids_sub, start_ids_t)
            end_pos_rel   = self._find_pattern_positions(ids_sub, end_ids_t)

            if debug:
                print(f"[think-mask] ID-based start_pos_rel={start_pos_rel}")
                print(f"[think-mask] ID-based end_pos_rel={end_pos_rel}")

            if len(start_pos_rel) > 0 and len(end_pos_rel) > 0:
                # Use the outermost pair: first <think>, last </think>
                first_start_rel = start_pos_rel[0]
                last_end_rel    = end_pos_rel[-1]

                # Global positions
                first_start = resp_start + first_start_rel
                last_end    = resp_start + last_end_rel

                # # Tokens inside <think>...</think>, *excluding* the tag tokens themselves.
                # inner_start = first_start + len(start_ids)  # skip start tag tokens
                # inner_end   = last_end                      # first token of end tag (exclusive)

                # Include both <think> and </think> tag tokens
                inner_start = first_start                    # start at '<think>'
                inner_end = last_end + len(end_ids)      # go past '</think>' tokens

                # safety clamp
                inner_end = min(inner_end, L)

                if debug:
                    print(f"[think-mask] ID-based outer span: [{first_start}, {last_end}]")
                    print(f"[think-mask] ID-based inner span (think only): [{inner_start}, {inner_end})")

                if 0 <= inner_start < inner_end <= L:
                    think_mask[inner_start:inner_end] = 1.0
                    return think_mask
                else:
                    if debug:
                        print("[think-mask] ID-based inner span invalid; falling back to char-level.")

        # 3) Char-level fallback: robust to BPE differences and nested tags
        #    - Decode the response region once
        #    - Take from first '<think>' to last '</think>'
        #    - Map char span back to tokens via per-token decode length

        ids_sub_cpu = ids_sub.detach().cpu()
        full_text = tokenizer.decode(ids_sub_cpu.tolist())

        s_char = full_text.find(start_str)
        e_char_start = full_text.rfind(end_str)  # index of '<' in '</think>'

        if debug:
            print(f"[think-mask] char-level: s_char={s_char}, e_char_start={e_char_start}")

        if s_char == -1 or e_char_start == -1 or s_char >= e_char_start:
            # No proper <think>...</think> in text
            if debug:
                print("[think-mask] No valid '<think> ... </think>' span found in decoded text; think_mask remains zero.")
            return think_mask

        # Char indices *inside* the think span:
        #   <think> ... </think>
        #        ^   ^
        # inner_start_char = s_char + len(start_str)
        # inner_end_char   = e_char_start           # start of "</think>"

        inner_start_char = s_char                          # from '<' of "<think>"
        inner_end_char   = e_char_start + len(end_str)     # past '>' of "</think>"

        if debug:
            snippet = full_text[max(0, s_char - 40) : e_char_start + len(end_str) + 40]
            print(f"[think-mask] full_text snippet around think-span:\n{snippet}")
            print(f"[think-mask] inner char range=[{inner_start_char}, {inner_end_char})")

        # Now we approximate char → token mapping by decoding each token individually,
        # taking cumulative character lengths.
        # Assumption: "decoded single tokens concatenated" ≈ "decoded full sequence".
        # This holds for most HF BPE tokenizers in practice.
        token_pieces = [tokenizer.decode([int(tid)]) for tid in ids_sub_cpu.tolist()]
        char_offsets = [0]
        acc = 0
        for piece in token_pieces:
            acc += len(piece)
            char_offsets.append(acc)
        # char_offsets[i] .. char_offsets[i+1] is the char span for token i in full_text approximation.

        S = ids_sub.size(0)
        for i in range(S):
            tok_s = char_offsets[i]
            tok_e = char_offsets[i + 1]

            # Overlap check: token overlaps [inner_start_char, inner_end_char)
            if tok_e <= inner_start_char or tok_s >= inner_end_char:
                continue

            global_idx = resp_start + i
            if 0 <= global_idx < L:
                think_mask[global_idx] = 1.0

        if debug:
            n_think = int(think_mask.sum().item())
            print(f"[think-mask] char-level fallback marked {n_think} tokens as think.")

        return think_mask


    @torch.no_grad()
    def reset(self, mini_batch:DataProto):
        """
        Reset environments based on provided configurations, reusing environments when possible.
        - For env with same config and env_name, reuse the same environment (reset)
        - For env with different config or env_name, close the old environment and create a new one
        - Reset the recorder

        Args:
            env_configs: List of environment configurations containing env_name, config, and seed

        Returns:
            Initial observations and info from all environments
        """
        # Step 1: Sort environments into buckets by env_name and config
        # Try to reuse environemnts with the same config and env_name
        env_configs = [
                mini_batch.non_tensor_batch['extra_info'][i]
                for i in range(len(mini_batch))
            ]
        env_buckets = defaultdict(set)

        if self.envs is None:
            self.envs = {} # This is now id:config_instance

        for env_id, env_config_instance in self.envs.items():
            env_config_id = env_config_instance.config_id()
            bucket_key = env_config_id
            env_buckets[bucket_key].add(env_id)

        # Step1. collect envs which need to be reset and new env configs
        ids2seeds_reset = {}
        configs_to_create=[]
        for i, cfg in enumerate(env_configs):
            # Create bucket key
            config_instance= REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
            env_config_id = config_instance.config_id()
            bucket_key = env_config_id

            # Check if we have an available environment with the same config
            if bucket_key in env_buckets and env_buckets[bucket_key]:
                old_env_id = env_buckets[bucket_key].pop()
                ids2seeds_reset[old_env_id] = cfg["seed"]
            else:
                # don't initialize the environment here, close unused environments first
                configs_to_create.append(cfg)

        # Step 2: Collect ids which need to be closed
        ids_to_close=[]
        # Close unused environments
        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                ids_to_close.append(env_id)
                self.envs.pop(env_id)

        # Step 3: Close unused environments
        #print(f"[DEBUG] ids_to_close: {ids_to_close}")
        self.env_client.close_batch(ids_to_close)
        # Step 4: Create new environments
        ids2configs_create = {}
        id=0
        for cfg in configs_to_create:
            id+=1
            while self.split+str(id) in self.envs:
                id+=1
            id_str = self.split+str(id)
            ids2configs_create[id_str] = cfg
            ids2seeds_reset[id_str] = cfg["seed"]
            self.envs[id_str] = REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
        #print(f"[DEBUG] ids2configs_create: {ids2configs_create}")
        self.env_client.create_environments_batch(ids2configs_create)
        # Step 5: Reset environments
        #print(f"[DEBUG] ids2seeds_reset: {ids2seeds_reset}")
        reset_results=self.env_client.reset_batch(ids2seeds_reset)


        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        self.rollout_analysis_cache = defaultdict(list)
        initial_obs = {}
        initial_info = {}


        for env_id, rst in reset_results.items():
            obs, info = rst
            initial_obs[env_id] = obs
            initial_info[env_id] = info
            self.record(
                env_id,
                obs=obs,
                reward=0,
                done=False,
                info=info
            )

        self.env_states = {env_id: {'step': 0, 'done': False,'metrics':{"turn_metrics":defaultdict(list),"traj_metrics":{}}} for env_id in self.envs}
        self.system_prompts=self.env_client.get_system_prompts_batch(list(self.envs.keys()))
        return initial_obs, initial_info

    @torch.no_grad()
    def _clone_rollout_artifact_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return copy.deepcopy(value)

    def _find_last_user_turn_start(self, prompt_ids) -> int:
        """Find the token position where the last user turn begins in the prompt.

        In our multi-turn rollout setting, the prompt is a chat conversation:
            <|im_start|>system\n{sys}<|im_end|>\n
            <|im_start|>user\n{obs_0}<|im_end|>\n
            <|im_start|>assistant\n{resp_0}<|im_end|>\n
            ...
            <|im_start|>user\n{obs_t}<|im_end|>\n
            <|im_start|>assistant\n                      ← generation prompt

        The boundary is the ``<|im_start|>`` token of the **last user turn**.
        Everything before it is P_hist (system + prior user/assistant pairs).
        Everything from it onward (within the P segment) is P_curr (the
        current turn's user message + generation prompt).

        This approach is environment-agnostic: it relies on the chat template
        structure rather than env-specific anchor text.  Even though the
        current user message may embed a condensed action history (e.g.
        SciWorld's ``Prior to this step ...``), that summary is part of
        the *current observation context* delivered to the model and is
        correctly classified as P_curr.

        Returns the token index (0-based, relative to the start of
        ``prompt_ids``).  Returns 0 if no user turn boundary is found.
        """
        ids_list = prompt_ids.tolist() if isinstance(prompt_ids, torch.Tensor) else list(prompt_ids)

        # Get <|im_start|> token ID
        im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        if im_start_id is None or im_start_id == self.tokenizer.unk_token_id:
            # Fallback: encode the special token string
            encoded = self.tokenizer.encode('<|im_start|>', add_special_tokens=False)
            im_start_id = encoded[0] if encoded else None

        if im_start_id is None:
            return 0

        # Encode the user role marker that follows <|im_start|>
        user_marker_ids = self.tokenizer.encode('user\n', add_special_tokens=False)
        marker_len = len(user_marker_ids)

        # Search backward for the last <|im_start|> followed by "user\n"
        for i in range(len(ids_list) - 1, -1, -1):
            if ids_list[i] == im_start_id:
                if i + 1 + marker_len <= len(ids_list):
                    if ids_list[i + 1: i + 1 + marker_len] == user_marker_ids:
                        return i

        return 0  # no user turn found → all P is current

    @torch.no_grad()
    def _build_single_turn_rollout_masks(
        self,
        response_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        response_loss_mask = response_attention_mask.to(torch.bool)
        response_eor_mask = torch.zeros_like(response_loss_mask, dtype=torch.bool)

        valid_positions = torch.nonzero(response_loss_mask, as_tuple=False).squeeze(-1)
        if valid_positions.numel() > 0:
            response_eor_mask[int(valid_positions[-1].item())] = True

        (
            think_mask_response,
            obs_mask_response,
            reason_mask_response,
            pred_mask_response,
        ) = self._build_think_subtag_mask_from_loss_1d(
            ids_resp_1d=response_ids,
            loss_mask_resp_1d=response_loss_mask,
            eor_mask_resp_1d=response_eor_mask,
            think_verifier_scores=None,
            include_tags=True,
            debug=False,
        )

        think_mask_response &= response_loss_mask
        obs_mask_response &= response_loss_mask
        reason_mask_response &= response_loss_mask
        pred_mask_response &= response_loss_mask

        return (
            response_loss_mask,
            response_eor_mask,
            think_mask_response,
            obs_mask_response,
            reason_mask_response,
            pred_mask_response,
        )

    @torch.no_grad()
    def _cache_rollout_analysis_artifact(
        self,
        env_id,
        step: int,
        window_size: Optional[int],
        batch_idx: int,
        gen_batch: DataProto,
        output_batch: DataProto,
    ) -> None:
        if self.rollout_analysis_cache is None:
            self.rollout_analysis_cache = defaultdict(list)

        sample = {
            'env_id': env_id,
            'config_id': self.envs[env_id].config_id() if env_id in self.envs else '',
            'turn': int(step),
            'window_size': window_size,
            'window_start_turn': max(0, step - window_size) if window_size is not None else 0,
            'prompts': self._clone_rollout_artifact_value(output_batch.batch['prompts'][batch_idx]),
            'responses': self._clone_rollout_artifact_value(output_batch.batch['responses'][batch_idx]),
            'input_ids': self._clone_rollout_artifact_value(output_batch.batch['input_ids'][batch_idx]),
            'attention_mask': self._clone_rollout_artifact_value(output_batch.batch['attention_mask'][batch_idx]),
            'position_ids': self._clone_rollout_artifact_value(output_batch.batch['position_ids'][batch_idx]),
            'raw_prompt_ids': self._clone_rollout_artifact_value(gen_batch.non_tensor_batch['raw_prompt_ids'][batch_idx]),
        }

        if 'multi_modal_data' in gen_batch.non_tensor_batch:
            sample['multi_modal_data'] = self._clone_rollout_artifact_value(
                gen_batch.non_tensor_batch['multi_modal_data'][batch_idx]
            )

        self.rollout_analysis_cache[env_id].append(sample)

    @torch.no_grad()
    def _build_single_turn_analysis_row(self, cache_entry: Dict) -> Dict:
        """Build a single-sample row dict (not DataProto) for later collation.

        Returns a dict whose tensor values are 1-D (or 2-D for position_ids
        with mrope).  The caller pads all rows to a common length and stacks
        them via collate_fn → DataProto.from_single_dict.

        IMPORTANT: The generation pipeline stores dummy tensors for prompts,
        input_ids, attention_mask, and position_ids (each just ``[0]``)
        because vLLM only uses ``raw_prompt_ids`` for actual generation.
        We reconstruct the proper full-sequence tensors here from
        ``raw_prompt_ids`` (real prompt token IDs) + ``responses`` (right-
        padded generated tokens).
        """
        # --- Reconstruct full-sequence tensors from raw_prompt_ids + responses ---
        raw_prompt_ids = cache_entry['raw_prompt_ids']
        if isinstance(raw_prompt_ids, torch.Tensor):
            prompt_ids = raw_prompt_ids.long()
        else:
            prompt_ids = torch.tensor(raw_prompt_ids, dtype=torch.long)

        responses = cache_entry['responses'].long()          # (config.response_length,)
        response_length = responses.shape[-1]

        # Position IDs: use mrope (3, seq_len) for Qwen2-VL with images,
        # otherwise simple cumsum (seq_len,).
        has_images = (
            'multi_modal_data' in cache_entry
            and self.processor is not None
            and isinstance(cache_entry['multi_modal_data'], dict)
            and len(cache_entry['multi_modal_data'].get('image', [])) > 0
        )
        if has_images:
            image_list = cache_entry['multi_modal_data']['image']
            image_inputs = self.processor.image_processor(image_list, return_tensors='pt')
            image_grid_thw = image_inputs.get('image_grid_thw', None)
        else:
            image_grid_thw = None

        # ------------------------------------------------------------------
        # Expand prompt_ids so each image region has the correct number of
        # <|image_pad|> tokens.  raw_prompt_ids (from the vLLM rollout path)
        # contains exactly 1 <|image_pad|> per image, but get_rope_index and
        # the HF model expect  (t*h*w // merge_size**2)  <|image_pad|> tokens
        # per image.  Without expansion the position_ids shape will not match
        # input_ids, causing a broadcasting error.
        # ------------------------------------------------------------------
        if has_images and image_grid_thw is not None:
            image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            merge_length = self.processor.image_processor.merge_size ** 2
            ids_list = prompt_ids.tolist()
            expanded: list = []
            img_idx = 0
            for tok in ids_list:
                if tok == image_pad_id and img_idx < len(image_grid_thw):
                    n_visual = int(image_grid_thw[img_idx].prod().item() // merge_length)
                    expanded.extend([image_pad_id] * n_visual)
                    img_idx += 1
                else:
                    expanded.append(tok)
            prompt_ids = torch.tensor(expanded, dtype=torch.long)

        prompt_length = prompt_ids.shape[-1]

        # Full input_ids = real prompt + padded response
        input_ids = torch.cat([prompt_ids, responses])       # (prompt_len + resp_len,)

        # Response attention mask: extract from cached attention_mask.
        # The cached attention_mask is [dummy_0 | response_eos_mask] (shape 1+resp_len)
        # because prompts in gen batch were dummy [0].
        cached_attn_mask = cache_entry['attention_mask'].long()
        if cached_attn_mask.shape[-1] == 1 + response_length:
            response_attn_mask = cached_attn_mask[-response_length:]
        else:
            # Fallback: derive from responses — 1 up to and including EOS, 0 after
            eos_id = self.tokenizer.eos_token_id
            resp_np = responses.numpy() if not responses.is_cuda else responses.cpu().numpy()
            eos_positions = (resp_np == eos_id).nonzero()[0]
            if len(eos_positions) > 0:
                valid_len = int(eos_positions[0]) + 1
            else:
                valid_len = response_length
            response_attn_mask = torch.zeros(response_length, dtype=torch.long)
            response_attn_mask[:valid_len] = 1

        # Full attention mask: all 1s for prompt + response_eos_mask
        prompt_attn_mask = torch.ones(prompt_length, dtype=torch.long)
        attention_mask = torch.cat([prompt_attn_mask, response_attn_mask])

        if has_images and image_grid_thw is not None:
            from verl.models.transformers.qwen2_vl import get_rope_index
            # get_rope_index expects 1-D input_ids and returns (3, seq_len)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )                                                # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(
                attention_mask.unsqueeze(0)
            ).squeeze(0)                                     # (seq_len,)

        # Response attention mask (bool) for mask derivation
        response_attention_mask_bool = response_attn_mask.to(torch.bool)

        (
            response_loss_mask,
            response_eor_mask,
            think_mask_response,
            obs_mask_response,
            reason_mask_response,
            pred_mask_response,
        ) = self._build_single_turn_rollout_masks(responses.clone(), response_attention_mask_bool)

        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        loss_mask[prompt_length:prompt_length + response_length] = response_loss_mask.to(torch.long)

        end_of_response_position_mask = torch.zeros_like(input_ids, dtype=torch.long)
        valid_positions = torch.nonzero(response_loss_mask, as_tuple=False).squeeze(-1)
        if valid_positions.numel() > 0:
            end_pos = prompt_length + int(valid_positions[-1].item())
            end_of_response_position_mask[end_pos] = 1

        row: Dict = {
            'prompts': prompt_ids,           # (prompt_len,) — real prompt tokens
            'responses': responses,          # (response_len,)
            'input_ids': input_ids,          # (prompt_len + resp_len,)
            'attention_mask': attention_mask, # (prompt_len + resp_len,)
            'position_ids': position_ids,    # (seq_len,) or (3, seq_len) for mrope
            'loss_mask': loss_mask,           # (prompt_len + resp_len,)
            'end_of_response_position_mask': end_of_response_position_mask,
            'think_mask': think_mask_response.to(torch.float32),  # (response_len,)
            'obs_mask': obs_mask_response.to(torch.float32),
            'reason_mask': reason_mask_response.to(torch.float32),
            'pred_mask': pred_mask_response.to(torch.float32),
            'truncated_turns': torch.zeros((), dtype=torch.long),
            # Metadata stored as plain Python values — collate_fn puts them
            # into non_tensor_batch automatically.
            'env_id': cache_entry['env_id'],
            'config_id': cache_entry.get('config_id', ''),
            'turn': cache_entry['turn'],
            'window_size': cache_entry.get('window_size'),
            'window_start_turn': cache_entry.get('window_start_turn', 0),
            'raw_prompt_ids': cache_entry.get('raw_prompt_ids'),
            # Token index in prompt_ids where the last user turn starts.
            # Used to split P into P_hist / P_curr for attention analysis.
            # P_hist = system + prior user/assistant chat turns.
            # P_curr = last user turn (current observation) + generation prompt.
            'obs_boundary_pos': self._find_last_user_turn_start(prompt_ids),
        }

        if 'multi_modal_data' in cache_entry:
            row['multi_modal_data'] = cache_entry['multi_modal_data']
            if self.processor is not None:
                image_bundle = cache_entry['multi_modal_data']
                image_list = image_bundle.get('image', []) if isinstance(image_bundle, dict) else []
                if image_list:
                    # Reuse image_inputs from mrope computation if available
                    if image_grid_thw is None:
                        image_inputs = self.processor.image_processor(image_list, return_tensors='pt')
                    row['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

        return row

    @torch.no_grad()
    def generate_rollout_analysis_batches(self) -> List[DataProto]:
        """Build collated DataProto batch(es) from cached per-turn rollout artifacts.

        Returns a list with 0 or 1 DataProto elements.  All single-turn rows
        are left-padded to the maximum sequence length, collated, and returned
        as one batch so that a single compute_log_prob / compute_attention_mass
        RPC can process them all, avoiding the O(N) dispatch overhead of
        sending one sample at a time.
        """
        if not self.rollout_analysis_cache:
            return []

        rows = []
        for env_id in self.envs.keys():
            for cache_entry in self.rollout_analysis_cache.get(env_id, []):
                rows.append(self._build_single_turn_analysis_row(cache_entry))

        if not rows:
            return []

        # ----- Determine max lengths for padding -----
        max_seq_len = max(r['input_ids'].shape[-1] for r in rows)
        max_resp_len = max(r['responses'].shape[-1] for r in rows)
        max_prompt_len = max(r['prompts'].shape[-1] for r in rows)

        # ----- Left-pad each row to common shape and collect -----
        padded_rows = []
        for row in rows:
            seq_len = row['input_ids'].shape[-1]
            resp_len = row['responses'].shape[-1]
            prompt_len = row['prompts'].shape[-1]

            seq_pad = max_seq_len - seq_len
            resp_pad = max_resp_len - resp_len
            prompt_pad = max_prompt_len - prompt_len

            def _left_pad_1d(t, pad_size, value=0):
                if pad_size == 0:
                    return t
                return torch.cat([torch.full((pad_size,), value, dtype=t.dtype), t])

            def _left_pad_pos(t, pad_size):
                """Left-pad position_ids; handles both (seq,) and (3, seq) mrope."""
                if pad_size == 0:
                    return t
                if t.dim() == 1:
                    return torch.cat([torch.zeros(pad_size, dtype=t.dtype), t])
                # mrope: (3, seq)
                pad_block = torch.zeros(t.shape[0], pad_size, dtype=t.dtype)
                return torch.cat([pad_block, t], dim=1)

            padded: Dict = {
                'input_ids': _left_pad_1d(row['input_ids'], seq_pad, value=self.tokenizer.pad_token_id),
                'attention_mask': _left_pad_1d(row['attention_mask'], seq_pad, value=0),
                'position_ids': _left_pad_pos(row['position_ids'], seq_pad),
                'loss_mask': _left_pad_1d(row['loss_mask'], seq_pad, value=0),
                'end_of_response_position_mask': _left_pad_1d(row['end_of_response_position_mask'], seq_pad, value=0),
                'prompts': _left_pad_1d(row['prompts'], prompt_pad, value=self.tokenizer.pad_token_id),
                'responses': _left_pad_1d(row['responses'], resp_pad, value=self.tokenizer.pad_token_id),
                'think_mask': _left_pad_1d(row['think_mask'], resp_pad, value=0),
                'obs_mask': _left_pad_1d(row['obs_mask'], resp_pad, value=0),
                'reason_mask': _left_pad_1d(row['reason_mask'], resp_pad, value=0),
                'pred_mask': _left_pad_1d(row['pred_mask'], resp_pad, value=0),
                'truncated_turns': row['truncated_turns'],
            }

            # Non-tensor metadata
            for key in ('env_id', 'config_id', 'turn', 'window_size',
                        'window_start_turn', 'raw_prompt_ids', 'obs_boundary_pos'):
                padded[key] = row[key]

            if 'multi_modal_data' in row:
                padded['multi_modal_data'] = row['multi_modal_data']
            if 'multi_modal_inputs' in row:
                padded['multi_modal_inputs'] = row['multi_modal_inputs']

            padded_rows.append(padded)

        batch_dict = collate_fn(padded_rows)
        batch = DataProto.from_single_dict(batch_dict)
        return [batch]

    @torch.no_grad()
    def record(self, env_id, obs, reward, done, info):
        """
        Record each step's obs, info, done, reward,
        Please include "llm_raw_response" in info # it will be decoded by rollout manager and pass to env, then should pass back
        """
        # Create a record entry for this step
        assert obs is not None, "obs cannot be None"
        assert info is not None, "info cannot be None"
        assert isinstance(reward, (int, float)), "reward must be a number"
        assert isinstance(done, bool), "done must be a boolean"
        record_entry = {
            'env_id': env_id,
            'done': done,
            'reward': reward,
            'info': info,
            'obs_str': obs['obs_str'],
        }
        image_placeholder = self.envs[env_id].get('image_placeholder', "<image>")
        if 'multi_modal_data' in obs:
            if image_placeholder in obs['multi_modal_data']:
                record_entry['image_data'] = [process_image(image) for image in obs['multi_modal_data'][image_placeholder]]
        self.recorder[env_id].append(record_entry)

    @torch.no_grad()
    def _single_recording_to_prompt(self,
                            recording: List[Dict],
                            step: int,
                            window_size: int = None,
                            is_final: bool = False,
                            prep_for_loss_mask: bool = False,
        ):
        """
        Given a recording, generate the prompt for MLLM
        Chat: Sys -> |InitUser| -> |Assistant, User| -> |Assistant, User| -> ... -> |Assistant, User Final|

        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate prompt for
            window_size: Number of past steps to include in the context
            is_final: Whether the prompt is for the final step
                - if True, the end of the chat is from the last assistant's response
            prep_for_loss_mask: whether to use special token to wrap llm response

        Returns:
            dict: prompt_with_chat_template : str, image_data: list of images, reward: list of reward
        """

        assert step >= 0
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= end_step + 1, 'History length is not enough'
        history = recording[start_step: end_step + 1]
        rewards=[]
        chat = []

        env_id = history[0]['env_id']
        image_placeholder = self.envs[env_id].get('image_placeholder', "<image>")
        chat.append({"role": "system", "content": self.system_prompts[env_id]})

        image_data=[]
        for i, record in enumerate(history):
            if i > 0:
                resp_source = record.get("info", {})
                llm_raw_response = (
                    resp_source.get("llm_rewritten_response")
                    or resp_source.get("llm_raw_response", "")
                )
                # llm_raw_response = record['info']['llm_raw_response']
                filtered_llm_raw_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=prep_for_loss_mask)  # llm_raw_response wrapped with <box_start|> ... <box_end|>
                chat.append({"role": "assistant", "content": filtered_llm_raw_response})
                rewards.append(record['reward'])
            if i < len(history) - 1 or not is_final:
                chat.append({"role": "user", "content": record['obs_str']})
                if 'image_data' in record:
                    for img in record['image_data']:
                        image_data.append(img)
                # user_content, user_images = self._prepare_user_turn_for_prompt(
                #     record=record,
                #     history_index=i,
                #     history_len=len(history),
                #     is_final=is_final,
                #     image_placeholder=image_placeholder,
                # )
                # chat.append({"role": "user", "content": user_content})
                # image_data.extend(user_images)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=(not is_final), tokenize=False)
        if is_final and self.model_type == 'qwen': # NOTE hard coded
            assert prompt_with_chat_template[-1] == '\n', f"The last token should be new line token, got {prompt_with_chat_template[-1]}"
            prompt_with_chat_template = prompt_with_chat_template[:-1] # remove the last in token
        # switch box_end and im_end so that the model can learn to generate <|im_end|>
        prompt_with_chat_template = prompt_with_chat_template.replace(
            f'{self.config.special_token_for_loss_mask[1]}{self.tokenizer.eos_token}',
            f'{self.tokenizer.eos_token}{self.config.special_token_for_loss_mask[1]}') # why?
        return {
            "prompt": prompt_with_chat_template,
            "image_data": image_data,
            "rewards": rewards,  # rewards is [0, 0, 0, 0, 0]
        }

    @torch.no_grad()
    def _generate_input_for_rollout(
            self,
            recording: List[Dict],
            step: int,
            window_size: int = None,
        ):
        """
        Given a recording, generate the input for MLLM

        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context

        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids:
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute
        """
        rst=self._single_recording_to_prompt(recording, step, window_size, is_final=False, prep_for_loss_mask=False)
        prompt_with_chat_template=rst['prompt']
        image_data=rst['image_data']
        has_images = len(image_data) > 0

        row_dict = {}
        if has_images:  # expand image token
            prompt_with_chat_template, row_dict, _, raw_prompt = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=False)
        else:
            raw_prompt = prompt_with_chat_template

        # use random input_ids and attention_mask for vllm only takes raw_prompt_ids as input when generating sequences
        # TODO check if this is correct
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['input_ids'] = torch.tensor([0], dtype=torch.long)
        row_dict['attention_mask'] = torch.tensor([0], dtype=torch.long)
        row_dict['position_ids'] = torch.tensor([0], dtype=torch.long)

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


    @torch.no_grad()
    def _generate_input_for_update(
            self,
            recording: List[Dict],
            step: int,
            window_size: int = None,
        ):
        """
        Given a recording, generate the final input for MLLM

        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context

        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids:
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute

        """



        # handle prompt, prompt=pad_token since we now have everything in response and compute a loss mask for them
        prompt_with_chat_template=self.tokenizer.pad_token

        # handle response
        response_rst=self._single_recording_to_prompt(recording, step, window_size, is_final=True, prep_for_loss_mask=True)
        response_with_chat_template=response_rst['prompt'] # response is formed with this pattern: <|im_start|>system\n ... <|im_end|>\n<|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n<|box_start|><think> ... </think><|im_end|><|box_end|>
        image_data=response_rst['image_data']
        rewards=response_rst['rewards']

        # --- NEW: derive per-turn verifier scores from env_states ---
        env_id = recording[0]['env_id']
        tm = self.env_states[env_id]['metrics']['turn_metrics']
        rubric = "self_consistency"
        vkey = f"{rubric}_verifier_score"

        raw_series = tm.get(vkey, [])
        total_turns = len(rewards)

        # If verifier never ran for this episode (schedule disabled), treat as "no shaping"
        if not raw_series:
            think_verifier_scores = None
        else:
            # Normalize to a Python list of floats
            if isinstance(raw_series, (list, tuple)):
                scores = [float(v) for v in raw_series]
            else:
                # Unexpected scalar case: treat as single-step list
                scores = [float(raw_series)]

            # Align lengths with the number of turns; pad/truncate with 0.0 as needed
            if len(scores) < total_turns:
                scores = scores + [0.0] * (total_turns - len(scores))
            elif len(scores) > total_turns:
                scores = scores[:total_turns]

            think_verifier_scores = scores

        # --- NEW: optional per-subtag scores per turn ---
        subtag_names = ("observation", "reasoning", "prediction")
        subtag_scores_by_turn: dict[str, Optional[List[float]]] = {}

        for sub in subtag_names:
            skey = f"{rubric}_{sub}_verifier_score"
            seq = tm.get(skey, [])
            if not seq:
                subtag_scores_by_turn[sub] = None
                continue
            if isinstance(seq, (list, tuple)):
                scores_sub = [float(v) for v in seq]
            else:
                scores_sub = [float(seq)]
            if len(scores_sub) < total_turns:
                scores_sub = scores_sub + [0.0] * (total_turns - len(scores_sub))
            elif len(scores_sub) > total_turns:
                scores_sub = scores_sub[:total_turns]
            subtag_scores_by_turn[sub] = scores_sub
        # --- END NEW ---

        ################################################################################
        # --- OLD: derive per-turn verifier scores from info in recording ---

        # # align lengths
        # if len(raw_series) < total_turns:
        #     raw_series = raw_series + [None] * (total_turns - len(raw_series))
        # elif len(raw_series) > total_turns:
        #     raw_series = raw_series[:total_turns]

        # think_verifier_scores = None
        # if raw_series:
        #     scores = []
        #     any_real = False
        #     for v in raw_series:
        #         if v is None:
        #             # verifier disabled / not run this step
        #             # TODO: should skip this step instead of assigning 0.0?
        #             scores.append(0.0)
        #         else:
        #             any_real = True
        #             scores.append(float(v))
        #     if any_real:
        #         # Only keep as usable series if we have at least one real verifier score
        #         think_verifier_scores = scores
        ################################################################################

        # # optional: zero-sum shaping of verifier scores within this episode
        # if (
        #     think_verifier_scores is not None
        #     and getattr(self.config, "verifier_zero_sum", False)
        #     and len(think_verifier_scores) > 0
        # ):
        #     # zero-sum shaping of verifier scores within this episode
        #     sv = torch.tensor(think_verifier_scores, dtype=torch.float32)
        #     sv = sv - sv.mean()
        #     think_verifier_scores = sv.tolist()

        # optional: zero-sum shaping of verifier scores within this episode
        if getattr(self.config, "verifier_zero_sum", False):
            # Global think scores
            if think_verifier_scores is not None and len(think_verifier_scores) > 0:
                sv = torch.tensor(think_verifier_scores, dtype=torch.float32)
                sv = sv - sv.mean()
                think_verifier_scores = sv.tolist()

            # NEW: per-subtag series (if present)
            for sub_name in ("observation", "reasoning", "prediction"):
                series = subtag_scores_by_turn.get(sub_name)
                if not series:
                    continue
                # Treat None as 0.0 for centering (keeps length aligned)
                vals = [0.0 if s is None else float(s) for s in series]
                sv = torch.tensor(vals, dtype=torch.float32)
                sv = sv - sv.mean()
                subtag_scores_by_turn[sub_name] = sv.tolist()

        has_images = len(image_data) > 0
        row_dict = {}
        if has_images:  # expand image token
            response_with_chat_template, row_dict, image_grid_thw, _ = self._handle_multi_modal_data(
                response_with_chat_template, row_dict, image_data, do_embedding=True)


        input_ids_response, attention_mask_response = verl_F.tokenize_and_postprocess_data(prompt=response_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.max_trajectory_length-1, # -1 for the prompt padding token
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=False,
                                                                         truncation=self.config.truncation)
        input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=1,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.config.truncation)
        attention_mask_prompt=torch.zeros_like(input_ids_prompt) # All prompt will be masked


        input_ids_response, attention_mask_response, loss_mask_response, end_of_response_position_mask_response = self._compute_loss_mask(input_ids_response, attention_mask_response)
        ## DEBUGs prints
        ################################################################################
        # input_ids_response is tensor([[151644,   8948,    198,  ..., 151643, 151643, 151643]]) with shape torch.Size([1, 4999])
        # attention_mask_response is tensor([[1, 1, 1,  ..., 0, 0, 0]]) with shape torch.Size([1, 4999])
        # loss_mask_response is tensor([[0, 0, 0,  ..., 0, 0, 0]]) with shape torch.Size([1, 4999]) and number of valid tokens to loss is tensor([[699]])
        # end_of_response_position_mask_response is tensor([[0, 0, 0,  ..., 0, 0, 0]]) with shape torch.Size([1, 4999]) and number of eor tokens is tensor([[5]])
        ################################################################################

        # --- Handle truncation: align rewards and scores to actual response positions ---
        num_response_positions = int(end_of_response_position_mask_response.sum().item())
        original_num_turns = len(rewards)
        truncated_turns = 0  # number of early turns removed by left truncation

        if num_response_positions != original_num_turns:
            if num_response_positions < original_num_turns:
                # Truncation removed some turns
                if self.config.truncation == "left":
                    # Left truncation: early turns were removed, keep the last num_response_positions
                    truncated_count = original_num_turns - num_response_positions
                    truncated_turns = truncated_count
                    rewards = rewards[truncated_count:]
                    if think_verifier_scores is not None:
                        think_verifier_scores = think_verifier_scores[truncated_count:]
                    for sub in subtag_names:
                        if subtag_scores_by_turn.get(sub) is not None:
                            subtag_scores_by_turn[sub] = subtag_scores_by_turn[sub][truncated_count:]
                    print(f"[INFO] Left truncation removed {truncated_count} early turns, "
                          f"keeping last {num_response_positions} turns for env_id={env_id}")
                else:
                    # Right truncation: later turns were removed, keep the first num_response_positions
                    rewards = rewards[:num_response_positions]
                    if think_verifier_scores is not None:
                        think_verifier_scores = think_verifier_scores[:num_response_positions]
                    for sub in subtag_names:
                        if subtag_scores_by_turn.get(sub) is not None:
                            subtag_scores_by_turn[sub] = subtag_scores_by_turn[sub][:num_response_positions]
                    print(f"[INFO] Right truncation removed {original_num_turns - num_response_positions} later turns, "
                          f"keeping first {num_response_positions} turns for env_id={env_id}")
            else:
                # More positions than rewards - unexpected, warn but continue
                print(f"[WARN] Found {num_response_positions} response positions but only {original_num_turns} rewards "
                      f"for env_id={env_id}. This may indicate a bug.")

        # --- NEW: think-mask over RESPONSE tokens only, intersected with loss_mask_response ---
        ids_resp_1d = input_ids_response[0]                # (seq_len_resp,)
        loss_mask_resp_1d = loss_mask_response[0].to(torch.bool)
        eor_mask_resp_1d = end_of_response_position_mask_response[0].to(torch.bool)

        # --- NEW: per-response think mask using loss_mask + end_of_response_positions ---
        # if (
        #     getattr(self.config, "use_think_verifier_reward", False)
        #     and think_verifier_scores is not None
        # ):
        #     # Verifier is active and scores are being used -> build real think mask
        #     think_mask_response = self._build_think_mask_from_loss_1d(
        #         ids_resp_1d=ids_resp_1d,
        #         loss_mask_resp_1d=loss_mask_resp_1d,
        #         eor_mask_resp_1d=eor_mask_resp_1d,
        #         think_verifier_scores=think_verifier_scores,              # may be None
        #         include_tags=True,
        #         debug=getattr(self.config, "debug_think_mask", True),
        #     )
        # else:
        #     # Verifier disabled / not used for shaping -> no need to parse think spans
        #     think_mask_response = torch.zeros_like(loss_mask_resp_1d, dtype=torch.bool)
        # think_mask_response = self._build_think_mask_from_loss_1d_skip_when_no_scores(
        #     ids_resp_1d=ids_resp_1d,
        #     loss_mask_resp_1d=loss_mask_resp_1d,
        #     eor_mask_resp_1d=eor_mask_resp_1d,
        #     think_verifier_scores=think_verifier_scores,              # may be None
        #     include_tags=True,
        #     debug=getattr(self.config, "debug_think_mask", True),
        # )
        ## subtags: think_mask, obs_mask, reason_mask, pred_mask
        (
            think_mask_response,
            obs_mask_response,
            reason_mask_response,
            pred_mask_response,
        ) = self._build_think_subtag_mask_from_loss_1d(
            ids_resp_1d=ids_resp_1d,
            loss_mask_resp_1d=loss_mask_resp_1d,
            eor_mask_resp_1d=eor_mask_resp_1d,
            think_verifier_scores=think_verifier_scores,
            include_tags=True,
            debug=getattr(self.config, "debug_think_mask", True),
        )

        # keep only tokens that belong to responses we train on (defensive)
        think_mask_response = think_mask_response & loss_mask_resp_1d
        obs_mask_response   = obs_mask_response & loss_mask_resp_1d
        reason_mask_response= reason_mask_response & loss_mask_resp_1d
        pred_mask_response  = pred_mask_response & loss_mask_resp_1d
        # --- END NEW ---

        # --- minimal debug to inspect what's under think_mask (optional) ---
        # if getattr(self.config, "debug_think_mask", True):
        #     num_think = int(think_mask_response.sum().item())
        #     if num_think > 0:
        #         idx = torch.nonzero(
        #             think_mask_response & (ids_resp_1d != self.tokenizer.pad_token_id),
        #             as_tuple=False,
        #         ).squeeze(-1)
        #         print(f"[DEBUG] idx is {idx} with sum {idx.sum()}")
        #         #idx = idx[:256]  # avoid huge spew
        #         text_snippet = self.tokenizer.decode(ids_resp_1d[idx])
        #         print(f"[DEBUG] think_mask_response tokens={num_think}, snippet:\n{text_snippet}\n")
        if getattr(self.config, "debug_think_mask", True):
            debug_masks = [
                ("think", think_mask_response),
                ("obs", obs_mask_response),
                ("reason", reason_mask_response),
                ("pred", pred_mask_response),
            ]
            for label, mask in debug_masks:
                num_tokens = int(mask.sum().item())
                if num_tokens <= 0:
                    continue
                idx = torch.nonzero(
                    mask & (ids_resp_1d != self.tokenizer.pad_token_id),
                    as_tuple=False,
                ).squeeze(-1)
                if idx.numel() == 0:
                    continue
                text_snippet = self.tokenizer.decode(ids_resp_1d[idx])
                print(f"[DEBUG] {label}_mask_response # tokens={num_tokens}, snippet:\n{text_snippet}\n")
        # -------------------------------------------------------------

        # # --- NEW: compute think-mask over RESPONSE tokens only ---
        # # We will store a 1D mask aligned with `responses` (no prompt tokens).
        # think_start_tokens = self.tokenizer.encode("<think>", add_special_tokens=False)
        # think_end_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        # # input_ids_response is (1, seq_len) here
        # think_mask_response = self._build_span_mask(
        #     input_ids_response[0], think_start_tokens, think_end_tokens
        # )

        input_ids_prompt=input_ids_prompt[0]  # (1, seq_len) -> (seq_len,)
        attention_mask_prompt=attention_mask_prompt[0]
        loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
        end_of_response_position_mask_prompt = torch.zeros_like(attention_mask_prompt)

        input_ids_response=input_ids_response[0]  # (1, seq_len) -> (seq_len,)
        attention_mask_response=attention_mask_response[0]
        loss_mask_response=loss_mask_response[0]
        end_of_response_position_mask_response=end_of_response_position_mask_response[0]

        # keep think_mask aligned with responses (1D, length == len(input_ids_response))
        think_mask = think_mask_response.to(torch.float32)
        obs_mask   = obs_mask_response.to(torch.float32)
        reason_mask = reason_mask_response.to(torch.float32)
        pred_mask   = pred_mask_response.to(torch.float32)

        loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
        end_of_response_position_mask = torch.cat([end_of_response_position_mask_prompt, end_of_response_position_mask_response], dim=-1)
        input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
        attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)
        # if self.image_key in row_dict:
        if has_images and image_grid_thw is not None:
            # Qwen model with image_grid_thw
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids_response = get_rope_index(
                self.processor,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids_response,
                attention_mask=attention_mask_response,
            )  # (3, seq_len)
            position_ids_prompt=position_ids_prompt.view(1, -1).expand(3, -1)
        else:
            # InternVL model or no images - use simple position computation
            response_length = input_ids_response.shape[0]
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
            position_ids_response = position_ids_prompt[-1:] + delta_position_id

        if self.config.use_multi_turn_reward:
            reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)  # reward_positions is tensor([1388, 2154, 2920, 3686, 4452]) with shape torch.Size([5])
            multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask, dtype=torch.float)
            assert len(reward_positions) == len(rewards), (
                f"Number of rewards ({len(rewards)}) does not match number of reward positions ({len(reward_positions)}). "
                f"This should not happen after truncation handling. env_id={env_id}"
            )
            for idx,reward in enumerate(rewards):
                multi_turn_token_level_rewards[reward_positions[idx]] = reward  # last (rewarded) token in x-response in input_ids: 151645, its string is <|im_end|>
            row_dict["multi_turn_token_level_rewards"] = multi_turn_token_level_rewards # (seq_len,)
            row_dict["end_of_response_position_mask"] = end_of_response_position_mask

        # TODO: delete
        # # --- NEW: per-token think verifier rewards (finer credit) ---
        # #
        # # Idea:
        # #  - `end_of_response_position_mask_response` tells us which response token index ends each turn.
        # #  - `think_mask` tells us which response tokens are inside ANY <think>...</think>.
        # #  - `think_verifier_scores[k]` is the score for turn k.
        # #  For each turn k, we:
        # #    * find slice [prev_end+1 : end_k+1] in RESPONSE indices,
        # #    * restrict to tokens where think_mask=True,
        # #    * set those tokens' bonus to think_verifier_scores[k].
        # #
        # if (
        #     getattr(self.config, "use_think_verifier_reward", False)
        #     and think_verifier_scores is not None
        #     and len(think_verifier_scores) > 0
        # ):
        #     reward_positions_resp = torch.nonzero(end_of_response_position_mask_response).squeeze(-1)
        #     # handle mismatches gracefully but warn via comment
        #     K = min(len(think_verifier_scores), int(reward_positions_resp.numel()))
        #     think_verifier_token_rewards_resp = torch.zeros_like(
        #         end_of_response_position_mask_response, dtype=torch.float
        #     )
        #     prev_end = -1
        #     for idx in range(K):
        #         score = float(think_verifier_scores[idx])
        #         end_pos = int(reward_positions_resp[idx].item())
        #         seg = slice(prev_end + 1, end_pos + 1)
        #         seg_think_mask = think_mask[seg]
        #         if seg_think_mask.any():
        #             # only reward <think> tokens in this turn
        #             think_verifier_token_rewards_resp[seg][seg_think_mask] = score
        #         prev_end = end_pos

        #     # expand to full seq (prompt + response) so shape matches multi_turn_token_level_rewards
        #     full_think_verifier_token_rewards = torch.zeros_like(
        #         end_of_response_position_mask, dtype=torch.float
        #     )
        #     prompt_len = input_ids_prompt.shape[0]
        #     full_think_verifier_token_rewards[prompt_len:] = think_verifier_token_rewards_resp
        #     row_dict["think_verifier_token_rewards"] = full_think_verifier_token_rewards
        # # --- END NEW ---


        # --- NEW: Finer-credit Option A using per-turn verifier scores ---
        # build token-level verifier rewards on RESPONSE tokens, then map to full sequence
        think_verifier_token_rewards_resp = torch.zeros_like(
            input_ids_response, dtype=torch.float32
        )  # (seq_len_resp,)

        if (
            getattr(self.config, "use_think_verifier_reward", False)
            and think_verifier_scores is not None
            and len(think_verifier_scores) > 0
        ):
            reward_positions_resp = torch.nonzero(
                end_of_response_position_mask_response, as_tuple=False
            ).squeeze(-1)  # (K,)
            K = min(len(think_verifier_scores), int(reward_positions_resp.numel()))
            prev_end = -1

            mode = getattr(self.config, "think_reward_mode", "last_token")
            # mode in {"last_token", "all_tokens_normalized"}

            # --- get env_id for warnings ---
            env_id = recording[0].get("env_id", "unknown")

            for k in range(K):
                score = float(think_verifier_scores[k])
                end_idx = int(reward_positions_resp[k].item())

                seg_indices = torch.arange(
                    prev_end + 1, end_idx + 1, device=input_ids_response.device
                )
                if seg_indices.numel() == 0:
                    prev_end = end_idx
                    continue

                seg_think_mask = think_mask[seg_indices].to(torch.bool)
                think_inds = seg_indices[seg_think_mask]

                # --- NEW: warning when score ≠ 0 but no <think> tokens in this turn ---
                if think_inds.numel() == 0:
                    if score != 0.0:
                        print(
                            f"[WARN] Non-zero verifier score ({score:.2f}) "
                            f"but no <think> tokens for env_id={env_id}, turn={k}"
                        )
                    prev_end = end_idx
                    continue

                if mode == "all_tokens_normalized":
                    # length-normalized: total bonus per turn = score
                    per_token = score / float(think_inds.numel())
                    think_verifier_token_rewards_resp[think_inds] += per_token
                else:
                    # "last_token": only last <think> token gets the score
                    last_idx = int(think_inds[-1].item())
                    think_verifier_token_rewards_resp[last_idx] += score

                prev_end = end_idx

        # map RESPONSE-level rewards into full sequence (prompt + response)
        seq_len_prompt = input_ids_prompt.shape[0]
        seq_len_resp = input_ids_response.shape[0]
        full_seq_len = seq_len_prompt + seq_len_resp

        think_verifier_token_rewards_full = torch.zeros(
            full_seq_len, dtype=torch.float32, device=input_ids_response.device
        )
        think_verifier_token_rewards_full[seq_len_prompt:] = think_verifier_token_rewards_resp

        if getattr(self.config, "use_think_verifier_reward", False):
            row_dict["think_verifier_token_rewards"] = think_verifier_token_rewards_full
        # --- END NEW ---

        # --- NEW: Finer-credit over subtags using per-turn sub_scores ---
        think_verifier_token_rewards_resp = torch.zeros_like(
            input_ids_response, dtype=torch.float32
        )  # (seq_len_resp,)

        use_think_reward = getattr(self.config, "use_think_verifier_reward", False)
        if use_think_reward:
            reward_positions_resp = torch.nonzero(
                end_of_response_position_mask_response, as_tuple=False
            ).squeeze(-1)  # (K,)
            K_turns = int(reward_positions_resp.numel())
            total_turns = len(rewards)
            K = min(K_turns, total_turns)

            mode = getattr(self.config, "think_reward_mode", "last_token")
            # mode = getattr(self.config, "think_reward_mode", "all_tokens_normalized")
            env_id = recording[0].get("env_id", "unknown")

            # subtag masks over RESPONSE tokens (aligned with input_ids_response)
            subtag_masks_resp = {
                "observation": obs_mask.to(torch.bool),
                "reasoning": reason_mask.to(torch.bool),
                "prediction": pred_mask.to(torch.bool),
            }

            # check if we have any usable subtag series
            subtag_names = ("observation", "reasoning", "prediction")
            has_any_subtag = any(
                (subtag_scores_by_turn.get(sub) is not None) for sub in subtag_names
            )

            prev_end = -1

            for k in range(K):
                end_idx = int(reward_positions_resp[k].item())
                seg_indices = torch.arange(
                    prev_end + 1, end_idx + 1, device=input_ids_response.device
                )
                if seg_indices.numel() == 0:
                    prev_end = end_idx
                    continue

                if has_any_subtag:
                    # --- subtag-based redistribution ---
                    for sub in subtag_names:
                        series = subtag_scores_by_turn.get(sub)
                        if series is None or k >= len(series):
                            continue
                        score_sub = float(series[k])
                        if score_sub == 0.0:
                            continue

                        mask_resp = subtag_masks_resp[sub]
                        sub_inds = seg_indices[mask_resp[seg_indices]]
                        if sub_inds.numel() == 0:
                            if score_sub != 0.0:
                                print(
                                    f"[WARN] Non-zero {sub} score ({score_sub:.2f}) "
                                    f"but no <{sub}> tokens for env_id={env_id}, turn={k}"
                                )
                            continue

                        if mode == "all_tokens_normalized":
                            per_token = score_sub / float(sub_inds.numel())
                            think_verifier_token_rewards_resp[sub_inds] += per_token
                        else:
                            last_idx = int(sub_inds[-1].item())
                            think_verifier_token_rewards_resp[last_idx] += score_sub
                else:
                    # --- fallback: old behavior with aggregate think_verifier_scores ---
                    if think_verifier_scores is None or k >= len(think_verifier_scores):
                        prev_end = end_idx
                        continue
                    score = float(think_verifier_scores[k])
                    seg_think_mask = think_mask[seg_indices].to(torch.bool)
                    think_inds = seg_indices[seg_think_mask]

                    if think_inds.numel() == 0:
                        if score != 0.0:
                            print(
                                f"[WARN] Non-zero verifier score ({score:.2f}) "
                                f"but no <think> tokens for env_id={env_id}, turn={k}"
                            )
                        prev_end = end_idx
                        continue

                    if mode == "all_tokens_normalized":
                        per_token = score / float(think_inds.numel())
                        think_verifier_token_rewards_resp[think_inds] += per_token
                    else:
                        last_idx = int(think_inds[-1].item())
                        think_verifier_token_rewards_resp[last_idx] += score

                prev_end = end_idx

        # map RESPONSE-level rewards into full sequence (prompt + response)
        seq_len_prompt = input_ids_prompt.shape[0]
        seq_len_resp = input_ids_response.shape[0]
        full_seq_len = seq_len_prompt + seq_len_resp

        think_verifier_token_rewards_full = torch.zeros(
            full_seq_len, dtype=torch.float32, device=input_ids_response.device
        )
        think_verifier_token_rewards_full[seq_len_prompt:] = think_verifier_token_rewards_resp

        if use_think_reward:
            row_dict["think_verifier_token_rewards"] = think_verifier_token_rewards_full
        # --- END NEW ---

        if self.config.use_loss_mask:
            row_dict['loss_mask'] = loss_mask
        if self.config.use_gae_mask:
            row_dict['gae_mask'] = loss_mask
        row_dict["end_of_response_position_mask"] = end_of_response_position_mask
        row_dict["truncated_turns"] = torch.tensor([truncated_turns], dtype=torch.long)
        position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)
        row_dict['prompts'] = input_ids_prompt
        row_dict['responses'] = input_ids_response
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        # --- think mask aligned with `responses` ---
        row_dict['think_mask'] = think_mask
        row_dict['obs_mask'] = obs_mask
        row_dict['reason_mask'] = reason_mask
        row_dict['pred_mask'] = pred_mask
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["step_reward_sum"]= sum(rewards)

        return row_dict

    @torch.no_grad()
    def generate_batch_for_rollout(self, step, window_size):
        """
        Generate a batch of data for the current step

        Args:
            step: Current step to generate input for
            window_size: Number of past steps to include in the context

        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - None if no data is available (all environments are done)
        """
        batch = []
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue

            batch.append(self._generate_input_for_rollout(self.recorder[env_id], step, window_size))
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if not batch:
            return None
        if len(batch) % self.config.n_gpus_per_node != 0:
            # Pad the batch to make it divisible by n_gpus_per_node
            while len(batch) % self.config.n_gpus_per_node != 0:
                # do we need to use copy or not here?
                batch.append(batch[-1].copy())
        return collate_fn(batch)

    @torch.no_grad()
    def rollout_loop(self):
        """
        Step the environment and record the results

        Returns:
            Dictionary containing the results of the step
        """
        for step in range(self.config.max_turns):
            input_batch_dict = self.generate_batch_for_rollout(step, self.config.window_size)
            if input_batch_dict is None:
                break
            input_batch = DataProto.from_single_dict(input_batch_dict)
            if 'multi_modal_data' in input_batch.non_tensor_batch.keys():
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                )
            else:
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            # transform raw_prompt_ids to list instead of numpy array
            # The reason is that when constructing raw_prompt_ids, if the all the list share the same length
            # Numpy array will automatically transfer list to numpy array.
            raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids']
            raw_prompt_ids_array = np.ndarray(shape=(len(raw_prompt_ids),), dtype=object)
            for i in range(len(raw_prompt_ids)):
                if isinstance(raw_prompt_ids[i],list):
                    raw_prompt_ids_array[i] = raw_prompt_ids[i]
                else:
                    raw_prompt_ids_array[i] = raw_prompt_ids[i].tolist()
            gen_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids_array

            output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)



            responses_str = self.tokenizer.batch_decode(
                output_batch.batch['responses'],
                skip_special_tokens=True
            ) # seems here will remove special token like "<|im_end|>"

            ids2actions = {}
            for batch_idx, env_id in self.batch_idx_to_env_id.items():
                self._cache_rollout_analysis_artifact(
                    env_id=env_id,
                    step=step,
                    window_size=self.config.window_size,
                    batch_idx=batch_idx,
                    gen_batch=gen_batch,
                    output_batch=output_batch,
                )
                ids2actions[env_id] = responses_str[batch_idx]

            step_results = self.env_client.step_batch(ids2actions)
            for env_id, rst in step_results.items():
                obs, reward, done, info = rst # obs is a dict for next observation
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = done
                self.env_states[env_id]['metrics']['traj_metrics'] = info['metrics'].get('traj_metrics', {})
                # for k,v in info['metrics']['turn_metrics'].items():
                #     self.env_states[env_id]['metrics']['turn_metrics'][k].append(v)

                turn_tm = info['metrics'].get('turn_metrics', {})
                tm_store = self.env_states[env_id]['metrics']['turn_metrics']  # defaultdict(list)
                for k, v in turn_tm.items():
                    tm_store[k].append(v)

                self.record(env_id, obs, reward, done, info)

    @torch.no_grad()
    def generate_batch_for_update(self) -> DataProto:
        """
        Get the final trajectory of all environments

        Returns:
            batch (DataProto): batch of final trajectory of all environments
        """
        batch_list = []
        reward_rst=self.env_client.compute_reward_batch(list(self.envs.keys()))  # for now, only 0 unless customized
        for env_id in self.envs.keys():
            row_dict = self._generate_input_for_update(
                recording=self.recorder[env_id],
                step=self.env_states[env_id]['step'],
                window_size=None,
            )
            step_reward_sum= row_dict['step_reward_sum']

            # --- NEW: propagate an aggregate self-consistency score for this trajectory ---
            tm = self.env_states[env_id]['metrics']['turn_metrics']
            vkey = "self_consistency_verifier_score"
            raw_scores = tm.get(vkey, [])

            # raw_scores is either:
            #  - a list of floats (verifier enabled), or
            #  - empty / missing (verifier disabled).
            if isinstance(raw_scores, (list, tuple)) and len(raw_scores) > 0:
                # e.g., mean over turns;
                # think_verifier_score = float(np.sum(tm[vkey]))
                think_verifier_score = float(np.mean(raw_scores))
            else:
                # No verifier series: treat as 0.0 for this trajectory
                think_verifier_score = 0.0

            # store as a 0-D tensor so it becomes part of batch.batch
            row_dict["think_verifier_score"] = torch.tensor(
                think_verifier_score, dtype=torch.float32
            )
            # --- END NEW ---

            row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": reward_rst[env_id]+step_reward_sum}}
            if self.config.use_multi_turn_reward:
                end_of_response_position_mask = row_dict['end_of_response_position_mask']
                reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
                last_reward_index = reward_positions[-1]
                row_dict['multi_turn_token_level_rewards'][last_reward_index] += reward_rst[env_id]
            # Embed env_id and config_id so they travel with the trajectory data,
            # avoiding positional-index mismatches with recording_to_log().
            row_dict['env_id'] = env_id
            row_dict['config_id'] = self.envs[env_id].config_id()
            batch_list.append(row_dict)
        batch_dict = collate_fn(batch_list)
        batch = DataProto.from_single_dict(batch_dict)
        ## debug prints
        ################################################################################
        # batch is DataProto(batch=TensorDict(
        # fields={
        # attention_mask: Tensor(shape=torch.Size([128, 5000]), device=cpu, dtype=torch.int64, is_shared=False),
        # end_of_response_position_mask: Tensor(shape=torch.Size([128, 5000]), device=cpu, dtype=torch.int64, is_shared=False),
        # gae_mask: Tensor(shape=torch.Size([128, 5000]), device=cpu, dtype=torch.int64, is_shared=False),
        # input_ids: Tensor(shape=torch.Size([128, 5000]), device=cpu, dtype=torch.int64, is_shared=False),
        # loss_mask: Tensor(shape=torch.Size([128, 5000]), device=cpu, dtype=torch.int64, is_shared=False),
        # multi_turn_token_level_rewards: Tensor(shape=torch.Size([128, 5000]), device=cpu, dtype=torch.float32, is_shared=False),
        # position_ids: Tensor(shape=torch.Size([128, 3, 5000]), device=cpu, dtype=torch.int64, is_shared=False),
        # prompts: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        # responses: Tensor(shape=torch.Size([128, 4999]), device=cpu, dtype=torch.int64, is_shared=False),
        # think_mask: Tensor(shape=torch.Size([128, 4999]), device=cpu, dtype=torch.float32, is_shared=False),
        # think_verifier_score: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.float32, is_shared=False)},
        # batch_size=torch.Size([128]),device=None,
        # is_shared=False), non_tensor_batch={'multi_modal_data': array([{'image': [<PIL.Image.Image image mode=RGB size=512x512 at 0x14BB26E92AA0>, <PIL.Image.Image image mode=RGB size=512x512 at 0x14BB26E92560>, <PIL.Image.Image image mode=RGB size=512x512 at 0x14BB26B92C50>, <PIL.Image.Image image mode=RGB size=512x512 at 0x14BB26E933A0>, <PIL.Image.Image image mode=RGB size=512x512 at 0x14BB26B3F640>]},
        ################################################################################
        return batch

    @torch.no_grad()
    def recording_to_log(self):
        """
        Get the recording of all environments

        Returns:
            Dictionary containing the recording of all environments
        """

        env_info = []
        reward_rst=self.env_client.compute_reward_batch(list(self.envs.keys()))  # for now, only 0 unless customized
        for env_id, record in self.recorder.items():
            config_id = self.envs[env_id].config_id()
            step= self.env_states[env_id]['step']
            output_rst = self._single_recording_to_prompt(record, self.env_states[env_id]['step'], window_size=None, is_final=False)
            image= output_rst['image_data']
            done = self.env_states[env_id]['done']
            score = reward_rst[env_id]+ sum(output_rst['rewards'])

            # --- Extract task info from initial reset info (for trajectory replay) ---
            initial_info = record[0].get('info', {}) if record else {}
            task_name = initial_info.get('task_name', None)
            task_variation = initial_info.get('task_variation', None)
            task_description = initial_info.get('task_description', '')

            # --- NEW: per-turn arrays ---
            # 1) per-turn env rewards
            turn_rewards = list(output_rst['rewards'])  # length == step

            # 2) per-turn verifier scores (collect all *_verifier_score series)
            tm = self.env_states[env_id]['metrics']['turn_metrics']
            turn_verifier_scores = {
                k: v[:step] for k, v in tm.items() if k.endswith("_verifier_score")
            }

            # 3) per-turn reasoning trace token length (inside the active reasoning tag)
            turn_reason_len = []
            for i, rec in enumerate(record):
                if i == 0:
                    continue  # initial reset entry (no assistant response)
                src = (rec.get('info', {}) or {})
                resp = src.get('llm_rewritten_response') or src.get('llm_raw_response', '') or ''
                reasoning = extract_reasoning_content(resp)
                try:
                    n_tokens = len(self.tokenizer.encode(reasoning, add_special_tokens=False))
                except Exception:
                    n_tokens = len(reasoning)  # safe fallback
                turn_reason_len.append(n_tokens)
            # --- END NEW ---

            metrics={
                "score": score,
                "done": done,
                "step": step,
            }

            turn_metrics={
                k: sum(v)/step if step != 0 else 0 for k, v in self.env_states[env_id]['metrics']['turn_metrics'].items()
            }
            traj_metrics=self.env_states[env_id]['metrics']['traj_metrics']
            metrics.update(turn_metrics)
            metrics.update(traj_metrics)
            env_info.append({
                "env_id": env_id,
                "config_id": config_id,
                "output_str": output_rst['prompt'],
                "image_data": image,
                "metrics": metrics,
                # --- NEW payloads exposed for table + aggregation ---
                "turn_rewards": turn_rewards,
                "turn_verifier_scores": turn_verifier_scores,  # dict: rubric -> list
                "turn_reason_len": turn_reason_len,
                # --- Task info for trajectory replay in offline LLM judge evaluation ---
                "task_name": task_name,
                "task_variation": task_variation,
                "task_description": task_description,
            })
        return env_info
