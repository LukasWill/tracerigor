import math, re

ANSWER_RE = re.compile(r"<answer>\s*(YES|NO)\s*</answer>")

def extract_verdict_and_offset(text: str):
    """
    Returns (verdict_str, char_offset_of_verdict_start).
    Verdict is 'YES' or 'NO'. If not found, returns (None, None).
    """
    if not text:
        return None, None
    m = ANSWER_RE.search(text)
    if not m:
        return None, None
    verdict = m.group(1)
    start = m.start(1)  # where the label starts
    return verdict, start

def find_token_index_at_offset(tokens: list[str], text: str, char_offset: int) -> int | None:
    """
    Given the generated tokens (as strings) and the final text, find the index i
    such that tokens[i] is the first token whose characters overlap the char_offset.
    """
    acc = ""
    for i, tk in enumerate(tokens):
        acc += tk
        if len(acc) >= char_offset:
            return i
    return None

def find_token_index_at_offset_tail(tokens: list[str], text: str, char_offset: int) -> int | None:
    """
    Works when 'tokens' only contain the tail of the output (e.g., last 10 tokens).
    Computes where that tail starts in 'text' and returns the index of the token
    overlapping 'char_offset', if the offset lies within this tail.
    """
    if not tokens or not text:
        return None
    tail = "".join(tokens)
    # Where does this tokens tail align in the final text?
    # (Assumes the logprob token strings concatenate to the text suffix.)
    suffix_start = len(text) - len(tail)
    if suffix_start < 0:
        # Defensive: if token join is "longer" than text due to formatting, give up.
        return None
    if char_offset < suffix_start:
        # Verdict occurs before the captured tail.
        return None
    local = char_offset - suffix_start  # local offset inside the tail
    acc = 0
    for i, tk in enumerate(tokens):
        acc += len(tk)
        if acc >= local:
            return i
    return None

def find_label_token_index(tokens: list[str]) -> int | None:
    """
    Fallback: directly find the token that is exactly YES or NO (ignoring spaces, case).
    """
    for i, tk in enumerate(tokens):
        t = (tk or "").strip().upper()
        if t in ("YES", "NO"):
            return i
    return None

def binary_probs_from_toplogprobs(top_logprobs: list[dict]) -> tuple[float|None, float|None]:
    """
    Extract P(YES) and P(NO) from the top-logprobs of a *single* position (i.e., YES or NO lives).
    Returns (p_yes, p_no) or (None, None) if neither is present.
    """
    yes_lp, no_lp = None, None
    for alt in top_logprobs or []:
        tok = alt.get("token")
        lp  = alt.get("logprob")
        if tok == "YES": yes_lp = lp
        if tok == "NO":  no_lp  = lp
    if yes_lp is None and no_lp is None:
        return None, None
    # treat missing as -inf so softmax works
    if yes_lp is None: yes_lp = -float("inf")
    if no_lp  is None: no_lp  = -float("inf")
    # stable two-class softmax
    m = max(yes_lp, no_lp)
    e_yes = math.exp(yes_lp - m) if math.isfinite(yes_lp) else 0.0
    e_no  = math.exp(no_lp  - m) if math.isfinite(no_lp)  else 0.0
    Z = e_yes + e_no
    if Z == 0:
        return None, None
    return e_yes / Z, e_no / Z

def binary_entropy(p_yes: float) -> float:
    """H(p) = -p log p - (1-p) log (1-p) with natural logs."""
    if p_yes is None:
        return None
    p = max(1e-12, min(1 - 1e-12, p_yes))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

def topk_entropy_from_toplogprobs(top_logprobs: list[dict]) -> float | None:
    """
    Approximate token entropy using only the available top-k alternatives.
    This *underestimates* true entropy because it ignores tail mass.
    """
    if not top_logprobs:
        return None
    ps = [math.exp(x["logprob"]) for x in top_logprobs if "logprob" in x]
    Z = sum(ps)
    if Z <= 0:
        return None
    ps = [p / Z for p in ps]
    return -sum(p * math.log(p) for p in ps)

def safe_exp_logprob(lp):
    if lp is None:
        return None
    if math.isfinite(lp):
        return math.exp(lp)
    # keep 0.0 only for -inf; anything else stays None
    if lp == float("-inf"):
        return 0.0
    return None

# import numpy as np

# def token_entropy(logprobs: List[Dict[str, float]]) -> float:
#     """
#     Compute entropy from a list of {token, logprob}.
#     logprob is natural log.
#     """
#     probs = np.array([np.exp(lp["logprob"]) for lp in logprobs])
#     probs /= probs.sum()  # normalize (just in case)
#     entropy = -np.sum(probs * np.log(probs + 1e-12))
#     return float(entropy)


def to_responses_input(messages):
    """
    Convert your current messages (system+user; possibly Chat-style multimodal)
    into Responses API 'instructions' + 'input'.
    Returns: (instructions_str, input_list)
    """
    # 1) pull system & user
    sys_txt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user_msg = next((m for m in messages if m.get("role") == "user"), {"content": ""})
    ucontent = user_msg.get("content", "")

    # 2) build user content parts for Responses
    parts = []
    if isinstance(ucontent, list):
        # Chat-style multimodal from your pipeline: [{"type":"text","text":...}, {"type":"image_url", ...}, ...]
        for p in ucontent:
            ptype = p.get("type")
            if ptype in ("text", "input_text"):
                txt = p.get("text") or p.get("input_text") or ""
                if txt:
                    parts.append({"type": "input_text", "text": txt})
            elif ptype in ("image_url", "input_image"):
                # chat format uses {"type":"image_url","image_url": {"url": "..."} } or sometimes {"image_url":"..."}
                iu = p.get("image_url")
                if isinstance(iu, dict):
                    iu = iu.get("url")
                if iu:
                    parts.append({"type": "input_image", "image_url": iu})
            # maybe wrong
            elif ptype == "image" and "data" in p:
                # if your code ever gives raw/base64 image blobs
                parts.append({"type": "input_image", "image_data": p["data"]})
    else:
        # plain text user content
        parts.append({"type": "input_text", "text": str(ucontent)})

    input_payload = [{"role": "user", "content": parts}]
    return sys_txt, input_payload



if __name__ == "__main__":
    text = "<answer> YES </answer> some other text"
    text = "<think>Reasoning explicitly states the sequence: right, down, right, which matches the action taken and uses only admissible moves. No contradictions or ambiguities are present.</think><answer>YES</answer> some other text"
    tokens = ["<answer>", "YES", "</answer>", "some", "other", "text"]
    verdict, ch_start = extract_verdict_and_offset(text)
    p_yes = p_no = H_bin = H_topk_labelpos = None

    if verdict and tokens:
        i0 = find_token_index_at_offset(tokens, text, ch_start)
        if i0 is not None and 0 <= i0 < len(tops):
            p_yes, p_no = binary_probs_from_toplogprobs(tops[i0])
            if p_yes is not None:
                H_bin = binary_entropy(p_yes)
            H_topk_labelpos = topk_entropy_from_toplogprobs(tops[i0])