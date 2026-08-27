
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .openai_mm_backup import OpenAIMultimodalVerifier, observation_to_openai_inputs

def run_openai_verifier_mm(
    client,
    *,
    model: str,
    template,                 # instance of VerifierTemplate with .build_messages(data) -> [system,user]
    data: Dict[str, Any],
    obs: Optional[Dict[str, Any]] = None,
    item_idx: int = 0,
    # multimodal formatting controls
    placeholder_strategy: str = "strip",   # 'strip' | 'label' | 'keep'
    image_layout: str = "append",          # 'append' | 'interleave'
    header_position: str = "top",          # 'top' | 'bottom'
    attach_image_refs_header: bool = True,
    label_format: str = "[Image {k}]",
    inline_label_in_interleave: bool = True,
    # generation kwargs
    **gen_kwargs,
) -> str:
    """Compose messages from a VerifierTemplate and auto-attach images if present in obs.

    If obs contains an image for item_idx, we build a multimodal request with the selected
    placeholder strategy, header placement, and layout. Otherwise, we send the template's
    text-only messages unchanged.
    """
    # Build base messages (pure text) from the template first
    msgs = template.build_messages(dict(data))
    assert len(msgs) >= 2 and msgs[0]['role'] == 'system' and msgs[1]['role'] == 'user',         "Template must return system+user messages"
    system_text = msgs[0]['content']
    user_text   = msgs[1]['content']

    images = None
    if obs is not None:
        o = observation_to_openai_inputs(obs, item_idx=item_idx)
        images = o.get("images")

    if not images:
        # Fallback: send original text-only messages
        resp = client.chat.completions.create(model=model, messages=msgs, **gen_kwargs)
        return resp.choices[0].message.content or ""

    # Multimodal path
    mm = OpenAIMultimodalVerifier(client, model=model, temperature=gen_kwargs.pop("temperature", 0.0))
    return mm.verify(
        system_text=system_text,
        user_text=user_text,
        images=images,
        placeholder_strategy=placeholder_strategy,
        image_layout=image_layout,
        header_position=header_position,
        attach_image_refs_header=attach_image_refs_header,
        label_format=label_format,
        inline_label_in_interleave=inline_label_in_interleave,
        **gen_kwargs,
    )

def run_openai_verifier_mm_batch(
    client,
    *,
    model: str,
    template,
    data_list: List[Dict[str, Any]],
    obs: Optional[Dict[str, Any]] = None,
    item_idx_list: Optional[List[int]] = None,
    # multimodal formatting controls
    placeholder_strategy: str = "strip",
    image_layout: str = "append",
    header_position: str = "top",
    attach_image_refs_header: bool = True,
    label_format: str = "[Image {k}]",
    inline_label_in_interleave: bool = True,
    # generation kwargs
    **gen_kwargs,
) -> List[str]:
    outs: List[str] = []
    if item_idx_list is None:
        item_idx_list = [0] * len(data_list)
    for d, idx in zip(data_list, item_idx_list):
        outs.append(
            run_openai_verifier_mm(
                client,
                model=model,
                template=template,
                data=d,
                obs=obs,
                item_idx=idx,
                placeholder_strategy=placeholder_strategy,
                image_layout=image_layout,
                header_position=header_position,
                attach_image_refs_header=attach_image_refs_header,
                label_format=label_format,
                inline_label_in_interleave=inline_label_in_interleave,
                **gen_kwargs,
            )
        )
    return outs


if __name__ == "__main__":
    from openai import OpenAI
    from verifier.prompt.sokoban import get_sokoban_verifier_templates

    client = OpenAI()
    verifier_templates = get_sokoban_verifier_templates()
    universal_template = verifier_templates["sokoban.universal"]

    data = {
        "admissible_actions": ["up","down","left","right"],
        "current_step": 3,
        "history": [],
        "reasoning_tokens": "<think>...</think>",
        "action_tokens": "left",
        "current_observation_text": "#####\n#_PXO#\n#####",
        "current_observation_image": "<image>",
    }

    # If an image is available in your env batch, it'll be attached automatically
    obs = {"text": [data["current_observation_text"]], "image": ["/path/to/frame.png"]}

    resp = run_openai_verifier_mm(
        client,
        model="gpt-4o-mini",          # vision-capable chat model
        template=universal_template,
        data=data,
        obs=obs,
        item_idx=0,
        max_tokens=128,
        temperature=0.0,
    )
    print(resp)
