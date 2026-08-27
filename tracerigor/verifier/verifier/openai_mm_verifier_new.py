from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Union

import base64, mimetypes, pathlib, io

try:
    import tiktoken  # optional, for text token estimates
except Exception:
    tiktoken = None

# -----------------------
# Image normalization
# -----------------------
def _b64_data_url(path: Union[str, pathlib.Path]) -> str:
    path = pathlib.Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _bytes_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _pil_to_data_url(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return _bytes_to_data_url(buf.getvalue(), mime="image/png")

def _np_to_data_url(np_image) -> str:
    from PIL import Image  # type: ignore
    pil = Image.fromarray(np_image)
    return _pil_to_data_url(pil)

def normalize_image_input(image):
    if isinstance(image, dict) and image.get("type") == "image_url":
        return image
    if isinstance(image, (str, pathlib.Path)):
        s = str(image)
        if s.startswith(("http://","https://","data:")):
            return {"type":"image_url","image_url":{"url":s}}
        return {"type": "image_url", "image_url": {"url":_b64_data_url(s)}}
    if isinstance(image, (bytes, bytearray)):
        return {"type": "image_url", "image_url": {"url":_bytes_to_data_url(bytes(image))}}
    try:
        from PIL.Image import Image as PILImage  # type: ignore
        if isinstance(image, PILImage):
            return {"type": "image_url", "image_url": {"url":_pil_to_data_url(image)}}
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        if isinstance(image, np.ndarray):
            return {"type": "image_url", "image_url": {"url":_np_to_data_url(image)}}
    except Exception:
        pass
    raise ValueError(f"Unsupported image input type: {type(image)}")

# -----------------------
# Placeholder labeling / header helpers
# -----------------------
def label_placeholders(user_text: str, num_images: int, label_format: str = "[Image {k}]") -> str:
    """Replace each occurrence of '<image>' with a numbered label.
    - If there are fewer images than placeholders, extra placeholders become '[Image ?]'.
    """
    parts = user_text.split("<image>")
    if len(parts) == 1:
        return user_text  # no placeholders
    out: List[str] = []
    used = 0
    for i, chunk in enumerate(parts):
        out.append(chunk)
        if i < len(parts) - 1:
            if used < num_images:
                used += 1
                out.append(label_format.format(k=used))
            else:
                out.append("[Image ?]")
    return "".join(out)

# def prepend_or_append_header(user_text: str, n_images: int, *, position: str = "top") -> str:
#     if n_images <= 0:
#         return user_text
#     seq = [i+1 for i in range(n_images)]
#     header = f"Attached images: {seq}. Refer as 'Image 1', 'Image 2'.\\n"
#     if position == "bottom":
#         return user_text + ("\\n" if not user_text.endswith("\\n") else "") + header
#     return header + user_text

def prepend_or_append_header(user_text: str, n_images: int, *, position: str = "top") -> str:
    if n_images <= 0:
        return user_text

    idxs = list(range(1, n_images + 1))
    refer = ", ".join(f"'Image {i}'" for i in idxs)
    header = f"Attached images: {idxs}. Refer as {refer}."

    if position == "bottom":
        # ensure exactly one newline before the header and one trailing newline
        body = user_text.rstrip("\n")
        return f"{body}\n{header}\n"

    # position == "top" (default): header first, then body
    body = user_text.lstrip("\n")
    return f"{header}\n{body}"

# -----------------------
# Message builders (with layouts)
# -----------------------
def build_mm_messages(
    *,
    system_text: str,
    user_text: str,
    images: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any], bytes]]] = None,
    attach_image_refs_header: bool = True,
    # new flexible controls
    placeholder_strategy: str = "strip",   # 'strip' | 'label' | 'keep'
    image_layout: str = "append",          # 'append' | 'interleave'
    header_position: str = "top",          # 'top' | 'bottom'
    label_format: str = "[Image {k}]",
    inline_label_in_interleave: bool = True,
) -> List[Dict[str, Any]]:
    """Build OpenAI chat messages (text + optional images).

    placeholder_strategy:
      - 'strip': remove '<image>' tokens (default legacy behavior)
      - 'label': replace '<image>' -> '[Image k]' (numbered)
      - 'keep' : keep literal '<image>' in text
    image_layout:
      - 'append': single text part + all images appended as image parts
      - 'interleave': split on '<image>', emit text part, then image part, repeat
                      (optionally insert inline labels at each split)
    """
    images = list(images) if images else []
    n_img = len(images)

    # 1) Apply placeholder strategy to user_text
    utxt = user_text
    if placeholder_strategy == "strip":
        utxt = utxt.replace("<image>", "").strip()
    elif placeholder_strategy == "label":
        utxt = label_placeholders(utxt, n_img, label_format=label_format)
    elif placeholder_strategy == "keep":
        pass
    else:
        raise ValueError("placeholder_strategy must be one of: 'strip','label','keep'")

    # 2) Attach optional header
    if n_img and attach_image_refs_header:
        utxt = prepend_or_append_header(utxt, n_img, position=header_position)

    # 3) Build content with chosen layout
    content: List[Dict[str, Any]] = []

    if n_img == 0 or image_layout == "append":
        # Single text part + all image parts
        content.append({"type": "text", "text": utxt})
        for img in images:
            content.append(normalize_image_input(img))

    elif image_layout == "interleave":
        # Split user_text on '<image>' occurrences. If none present, fall back to append.
        if "<image>" not in user_text:
            content.append({"type": "text", "text": utxt})
            for img in images:
                content.append(normalize_image_input(img))
        else:
            chunks = user_text.split("<image>")
            img_idx = 0
            for i, chunk in enumerate(chunks):
                # Base text for this chunk
                text_chunk = chunk
                # Optionally add inline label where the placeholder was
                if i < len(chunks) - 1 and inline_label_in_interleave:
                    label = label_format.format(k=img_idx + 1) if img_idx < n_img else "[Image ?]"
                    text_chunk = (text_chunk + ("\n" if text_chunk and not text_chunk.endswith("\\n") else "") + label)
                # Apply header to the *first* chunk only (to avoid repetition)
                if i == 0 and n_img and attach_image_refs_header:
                    text_chunk = prepend_or_append_header(text_chunk, n_img, position=header_position)
                content.append({"type": "text", "text": text_chunk})
                # Insert the image part after this split if available
                if i < len(chunks) - 1 and img_idx < n_img:
                    content.append(normalize_image_input(images[img_idx]))
                    img_idx += 1
            # If extra images remain, append them (rare)
            while img_idx < n_img:
                content.append(normalize_image_input(images[img_idx]))
                img_idx += 1
    else:
        raise ValueError("image_layout must be one of: 'append','interleave'")

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]

def estimate_text_tokens(model: str, messages):
    img_count = 0
    txt_segments = []
    for msg in messages:
        c = msg.get("content","")
        if isinstance(c, str):
            txt_segments.append(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type")=="text":
                    txt_segments.append(part.get("text",""))
                elif part.get("type")=="image_url":
                    img_count += 1
    if tiktoken is None:
        return {"text_tokens": None, "image_count": img_count, "note": "tiktoken not installed"}
    enc = tiktoken.get_encoding("cl100k_base")
    text_tokens = sum(len(enc.encode(s)) for s in txt_segments)
    return {"text_tokens": text_tokens, "image_count": img_count, "note": "images not counted"}

def make_user_text_for_verifier(*, reasoning_tokens: str, action_tokens=None, admissible_actions=None, observation_text=None, header=None):
    parts = []
    if header: parts.append(header.strip())
    parts.append(f"Reasoning:\\n{reasoning_tokens}")
    if action_tokens is not None: parts.append(f"\\nAction taken:\\n{action_tokens}")
    if admissible_actions: parts.append(f"\\nAdmissible actions:\\n{admissible_actions}")
    if observation_text: parts.append(f"\\nCurrent Observation:\\n{observation_text}")
    return "\\n".join(parts).strip()

class OpenAIMultimodalVerifier:
    """Compose messages for multimodal (text+image) prompts and call OpenAI chat completions."""
    def __init__(self, client, model: str, *, temperature: float = 0.0, **gen_kwargs):
        self.client = client; self.model = model; self.default_kwargs = dict(temperature=temperature, **gen_kwargs)

    def build_messages(
        self,
        *,
        system_text: str,
        user_text: str,
        images: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any], bytes]]] = None,
        strip_image_token: bool = True,
        attach_image_refs_header: bool = True,
        placeholder_strategy: str = "strip",
        image_layout: str = "append",
        header_position: str = "top",
        label_format: str = "[Image {k}]",
        inline_label_in_interleave: bool = True,
    ) -> List[Dict[str, Any]]:
        return build_mm_messages(
            system_text=system_text,
            user_text=user_text,
            images=images,
            attach_image_refs_header=attach_image_refs_header,
            placeholder_strategy=placeholder_strategy,
            image_layout=image_layout,
            header_position=header_position,
            label_format=label_format,
            inline_label_in_interleave=inline_label_in_interleave,
        )

    def estimate_tokens(self, messages): return estimate_text_tokens(self.model, messages)

    def verify(
        self,
        *,
        system_text: str,
        user_text: str,
        images: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any], bytes]]] = None,
        strip_image_token: bool = True,
        attach_image_refs_header: bool = True,
        placeholder_strategy: str = "strip",
        image_layout: str = "append",
        header_position: str = "top",
        label_format: str = "[Image {k}]",
        inline_label_in_interleave: bool = True,
        **gen_kwargs,
    ) -> str:
        messages = self.build_messages(
            system_text=system_text,
            user_text=user_text,
            images=images,
            strip_image_token=strip_image_token,
            attach_image_refs_header=attach_image_refs_header,
            placeholder_strategy=placeholder_strategy,
            image_layout=image_layout,
            header_position=header_position,
            label_format=label_format,
            inline_label_in_interleave=inline_label_in_interleave,
        )
        # resp = self.client.chat.completions.create(model=self.model, messages=messages, **(self.default_kwargs | gen_kwargs))
        return ""

def observation_to_openai_inputs(obs: Dict[str, Any], item_idx: int):
    obs_texts = obs.get("text"); obs_images = obs.get("image")
    observation_text = obs_texts[item_idx] if obs_texts is not None else None
    image_item = obs_images[item_idx] if obs_images is not None else None
    images = [image_item] if image_item is not None else None
    return {"observation_text": observation_text, "images": images}


def run_openai_verifier_mm(
    client,
    *,
    model: str,
    template,                 # instance with .build_messages(data) -> [system,user]
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
