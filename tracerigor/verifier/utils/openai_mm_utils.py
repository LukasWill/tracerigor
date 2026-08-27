from __future__ import annotations
from typing import Any, Dict, List, Union
from collections import OrderedDict
import base64, mimetypes, pathlib, io, hashlib
import PIL  # type: ignore
from PIL import Image  # type: ignore
import numpy as np  # type: ignore

# Optional: text token estimate (images cannot be pre-counted reliably)
try:
    import tiktoken  # pip install tiktoken
except Exception:
    tiktoken = None

# -----------------------
# Image normalization
# -----------------------
## Optional in-memory cache to avoid re-encoding identical images
_IMG_CACHE: OrderedDict[str, str] = OrderedDict()
_IMG_CACHE_CAP = 4096 * 3  # tune

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def _hash_img(img: Image.Image) -> str:
    # robust content hash (no metadata)
    with io.BytesIO() as buf:
        # use fast settings: see next section
        img.save(buf, format="PNG", optimize=False, compress_level=1)
        return hashlib.blake2b(buf.getvalue(), digest_size=16).hexdigest()

def pil_to_data_url_cached(img: Image.Image) -> str:
    h = _hash_img(img)
    if h in _IMG_CACHE:
        _IMG_CACHE.move_to_end(h)
        return _IMG_CACHE[h]
    with io.BytesIO() as buf:
        # re-encode once (fast settings)
        img.save(buf, format="PNG", optimize=False, compress_level=1)
        url = f"data:image/png;base64,{_b64(buf.getvalue())}"
    _IMG_CACHE[h] = url
    if len(_IMG_CACHE) > _IMG_CACHE_CAP:
        _IMG_CACHE.popitem(last=False)
    return url
## end cache

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
    # requires Pillow
    try:
        from PIL import Image
    except Exception as e:
        raise ValueError("NumPy array -> data URL requires Pillow. Install pillow.") from e
    if np_image.dtype != np.uint8:
        raise ValueError("Expected uint8 image array")
    pil = Image.fromarray(np_image)
    return _pil_to_data_url(pil)

def normalize_image_input(
    image: Union[str, pathlib.Path, bytes, Dict[str, Any], PIL.Image.Image, np.ndarray]  # type: ignore[name-defined]
) -> Dict[str, Any]:
    """
    Accepts:
      - http(s):// URL or data: URL string
      - local file path (converted to a data: URL)
      - raw bytes (PNG/JPEG) -> converted to data: URL (assumes PNG if format unknown; prefer passing a dict)
      - dict: {"type": "image_url", "image_url": {"url": ...}} (passed through unchanged)
      - Pillow Image or NumPy ndarray (converted to PNG data: URL)

    Returns:
      An OpenAI chat content part dict with type "image_url".
    """
    if isinstance(image, dict) and image.get("type") == "image_url":
        return image

    if isinstance(image, (str, pathlib.Path)):
        s = str(image)
        if s.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": s}}
        # local file -> data URL
        return {"type": "image_url", "image_url": {"url": _b64_data_url(s)}}

    if isinstance(image, (bytes, bytearray)):
        return {"type": "image_url", "image_url": {"url": _bytes_to_data_url(bytes(image))}}

    # Pillow image
    try:
        from PIL.Image import Image as PILImage  # type: ignore
        if isinstance(image, PILImage):
            return {"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}}
    except Exception:
        pass

    # NumPy array
    try:
        import numpy as np  # type: ignore
        if isinstance(image, np.ndarray):
            return {"type": "image_url", "image_url": {"url": _np_to_data_url(image)}}
    except Exception:
        pass

    raise ValueError(f"Unsupported image input type: {type(image)}")

def estimate_text_tokens(model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best-effort **text-only** token estimate (images are not counted).
    Returns:
      {"text_tokens": int|None, "image_count": int, "note": "..."}
    """
    img_count = 0
    txt_segments: List[str] = []
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str):
            txt_segments.append(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    txt_segments.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    img_count += 1
    if tiktoken is None:
        return {"text_tokens": None, "image_count": img_count, "note": "tiktoken not installed"}
    enc = tiktoken.get_encoding("cl100k_base")
    text_tokens = sum(len(enc.encode(s)) for s in txt_segments)
    return {"text_tokens": text_tokens, "image_count": img_count, "note": "images not counted"}

# -----------------------
# Adapters for your obs dict
# -----------------------
def observation_to_openai_inputs(
    obs: Dict[str, Any],
    item_idx: int,
) -> Dict[str, Any]:
    """
    Convert one item from your 'obs' batch dict into:
      - observation_text (str|None)
      - images (list|None)
    Expected obs keys: 'text', 'image'.
    """
    obs_texts = obs.get("text")
    obs_images = obs.get("image")

    observation_text = obs_texts[item_idx] if obs_texts is not None else None
    image_item = obs_images[item_idx] if obs_images is not None else None

    images = None
    if image_item is not None:
        # Accepts URL, path, dict, PIL.Image, numpy array, or bytes
        images = [image_item]

    return {"observation_text": observation_text, "images": images}

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
