# image_fast.py
import io, base64, hashlib
from collections import OrderedDict
from typing import Optional
from PIL import Image

# Simple LRU cache for data URLs
_IMG_DATAURL_CACHE: OrderedDict[str, str] = OrderedDict()
_IMG_DATAURL_CACHE_CAP = 4096  # tune for your workload

def _hash_img(img: Image.Image, mode: str = "RGB") -> str:
    """
    Stable, encoding-independent hash of an image.
    We hash (mode, size, raw pixels) so the hash doesn't depend on PNG parameters.
    """
    im = img.convert(mode)  # no dithering; keep pixels exact
    w, h = im.size
    hsh = hashlib.blake2b(digest_size=16)
    hsh.update(mode.encode("ascii"))
    hsh.update(w.to_bytes(2, "big"))
    hsh.update(h.to_bytes(2, "big"))
    hsh.update(im.tobytes())  # raw pixel bytes
    return hsh.hexdigest()

def _fast_png_bytes(img: Image.Image) -> bytes:
    """
    Fast, lossless PNG: skip expensive optimizations.
    Great for tiny frames (e.g., 96x96). Pixel values unchanged.
    """
    buf = io.BytesIO()
    # NOTE: for low-color grid worlds, you *may* downconvert to palette first:
    # pal = img.convert("P", palette=Image.ADAPTIVE, colors=16)
    # pal.save(buf, format="PNG", optimize=False, compress_level=1)
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()

def pil_to_data_url_cached(img: Image.Image, *, cache_cap: Optional[int] = None) -> str:
    """
    Convert a PIL image to a PNG data URL with a small LRU cache to avoid re-encoding.
    """
    key = _hash_img(img)  # independent of encoding toggles
    url = _IMG_DATAURL_CACHE.get(key)
    if url:
        _IMG_DATAURL_CACHE.move_to_end(key)
        return url

    png = _fast_png_bytes(img)
    b64 = base64.b64encode(png).decode("ascii")
    url = f"data:image/png;base64,{b64}"

    _IMG_DATAURL_CACHE[key] = url
    cap = _IMG_DATAURL_CACHE_CAP if cache_cap is None else int(cache_cap)
    if len(_IMG_DATAURL_CACHE) > cap:
        _IMG_DATAURL_CACHE.popitem(last=False)
    return url