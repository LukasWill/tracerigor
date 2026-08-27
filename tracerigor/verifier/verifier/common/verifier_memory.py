# verifier_memory.py
from typing import List, Dict, Any, Optional
import io

class VerifierMemory:
    """Keeps short per-episode history for LLM verifier (text + optional images + tokens)."""
    def __init__(self, max_history:int = 5):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []  # newest appended at end

    def reset(self) -> None:
        self.history.clear()

    @staticmethod
    def _pil_to_png_bytes(img) -> bytes:
        # Accept PIL.Image or bytes; return bytes
        if hasattr(img, "save"):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        if isinstance(img, (bytes, bytearray)):
            return bytes(img)
        return img  # paths/urls/dicts: let your normalizer handle them

    def add_step(self,
                 *,
                 observation_text: Optional[str],
                 observation_images: Optional[List[Any]],
                 reasoning_tokens: Optional[str],
                 action_tokens: Optional[str]) -> None:
        imgs = []
        for im in (observation_images or []):
            imgs.append(self._pil_to_png_bytes(im))
        self.history.append({
            "observation_text": observation_text,
            "observation_image": bool(imgs),
            "images": imgs,                    # kept internal; not emitted in "recent"
            "reasoning_tokens": reasoning_tokens,
            "action_tokens": action_tokens,
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def recent(self, k:int = 3, include_images: bool = False) -> List[Dict[str, Any]]:
        """Return last k items in the format your templates expect (no raw bytes)."""
        out: List[Dict[str, Any]] = []
        for h in self.history[-k:]:
            entry = {
                "observation_text": h.get("observation_text"),
                "observation_image": h.get("observation_image", False),
                "reasoning_tokens": h.get("reasoning_tokens"),
                "action_tokens": h.get("action_tokens"),
            }
            if include_images:
                entry["images"] = list(h.get("images") or [])
            out.append(entry)
        return out

    def last_obs_text_and_images(self) -> tuple[str, List[Any]]:
        """Return most recent observation text + image bytes list (or empty)."""
        for h in reversed(self.history):
            images = h.get("images") or []
            if h.get("observation_text") is not None or images:
                return h.get("observation_text") or "", images
        return "", []

    # a tiny helper - if the last entry has no tokens, overwrite it
    def coalesce_or_add(self, observation_text, observation_images, reasoning_tokens, action_tokens):
        if self.history and self.history[-1].get("reasoning_tokens") is None and self.history[-1].get("action_tokens") is None:
            self.history[-1]["reasoning_tokens"] = reasoning_tokens
            self.history[-1]["action_tokens"]    = action_tokens
        else:
            self.add_step(observation_text=observation_text, observation_images=observation_images,
                        reasoning_tokens=reasoning_tokens, action_tokens=action_tokens)
