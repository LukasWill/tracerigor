
from dataclasses import dataclass, field
from typing import Dict, Any, List, Sequence

@dataclass
class VerifierTemplate:
    """Base class for verifier prompt templates.

    Subclasses should fill:
      - template_id
      - system_prompt
      - user_prompt
      - required_keys (for build_messages input)
    """
    template_id: str
    description: str
    required_keys: Sequence[str] = field(default_factory=tuple)
    system_prompt: str = ""
    user_prompt: str = ""

    def build_messages(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        self._ensure_keys(data)
        rendered_data = self._render_prompt(data)
        user = self.user_prompt.format(**rendered_data)
        # sys = self.system_prompt.format(**rendered_data)
        sys = self.system_prompt  # no .format on system
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

    def _ensure_keys(self, data: Dict[str, Any]) -> None:
        missing = [k for k in self.required_keys if k not in data]
        if missing:
            raise KeyError(f"Missing required keys for {self.template_id}: {missing}")

    def _render_prompt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # auto-handle observation choice
        text = data.get("current_observation_text")
        if text and text.strip():  # was: `is not None`
            data["_current_observation_text_or_image"] = f"{text}"
        # if data.get("current_observation_text") is not None:
        #     # data["_current_observation_text_or_image"] = data["current_observation_text"]
        #     data["_current_observation_text_or_image"] = f"{data['current_observation_text']}"
        elif data.get("current_observation_image"):
            data["_current_observation_text_or_image"] = "<image>"
        else:
            # allow empty for self-consistency/history-only checks
            data.setdefault("_current_observation_text_or_image", "")

        # history pretty-print
        history = data.get("history", [])
        if isinstance(history, list):
            # use_pretty = data.get("pretty_history", False)
            # if use_pretty:
            #     from pprint import pformat
            #     data["_history_str"] = pformat(history, compact=True, width=88)
            # else:
            # TODO: ensure the best way to display the history item in the prompt
            data["_history_str"] = str(history)
        else:
            data["_history_str"] = "[]"

        # Fill step snippet defaults if not provided
        if "current_step" not in data and "step_index" in data:
            data["current_step"] = data["step_index"]
        data.setdefault("step_count", max(int(data.get("current_step") or 0) - 1, 0))
        data.setdefault("history_length", len(history) if isinstance(history, list) else 0)
        # TODO: this may need to account for non-list history formats, in place of the above data["_history_str"]
        if "action_history" not in data:
            try:
                pairs = []
                for h in history:
                    a = h.get("action_tokens")
                    o = h.get("observation_text") or h.get("observation_image")
                    pairs.append({"action": a, "observation": o})
                data["action_history"] = pairs
            except Exception:
                data["action_history"] = history
        return data
