def sanitize_batch_items_for_server(items: list[dict]) -> list[dict]:
    cleaned = []
    for d in items:
        d = dict(d)

        # Required fields sanity
        for k in ("id", "reasoning_tokens", "action_tokens"):
            if k not in d or d[k] is None:
                raise ValueError(f"Missing required key: {k}")

        # admissible_actions must be a list[str]
        if "admissible_actions" in d:
            aa = d["admissible_actions"]
            if not isinstance(aa, list):
                aa = list(aa)  # convert sets/tuples/np arrays
            d["admissible_actions"] = [str(x) for x in aa]

        # observation must be string; if dict/bytes given, map to "<image>"
        val = d.get("_current_observation_text_or_image", "")
        if isinstance(val, (dict, bytes, bytearray)):
            d["_current_observation_text_or_image"] = "<image>"
        elif val is None:
            d["_current_observation_text_or_image"] = ""

        # history normalization (if you send it)
        if "history" in d and isinstance(d["history"], list):
            for h in d["history"]:
                # ensure only plain JSON primitives/string fields
                if isinstance(h.get("observation_image"), (str, dict, bytes, bytearray)):
                    # server expects bool; keep your image for MM path only
                    h["observation_image"] = True
                if "observation_text" in h and h["observation_text"] is None:
                    h["observation_text"] = ""

        # remove fields the server doesn't know (prevents extra="forbid" explosions)
        for k in list(d.keys()):
            if k in {"images", "image", "image_parts", "openai_content"}:
                d.pop(k)

        cleaned.append(d)
    return cleaned

payload = {"items": sanitize_batch_items_for_server(items),
           "rubric": rubric, "models": models, "model_params": model_params}
resp = httpx.post("http://localhost:8000/batch_verify", json=payload)
print(resp.status_code, resp.text)  # print .text to see FastAPI validation details