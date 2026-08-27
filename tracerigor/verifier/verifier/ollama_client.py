import pandas as pd
import httpx
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt
import os, math, base64, mimetypes, pathlib


API_URL = "http://localhost:8000/batch_verify"

batch_items = [
    {
        "id": "1",
        "reasoning_tokens": "<think><observation>The player is in the top-left corner, and there is a box to the right.</observation><reasoning>Since the player is in the top-left corner, the first action should be to move left to get closer to the box. The box is to the right, so the next action should be to move right to get closer to the box.</reasoning><prediction>The player will move left, then right.</prediction></think>",
        "action_tokens": "<answer>Left,Right,Down</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)",
    },
    {
        "id": "2",
        "reasoning_tokens": "<think><observation>The player is to the left of the box and the target is to the right of the box.</observation><reasoning>Since the player is to the left of the box and the target is to the right of the box, the first action should be to move to the right to align with the target. The box is in the way, so the next action should be to move down to push the box to the right.</reasoning><prediction>The player will move right, then down.</prediction></think>",
        "action_tokens": "<answer>Right,Down,Right</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)",
    },
    {
        "id": "3",
        "reasoning_tokens": "<think><observation>The player is in the top right corner, and there is a box to the left.</observation><reasoning>Since the player is in the top right corner, the first action should be to move left to get closer to the box. The box is to the left, so the next action should be to move left.</reasoning><prediction>The player will move left.</prediction></think>",
        "action_tokens": "<answer>Left,Down,Left</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)",
    },
    {
        "id": "4",
        "reasoning_tokens": "<think><observation>The player is in the bottom left corner, and there is a box to the left.</observation><reasoning>Since the player is in the bottom left corner, the first action should be to move up to the top of the screen. There is a box to the left, so the next action should be to move left to reach the box.</reasoning><prediction>The player will move up, then left.</prediction></think>",
        "action_tokens": "<answer>Up,Left,Up</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "",
    },
    {
        "id": "5",
        "reasoning_tokens": "<think><observation>The player is in the top left corner, and there is a box to the right.</observation><reasoning>Since the player is in the top left corner, the first action should be to move left to get closer to the box. The box is to the right, so the next action should be to move right to get closer to the box.</reasoning><prediction>The player will move left, then right.</prediction></think>",
        "action_tokens": "<answer>Left,Right,Down</answer>",
        "admissible_actions": ["Left", "Down", "Right", "Up"],
        "current_observation_text": "",
    }
]


data = {
    "admissible_actions": ["up","down","left","right"],
    "current_step": 3,
    "history": [],
    "reasoning_tokens": "<think>...</think>",
    "action_tokens": "left",
    "current_observation_text": "#####\n#_PXO#\n#####",
}

items = [{
    "id":"ex-1",
    "current_step":17,
    "history":[
        {"observation_text":"...", "reasoning_tokens":"<think>t-1</think>", "action_tokens":"left"},
        {"observation_text":"...", "reasoning_tokens":"<think>t-2</think>", "action_tokens":"up"}
    ],
    "current_observation_text":"#####\n#_PXO#\n#####",
    "reasoning_tokens":"<think>now…</think>",
    "action_tokens":"left",
    "admissible_actions": ["up","down","left","right"]
}]

# messages = uv.build_messages({
#     "admissible_actions": ["up","down","left","right"],
#     "current_step": 7,
#     "history": [
#         {"observation_text":"...", "reasoning_tokens":"<think>t-1...</think>", "action_tokens":"left"},
#         {"observation_text":"...", "reasoning_tokens":"<think>t-2...</think>", "action_tokens":"up"}
#     ],
#     "current_observation_text": "#####\n#_PXO#\n#####",
#     "reasoning_tokens": "<think>current...</think>",
#     "action_tokens": "left"
# })

# payload = build_universal_input(
#     admissible_actions=["up","down","left","right"],
#     current_step=7,
#     history=[{"observation_text":"...", "reasoning_tokens":"<think>t-1...</think>", "action_tokens":"left"},
#              {"observation_text":"...", "reasoning_tokens":"<think>t-2...</think>", "action_tokens":"up"}],
#     reasoning_tokens="<think>current...</think>",
#     action_tokens="left",
#     current_observation_text="#####\n#_PXO#\n#####",
# )


# "Player at (0,0), Box at (0,1), Target at (1,0), Walls at (0,2), (1,1)"

# # Convert to DataFrame
# df = pd.DataFrame(data)



MEDIA_BASE = os.environ.get("TRACERIGOR_MEDIA_BASE", "")

def _to_abs_path(p: str) -> str:
    """Resolve relative W&B media paths to absolute by prefixing WANDB_MEDIA_BASE if set."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    if MEDIA_BASE:
        return os.path.join(MEDIA_BASE, p.lstrip("./"))
    return os.path.abspath(p)

def path_to_data_url(p: str) -> str | None:
    """File path -> data URL, or None if missing."""
    if not p:
        return None
    p_abs = _to_abs_path(p)
    if not os.path.exists(p_abs):
        print(f"[warn] image path not found: {p_abs}")
        return None
    mime = mimetypes.guess_type(p_abs)[0] or "application/octet-stream"
    b64  = base64.b64encode(pathlib.Path(p_abs).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _coerce_images_to_dataurl(v):
    # Accept str path/URL -> [data-url or URL], else None
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    # already a URL/data URL? pass through
    if s.startswith(("http://", "https://", "data:")):
        return [s]
    # support multiple paths separated by ';' (optional)
    parts = [p for p in s.split(";") if p.strip()]
    out = []
    for p in parts:
        du = path_to_data_url(p.strip())
        if du:
            out.append(du)
    return out or None

def _opt_text(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v)
    return s if s.strip() else None

def load_and_prepare_items(csv_path: str):
    # Read transformed CSV and filter successful parses
    df = pd.read_csv(csv_path)
    df = df[df['parse_ok'] == True]

    batch_items = []
    for idx, row in df.iterrows():
        # Create a unique id encoding step and example and row index
        item_id = f"{row['step']}_{row['val_example']}_{idx}"
        ## This requires no shared filesystem; the server will accept the data URL directly.
        # img_path = row.get('image_filepath')
        # images = [path_to_data_url(img_path)] if img_path and str(img_path).strip() else None
        batch_items.append({
            "id": item_id,
            "reasoning_tokens": row['reasoning_tokens'],
            "action_tokens": row['action_tokens'],
            "admissible_actions": ["Left", "Down", "Right", "Up"],
            "current_observation_text": _opt_text(row.get('current_observation_text')),  # Handle missing observation gracefully
            # send data URLs (or http(s)) — never raw client paths
            "current_observation_image": _coerce_images_to_dataurl(row.get('image_filepath')),
            "history": [],  # optionally could fill in prior history if available
        })
    return batch_items


def call_verifier_api(batch_items, models, model_params: Dict[str, Any], rubric: str = "self_consistency"):
    payload = {
        "items": batch_items,
        "rubric": rubric,
        "models": models,
        "model_params": model_params,
    }
    timeout = httpx.Timeout(500.0, connect=5.0, read=500.0, write=10.0)
    resp = httpx.post(API_URL, json=payload, timeout=timeout)
    # resp = httpx.post(API_URL, json=payload, timeout=60.0)
    resp.raise_for_status()
    print(resp.status_code, resp.text)
    return resp.json()["results"]


def plot_judge(results, rubric, save_path: str = 'faithfulness_vs_step.png'):
    # Build DataFrame from results
    df = pd.DataFrame(results)
    # Extract step from id
    df['step'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    # Average score across all items per step
    summary = df.groupby('step')['score'].mean().reset_index()

    # Plot
    plt.figure()
    plt.xlabel('Training Step')
    # if we're plotting simulatability, use 'ratio' instead of 'score'
    if rubric == "simulatability":
        plt.ylabel('Average Simulatability Ratio')
        plt.title('Simulatability vs. Training Step')
        summary = df.groupby('step')['ratio'].mean().reset_index()
        plt.plot(summary['step'], summary['ratio'])
    elif rubric == "logicalness":
        # logicalness still uses score
        plt.plot(summary['step'], summary['score'])
        plt.ylabel('Average Logicalness Score')
        plt.title('Logicalness vs. Training Step')
    else:
        # faithfulness
        plt.plot(summary['step'], summary['score'])
        plt.ylabel('Average Faithfulness Score')
        plt.title('Faithfulness vs. Training Step')
    plt.grid(True)
    # Save
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Send extracted validation interactions to Ollama verifier and plot results.'
    )
    parser.add_argument(
        '-i', '--input',
        dest='input_csv',
        required=True,
        help='Path to transformed CSV with parse_ok, reasoning_tokens, action_tokens'
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        default=["llama3.3", "deepseek-r1:32b"],
        help='List of models to evaluate'
    )
    parser.add_argument(
        '-a', '--rubric',
        choices=['universal','grounding','self_consistency', 'history'],
        default='self_consistency',
        help='Which rubric to run'
    )
    # parser.add_argument(
    #     '-p', '--plot-output',
    #     dest='plot_path',
    #     default='judge_vs_step.png',
    #     help='Filename to save the judge plot (default: judge_vs_step.png)'
    # )
    args = parser.parse_args()

    # Prepare batch items
    items = load_and_prepare_items(args.input_csv)
    # items = batch_items  # Use the predefined batch_items for testing
    if not items:
        print("No valid items found (parse_ok=True). Exiting.")
        return

    # Call Verifier API
    results = call_verifier_api(items, args.models, {}, args.rubric)

    # Print individual results
    for r in results:
        print(f"{r['model']} | {r['id']}: score={r['score']} parse_ok={r['parse_success']}")
        line = f"{r['model']} | {r['id']}: score={r['score']} parse_ok={r['parse_success']}"
        if args.rubric == "simulatability":
            line += f" ratio={r['ratio']:.2f}"
        print(line)

    # # Plot and save aggregated verifier results
    # plot_judge(results, rubric=args.rubric, save_path=args.plot_path)


if __name__ == '__main__':
    main()
