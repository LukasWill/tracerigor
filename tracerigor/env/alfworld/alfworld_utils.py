"""
Helpers for loading an upstream-installed ALFWorld package inside TraceRigor.

The public release does not vendor ALFWorld. We import ``AlfredTWEnv`` or
``AlfredThorEnv`` directly from the separately installed dependency.
"""
import importlib
import os
import threading
import yaml
from typing import Optional, Tuple

import numpy as np

# Process-local cache of underlying ``AlfredTWEnv`` / ``AlfredThorEnv``
# instances keyed by ``(abs_alf_config_path, train_eval, env_type)``. The
# constructor runs ``collect_game_files`` which walks every traj_data.json
# under the dataset path — ~1 s for eval, ~tens of minutes to an hour for the
# 8810-directory train split on a network filesystem. The walk's result (game_files,
# config) is read-only after construction, and ``base_env.init_env(batch_size=1)``
# already returns a fresh textworld-gym wrapper per call, so sharing the
# ``base_env`` across all ALFWorldEnv instances in the env-server process is
# both safe and required to avoid /batch/reset timeouts during validation.
_BASE_ENV_CACHE: dict = {}
_BASE_ENV_CACHE_LOCK = threading.Lock()

def _alfworld_is_importable() -> bool:
    """Return True iff ``import alfworld`` works in the current interpreter."""
    return importlib.util.find_spec("alfworld") is not None


def _ensure_alfworld_on_path() -> None:
    """Require ALFWorld and install the synchronous TextWorld compatibility patch."""
    if not _alfworld_is_importable():
        raise ImportError(
            "Could not import ALFWorld. Install the upstream `alfworld` and "
            "`textworld` packages and download the required ALFWorld data."
        )
    _ensure_textworld_register_games_sync()


def _ensure_textworld_register_games_sync() -> None:
    """Force ``textworld.gym.register_games`` to use ``asynchronous=False``.

    The bundled ``AlfredTWEnv.init_env`` hard-codes ``asynchronous=True`` in
    its ``textworld.gym.register_games`` call. At ``batch_size=1`` (the TraceRigor
    pattern) that spawns one subprocess worker per ALFWorldEnv instance,
    talking to the env-server process via a ``multiprocessing.Pipe``. With
    128 train + 128 eval envs per service that adds ~256 textworld
    subprocesses on top of vLLM / Ray / Flask. Common failure modes we have
    actually hit in this configuration:

      * fd / nproc exhaustion under Slurm resource limits → silent stall.
      * Pipe deadlock when a single worker hangs (e.g. fast-downward
        spinning on a pathological ALFRED PDDL game) → the parent blocks
        forever on the worker's pipe and the entire ``service.reset_batch``
        loop wedges there.
      * Network filesystems amplifying the above because every fast-downward invocation
        in every worker opens PDDL files over the network.

    Synchronous mode is **functionally identical** at batch_size=1 (textworld
    doesn't parallelise within a single env), removes the deadlock surface
    entirely, and avoids spawning hundreds of unnecessary subprocesses.
    """
    try:
        import textworld.gym as tw_gym  # type: ignore
    except ImportError:
        return  # alfworld will fail later with a clearer error.

    if getattr(tw_gym, "_tracerigor_sync_patched", False):
        return

    _original_register_games = tw_gym.register_games

    def _register_games_sync(*args, **kwargs):
        # Force synchronous mode regardless of caller's choice.
        kwargs["asynchronous"] = False
        return _original_register_games(*args, **kwargs)

    tw_gym.register_games = _register_games_sync
    tw_gym._tracerigor_sync_patched = True


def load_alf_config(alf_config_path: str) -> dict:
    """Load a YAML alf-config file, expanding $ENV references in paths."""
    path = os.path.expandvars(alf_config_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ALFWorld config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _import_env_class(env_type: str):
    """Import the underlying AlfredTW/AlfredThor env class.

    Bypasses the upstream ``get_environment`` factory so the same code works
    for both pip-installed and bundled alfworld.
    """
    _ensure_alfworld_on_path()

    if env_type == "AlfredTWEnv":
        module = importlib.import_module("alfworld.agents.environment.alfred_tw_env")
        return module.AlfredTWEnv
    if env_type == "AlfredThorEnv":
        module = importlib.import_module("alfworld.agents.environment.alfred_thor_env")
        return module.AlfredThorEnv
    raise ValueError(
        f"Unsupported alfworld env type: {env_type!r}. "
        f"Expected 'AlfredTWEnv' or 'AlfredThorEnv'."
    )


def build_alfworld_env(
    alf_config_path: str,
    train_eval: str = "train",
    batch_size: int = 1,
    seed: Optional[int] = None,
    *,
    config_override: Optional[dict] = None,
) -> Tuple[object, object, str]:
    """Build a single ALFWorld environment.

    The underlying ``AlfredTWEnv`` / ``AlfredThorEnv`` factory is *cached* in
    a process-local dict keyed by
    ``(abs alf_config_path, train_eval, env_type)``. The factory's
    constructor walks every traj_data.json under the dataset path (cheap for
    eval, very expensive for train on a network filesystem), but its
    post-construction state is read-only and ``init_env(batch_size=1)``
    returns an independent textworld-gym wrapper on every call. Sharing the
    factory across ALFWorldEnv instances in the same process is therefore
    safe and turns a per-env O(N) walk into a one-time cost per split.

    Args:
        alf_config_path: Path to the alf-config.yaml file (may use ``$VAR``).
            This is also the cache key, so the *same* user-supplied path must
            be passed for every ALFWorldEnv that should share a factory.
            Do NOT pass a transient/tempfile path — that defeats the cache.
        train_eval: Dataset split — ``"train"``, ``"eval_in_distribution"``,
            or ``"eval_out_of_distribution"``.
        batch_size: Number of sub-environments. TraceRigor uses ``1``.
        seed: Optional seed forwarded to ``env.seed`` when supported.
        config_override: Optional pre-loaded / patched alf-config dict. When
            provided, the YAML at ``alf_config_path`` is NOT re-read; the
            dict is used as-is for factory construction. This lets callers
            (e.g. ``ALFWorldEnv``) inject in-memory overrides such as
            ``env.type=AlfredThorEnv`` for vision mode while keeping
            ``alf_config_path`` stable for cache-hit purposes.

    Returns:
        ``(env, base_env, env_type)`` where ``env`` is the initialised batched
        environment, ``base_env`` is the factory (possibly shared), and
        ``env_type`` is ``"AlfredTWEnv"`` or ``"AlfredThorEnv"``.
    """
    config = config_override if config_override is not None else load_alf_config(alf_config_path)
    env_type = config["env"]["type"]

    cache_key = (os.path.abspath(alf_config_path), train_eval, env_type)
    with _BASE_ENV_CACHE_LOCK:
        base_env = _BASE_ENV_CACHE.get(cache_key)
        if base_env is None:
            EnvCls = _import_env_class(env_type)
            base_env = EnvCls(config, train_eval=train_eval)
            _BASE_ENV_CACHE[cache_key] = base_env

    env = base_env.init_env(batch_size=batch_size)
    if seed is not None and hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass
    return env, base_env, env_type


def numpy_frame_to_pil(frame: np.ndarray):
    """Convert a HWC numpy frame (BGR or RGB uint8) to a PIL RGB image."""
    from PIL import Image

    if frame is None:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 3:
        # AlfredThorEnv.get_last_frame already returns BGR (note the [:, :, ::-1]
        # slice upstream); a second BGR->RGB flip here yields the correct order.
        return Image.fromarray(frame[:, :, ::-1], mode="RGB")
    raise ValueError(
        f"Unsupported frame shape for ALFWorld vision rendering: {frame.shape}"
    )


def get_thor_frame(env) -> Optional[np.ndarray]:
    """Return the most recent ThorEnv frame as a numpy array, or ``None``."""
    if not hasattr(env, "get_frames"):
        return None
    try:
        frames = env.get_frames()
    except Exception:
        return None
    if frames is None or len(frames) == 0:
        return None
    return np.asarray(frames[0])
