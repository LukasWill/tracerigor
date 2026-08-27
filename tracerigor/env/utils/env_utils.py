# env_utils.py
import os
import random
import logging
from importlib import import_module
import numpy as np
from datetime import timezone, datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager

UTC = timezone.utc

def permanent_seed(seed: int) -> None:
    """Seed CPU RNGs and PyTorch too when the RLVR stack is installed."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def set_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.Formatter.converter = lambda *args: (datetime.now(UTC) - timedelta(hours=2)).timetuple()
    return logging.getLogger()


@contextmanager
def NoLoggerWarnings():
    from gym import logger
    logger.set_level(logger.ERROR)
    try:
        yield
    finally:
        logger.set_level(logger.INFO)
