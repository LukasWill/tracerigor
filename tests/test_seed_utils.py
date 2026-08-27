import random
from types import SimpleNamespace

import numpy as np

from tracerigor.env.utils import env_utils


def test_permanent_seed_without_torch(monkeypatch):
    def unavailable(name):
        assert name == "torch"
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(env_utils, "import_module", unavailable)

    env_utils.permanent_seed(17)
    first = (random.random(), np.random.random())
    env_utils.permanent_seed(17)
    second = (random.random(), np.random.random())

    assert first == second


def test_permanent_seed_includes_torch_when_available(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
    )
    monkeypatch.setattr(env_utils, "import_module", lambda name: fake_torch)

    env_utils.permanent_seed(23)

    assert calls == [("cpu", 23), ("cuda", 23)]
