import json
import subprocess
import sys

import pytest
import torch

from conftest import REPO_ROOT


def run_main(*args):
    return subprocess.run(
        [sys.executable, "main.py", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_list_tasks():
    out = run_main("list")
    assert out.returncode == 0
    for task in [
        "verify",
        "brief",
        "toy_model",
        "train",
        "load_model",
        "fine_tuning_classify",
        "fine_tuning_instruction",
    ]:
        assert task in out.stdout


def test_verify_cpu():
    out = run_main("verify", "--device", "cpu")
    assert out.returncode == 0
    assert "Torch device should be: cpu" in out.stdout


def test_unknown_task_fails():
    assert run_main("no_such_task").returncode != 0


def test_config_comes_from_shared_json():
    from python_impl.toy_model.config import ToyModelConfig

    shared = json.loads((REPO_ROOT / "configs" / "toy_gpt2.json").read_text())
    assert ToyModelConfig == shared["model"]


@pytest.mark.parametrize("small", [True])
def test_small_model_forward_shape(small):
    from python_impl.toy_model.model import ToyModel
    from python_impl.utils import runtime

    runtime.set_options(small=small)
    cfg = runtime.model_config()
    torch.manual_seed(0)
    model = ToyModel(cfg).eval()
    x = torch.randint(0, cfg["vocab_size"], (2, 8))
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 8, cfg["vocab_size"])
    runtime.set_options()
