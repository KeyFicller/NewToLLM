"""Runtime options + shared config (configs/toy_gpt2.json).

Mirrors cpp_impl/my_torch/utils/runtime.{h,cpp}. Set from main.py flags:
--seed, --device, --small.
"""

import copy
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "toy_gpt2.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    SHARED_CONFIG = json.load(f)

# None = use config / auto-detect.
OPTIONS = {"seed": None, "device": None, "small": False}


def set_options(seed=None, device=None, small=False):
    OPTIONS.update(seed=seed, device=device, small=small)


def seed():
    return OPTIONS["seed"] if OPTIONS["seed"] is not None else SHARED_CONFIG["seed"]


def model_config(allow_small=True):
    """Toy model config. `allow_small=False` for tasks that load GPT-2 weights."""
    key = "model_small" if (allow_small and OPTIONS["small"]) else "model"
    return copy.deepcopy(SHARED_CONFIG[key])


def train_config():
    return copy.deepcopy(SHARED_CONFIG["train"])
