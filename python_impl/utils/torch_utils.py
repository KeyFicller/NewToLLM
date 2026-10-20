import torch

from python_impl.utils.runtime import OPTIONS


def decl_device():
    if OPTIONS["device"]:
        return OPTIONS["device"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    return device
