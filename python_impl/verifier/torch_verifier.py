import torch

from python_impl.utils.torch_utils import decl_device

def verify_torch():
    # Verify version
    print(f"[verify] Torch version is: {torch.__version__}")

    # Verify device
    print(f"[verify] Torch device should be: {decl_device()}")