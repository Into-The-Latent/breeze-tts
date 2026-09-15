"""Breeze TTS inference runtime package."""
import torch as _torch

# The package metadata deliberately declares `torch` without a version floor: a floor makes pip
# replace the host's torch build (a ComfyUI venv on Windows then gets the CPU-only wheel from PyPI,
# which breaks ComfyUI itself). The floor is enforced here instead, with a message that says what
# to do. torch.compiler.is_compiling (breeze_models/breeze.py) appeared in torch 2.3.
_MIN_TORCH = "2.3"

if _torch.__version__ < _MIN_TORCH:  # TorchVersion compares against a version string
    # RuntimeError on purpose: callers wrap ImportError as "package not installed", which this is not.
    raise RuntimeError(
        f"breeze-tts needs torch >= {_MIN_TORCH}, found {_torch.__version__}. The package does not pin "
        "torch, so pip never replaces your torch build; upgrade torch yourself with a wheel that matches "
        "your GPU (see https://pytorch.org/get-started/locally/)."
    )
