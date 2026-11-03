"""Loading must not require the optional `accelerate` package.

`from_pretrained(..., device_map=...)` makes transformers raise "requires `accelerate`" when it is
not installed, and a fresh ComfyUI venv does not have it. accelerate also declares a torch floor,
which this package must never pull in (see test_requirements.py). Models are loaded on CPU and
moved with `.to(device)` instead.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = [ROOT / "breeze_models", ROOT / "breeze_infer"]


def _py_files():
    for pkg in PACKAGES:
        yield from pkg.rglob("*.py")


def test_no_device_map_keyword_in_calls():
    offenders = []
    for py in _py_files():
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(k.arg == "device_map" for k in node.keywords):
                offenders.append(f"{py.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


def test_accelerate_is_never_imported():
    offenders = []
    for py in _py_files():
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "accelerate" for a in node.names):
                offenders.append(f"{py.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "accelerate":
                offenders.append(f"{py.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders
