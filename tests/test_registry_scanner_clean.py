"""The vendored packages must pass the Comfy registry's YARA scan.

ComfyUI-IntoTheLatent-Utils ships `breeze_models` and `breeze_infer` inside the pack. The registry
scans every published file and *flags* a version (which hides it from ComfyUI Manager) on any hit.
Pack 1.10.0 was lost to two rules, both matching code in here:

- `python_environment_manipulation`: `os.environ.get(`, `os.environ[...]`, `os.environ[...] = `
- `python_network_operations`: `urllib.request.urlopen(`

The rules are plain text matches, so comments and strings count too: this test greps the source
instead of walking the AST. `breeze_infer/api.py` is exempt because the pack never vendors it.
Findings for a published version:
GET https://api.comfy.org/nodes/comfyui-intothelatent-utils/versions?include_status_reason=true
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = [ROOT / "breeze_models", ROOT / "breeze_infer"]
NOT_VENDORED = {ROOT / "breeze_infer" / "api.py"}

FORBIDDEN = re.compile(
    r"os\.environ|os\.getenv|os\.putenv|os\.unsetenv"
    r"|urllib\.request|urlopen|urlretrieve|http\.client|\brequests\.|\bhttpx\b|\baiohttp\b|\bsocket\."
)


def test_vendored_packages_have_no_env_or_network_access():
    offenders = []
    for pkg in PACKAGES:
        for py in pkg.rglob("*.py"):
            if py in NOT_VENDORED:
                continue
            for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    offenders.append(f"{py.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
