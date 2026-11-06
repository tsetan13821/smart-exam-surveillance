"""Installs insightface 0.7.3 without a C++ compiler (for Windows without the Build Tools).

`pip install insightface` compiles a Cython extension that is only used to render
face masks, which SmartSquad never does. This script downloads the official source
package from PyPI, checks its SHA-256 against PyPI's, copies the Python package into
the current environment and disables the one import that needs the compiled part.

    .\\.venv\\Scripts\\python.exe scripts\\install_insightface_nobuild.py

insightface's own dependencies (onnx, scikit-image, easydict, ...) must be installed
separately; see the README.
"""
import hashlib
import io
import json
import os
import shutil
import sysconfig
import tarfile
import tempfile
import urllib.request

VERSION = "0.7.3"
DISABLED_IMPORT = "from .mask_renderer import *"


def main():
    with urllib.request.urlopen(f"https://pypi.org/pypi/insightface/{VERSION}/json", timeout=30) as r:
        release = json.load(r)
    sdist = next(f for f in release["urls"] if f["packagetype"] == "sdist")
    with urllib.request.urlopen(sdist["url"], timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sdist["digests"]["sha256"]:
        raise SystemExit("Checksum mismatch: the download does not match PyPI. Nothing was installed.")

    site_packages = sysconfig.get_paths()["purelib"]
    target = os.path.join(site_packages, "insightface")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            try:
                tar.extractall(tmp, filter="data")      # refuses paths outside tmp
            except TypeError:                           # Python older than 3.10.12 / 3.11.4
                root = os.path.realpath(tmp)
                for member in tar.getmembers():
                    dest = os.path.realpath(os.path.join(tmp, member.name))
                    if not dest.startswith(root + os.sep) or member.issym() or member.islnk():
                        raise SystemExit(f"Unsafe path in archive: {member.name}")
                tar.extractall(tmp)
        source = os.path.join(tmp, f"insightface-{VERSION}", "insightface")
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(source, target)

    init = os.path.join(target, "app", "__init__.py")
    with open(init, encoding="utf-8") as f:
        text = f.read()
    with open(init, "w", encoding="utf-8") as f:
        f.write(text.replace(DISABLED_IMPORT, f"# {DISABLED_IMPORT}  # needs the compiled extension; unused"))

    print(f"Installed insightface {VERSION} (without mask rendering) into {target}")


if __name__ == "__main__":
    main()
