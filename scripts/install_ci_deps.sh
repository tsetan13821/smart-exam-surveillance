#!/usr/bin/env bash
# Installs the Python dependencies for CI and the Docker image (CPU only, no compiler needed).
#   - PyTorch from the CPU wheel index (the default Linux wheel bundles CUDA: ~2 GB more)
#   - onnxruntime instead of onnxruntime-gpu
#   - insightface's Python code via scripts/install_insightface_nobuild.py
set -euo pipefail

cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

requirements="$(mktemp)"
grep -v '^insightface' requirements.txt | sed 's/^onnxruntime-gpu/onnxruntime/' > "$requirements"
# onnx, scikit-image, easydict: insightface's own runtime dependencies
python -m pip install -r "$requirements" onnx scikit-image easydict "$@"
rm -f "$requirements"

python scripts/install_insightface_nobuild.py
