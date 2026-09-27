#!/usr/bin/env bash
# One-time setup: Python venv in code/business_entity_resolution/.venv + pretrained checkpoints.
# Usage: bash scripts/setup_env.sh            (PYBIN=python3.12 to choose the interpreter)
#        BER_HF_MODELS="..." bash scripts/setup_env.sh   to change the checkpoints to fetch
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${PYBIN:-python3}"
"$PYBIN" -m venv "$PKG/.venv"
"$PKG/.venv/bin/pip" install --upgrade pip wheel
"$PKG/.venv/bin/pip" install -r "$PKG/requirements.txt"
"$PKG/.venv/bin/python" - <<'EOF'
import os
from huggingface_hub import snapshot_download
# all MIT or Apache-2.0, <= 8B parameters
models = os.environ.get("BER_HF_MODELS", " ".join([
    "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large",
    "BAAI/bge-m3", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen3-4B-Base"])).split()
for m in models:
    print("downloading", m, flush=True)
    snapshot_download(m)
EOF
"$PKG/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
