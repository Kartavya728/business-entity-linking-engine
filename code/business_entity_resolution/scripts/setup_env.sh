#!/bin/bash
# One-time setup: conda env "ber" (Python 3.12) + requirements + pretrained checkpoints (~40 GB).
# Usage (on the LOGIN node, no GPU needed):
#   nohup bash scripts/setup_env.sh > ../../work/logs/setup_env.log 2>&1 &
#   BER_ENV       : conda env name (default ber)
#   BER_HF_MODELS : space-separated checkpoints to fetch instead of the default list
# Run it on the login node, not through sbatch: measured on 2026-09-27, the GPU node dgx-b200
# downloads from PyPI at ~0.3 MB/s and from Hugging Face at ~10 MB/s, the login node at 58 and
# 39 MB/s. /home is shared, so the env and the HF cache are visible to the SLURM jobs.
# The training jobs (slurm_*.sh) print the CUDA check on the GPU.
set -eo pipefail
echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="

# a VIRTUAL_ENV exported by the submitting shell would shadow the conda env
unset VIRTUAL_ENV
CONDA=/home/d22001/miniconda3/bin/conda
ENV=${BER_ENV:-ber}
PY=/home/d22001/miniconda3/envs/$ENV/bin/python
PKG=/home/d22001/business-entity-linking-engine/code/business_entity_resolution

# Create a fresh environment with Python 3.12 (skipped if it already exists)
[ -x "$PY" ] || $CONDA create -n "$ENV" python=3.12 -y -c conda-forge --override-channels

# PyTorch 2.11.0 with CUDA 12.8 (Blackwell / B200 needs >= cu128), then the pinned requirements
$PY -m pip install --upgrade pip wheel
$PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -r "$PKG/requirements.txt"

# Pretrained checkpoints into the Hugging Face cache (all MIT or Apache-2.0, <= 8B parameters)
$PY - <<'EOF'
import os
import time
from huggingface_hub import snapshot_download
models = os.environ.get("BER_HF_MODELS", " ".join([
    "Qwen/Qwen2.5-7B", "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-0.6B",   # default LLM matchers first
    "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large",
    "BAAI/bge-m3", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"])).split()
for m in models:
    for attempt in range(5):  # the hub sometimes drops connections; files already fetched are kept
        try:
            print("downloading", m, flush=True)
            snapshot_download(m)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e!r}", flush=True)
            time.sleep(30)
    else:
        raise SystemExit(f"could not download {m}")
EOF

echo "python  : $($PY -c 'import sys; print(sys.executable)')"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(),
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=== Setup Complete ==="
