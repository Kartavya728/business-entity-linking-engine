#!/bin/bash
#SBATCH --job-name=ber_exp
#SBATCH --partition=medium-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/home/d22001/business-entity-linking-engine/work/logs/slurm_%x_%j.log
#SBATCH --error=/home/d22001/business-entity-linking-engine/work/logs/slurm_%x_%j.err
# run_experiments.sh (extra cross-encoders / LLM matchers + stage 2 + variants) on one 90 GB MIG slice.
# Usage: RUN=v11 EXTRA_CE="qwen3rr_4b" sbatch scripts/slurm_experiments.sh
#   every variable of run_experiments.sh (RUN, EXTRA_CE, REF, FORCE_CE, STAGE2_EACH) passes through
# Progress: tail -f /home/d22001/business-entity-linking-engine/work/logs/progress.txt

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L

# The submitting shell may export VIRTUAL_ENV / put another env first on PATH; with --export=ALL
# that leaks in, so the interpreter is pinned explicitly (common.sh uses $PY).
unset VIRTUAL_ENV
export PY=/home/d22001/miniconda3/envs/${BER_ENV:-ber}/bin/python

cd /home/d22001/business-entity-linking-engine/code/business_entity_resolution
export RUN=${RUN:-v11}
export EXTRA_CE=${EXTRA_CE:-qwen3rr_4b}

echo "python  : $($PY -c 'import sys; print(sys.executable)')"
echo "torch   : $($PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN=$RUN  EXTRA_CE=[$EXTRA_CE]  REF=${REF:-v6}  FORCE_CE=${FORCE_CE:-}  STAGE2_EACH=${STAGE2_EACH:-1}"

bash scripts/run_experiments.sh
rc=$?
echo "=== Done (exit $rc) ==="
exit $rc
