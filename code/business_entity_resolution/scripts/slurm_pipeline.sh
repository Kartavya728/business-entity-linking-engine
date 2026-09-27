#!/bin/bash
#SBATCH --job-name=ber_pipeline
#SBATCH --partition=medium-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/d22001/business-entity-linking-engine/work/logs/slurm_%x_%j.log
#SBATCH --error=/home/d22001/business-entity-linking-engine/work/logs/slurm_%x_%j.err
# Full reproduction (run_pipeline.sh: raw TSVs -> output/*.tsv) on one 90 GB MIG slice.
# Usage: sbatch scripts/slurm_pipeline.sh
#        START=ce sbatch scripts/slurm_pipeline.sh      resume from a stage
#   every variable of run_pipeline.sh (START, RUN, CE_LIST, EXTRA_CE, FORCE_CE) passes through
# Progress: tail -f /home/d22001/business-entity-linking-engine/work/logs/progress.txt

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L

# The submitting shell may export VIRTUAL_ENV / put another env first on PATH; with --export=ALL
# that leaks in, so the interpreter is pinned explicitly (common.sh uses $PY).
unset VIRTUAL_ENV
export PY=/home/d22001/miniconda3/envs/${BER_ENV:-ber}/bin/python

cd /home/d22001/business-entity-linking-engine/code/business_entity_resolution

echo "python  : $($PY -c 'import sys; print(sys.executable)')"
echo "torch   : $($PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "START=${START:-prepare}  RUN=${RUN:-v8}  CE_LIST=[${CE_LIST:-small base canon large}]  EXTRA_CE=[${EXTRA_CE-qwen25_7b qwen3_4b}]"

bash scripts/run_pipeline.sh
rc=$?
echo "=== Done (exit $rc) ==="
exit $rc
