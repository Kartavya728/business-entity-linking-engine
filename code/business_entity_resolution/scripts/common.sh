# Shared settings and helpers for the pipeline scripts. Source it (do not run it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"                     # code/business_entity_resolution
ROOT="$(cd "$PKG/../.." && pwd)"                  # repository root (dataset/, work/, output/)
export BER_ROOT="${BER_ROOT:-$ROOT}"
export BER_DATA="${BER_DATA:-$BER_ROOT/dataset}"
export BER_WORK="${BER_WORK:-$BER_ROOT/work}"
export BER_OUT="${BER_OUT:-$BER_ROOT/output}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-$PKG/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
SRC="$PKG/src"
LOG="$BER_WORK/logs"
mkdir -p "$LOG" "$BER_WORK/models" "$BER_OUT"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
GPUS=(${GPUS:-$(seq -s ' ' 0 $((NGPU - 1)))})    # physical GPU ids to use, e.g. GPUS="0 1 2 3"
NGPU=${#GPUS[@]}

step() { echo "$(date '+%F %T') $*" | tee -a "$LOG/progress.txt"; }

# run a pipeline module in the foreground: ber <module> [args...]
ber() { (cd "$SRC" && "$PY" -u -m "ber.$1" "${@:2}"); }

# run a pipeline module in the background on one GPU: on_gpu <slot> <logfile> <module> [args...]
on_gpu() {
  local slot=$1 log=$2; shift 2
  (cd "$SRC" && CUDA_VISIBLE_DEVICES="${GPUS[$((slot % NGPU))]}" "$PY" -u -m "ber.$1" "${@:2}" > "$LOG/$log" 2>&1) &
}

wait_all() {
  local rc=0
  for p in $(jobs -p); do wait "$p" || rc=1; done
  if [ "$rc" -ne 0 ]; then echo "a background job failed - see $LOG" >&2; exit 1; fi
}

ce_out() { (cd "$SRC" && "$PY" -c "from ber.ce_registry import CE_MODELS; print(CE_MODELS['$1']['out'])"); }

# train cross-encoders in parallel, one per GPU. Items are NAME[:N_S1[:EPOCHS]] (0 = registry default);
# models already trained are skipped unless FORCE_CE=1
train_ces() {
  local slot=0
  for item in "$@"; do
    IFS=: read -r m ns ep <<< "$item"
    if [ -f "$(ce_out "$m")/head.pt" ] && [ -z "${FORCE_CE:-}" ]; then echo "  $m already trained - skip"; continue; fi
    step "  train cross-encoder $m (n_s1=${ns:-default}, epochs=${ep:-default}) on GPU ${GPUS[$((slot % NGPU))]}"
    on_gpu "$slot" "ce_train_$m.log" train_ce train --model "$m" --n-s1 "${ns:-0}" --epochs "${ep:-0}"
    slot=$((slot + 1))
    if [ $((slot % NGPU)) -eq 0 ]; then wait_all; fi
  done
  wait_all
}

ce_names() { for item in "$@"; do echo "${item%%:*}"; done; }

# score cross-encoders on train and test; each model is sharded over all GPUs, then merged
score_ces() {
  for m in "$@"; do
    for s in train test; do
      step "  score $m on $s ($NGPU shards)"
      if [ "$NGPU" -eq 1 ]; then
        ber train_ce score --split "$s" --model "$m" > "$LOG/ce_score_${m}_$s.log" 2>&1
      else
        for i in $(seq 0 $((NGPU - 1))); do
          on_gpu "$i" "ce_score_${m}_${s}_$i.log" train_ce score --split "$s" --model "$m" --shard "$i" --nshard "$NGPU"
        done
        wait_all
        ber train_ce merge --split "$s" --model "$m" --nshard "$NGPU" >> "$LOG/ce_score_${m}_$s.log" 2>&1
      fi
    done
  done
}

# stage 2 (XGBoost + LightGBM, word-difference features) + test inference; keeps a copy per run
stage2_and_infer() {
  local run=$1 M="$BER_WORK/models"
  mkdir -p "$M/prev" && cp -f "$M"/stage2* "$M/prev/" 2>/dev/null || true
  rm -f "$M"/stage2_cv*.json "$M"/stage2lgb_cv*.txt          # fold models are resumable: never reuse stale ones
  BER_LGB=1 BER_DT=1 ber train_ranker stage2 > "$LOG/stage2_$run.log" 2>&1
  grep "best OOF" "$LOG/stage2_$run.log" | tee -a "$LOG/progress.txt"
  cp "$M/decision.json" "$M/decision_$run.json"
  BER_DT=1 ber infer --skip-ce > "$LOG/infer_$run.log" 2>&1
  cp "$BER_WORK/test/p2.parquet" "$BER_WORK/test/p2_$run.parquet"
}

validate() {  # validate <matching_results.tsv>
  "$PY" "$BER_ROOT/utils/validate_submission.py" --matching "$1" --candidate "$BER_OUT/candidate_pairs.tsv" \
    --test-dir "$BER_DATA/test" --check-ids 2>&1 | tail -1
}
