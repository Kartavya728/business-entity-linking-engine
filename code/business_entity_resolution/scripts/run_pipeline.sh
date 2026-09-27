#!/usr/bin/env bash
# Full reproduction from the raw TSVs (dataset/{train,test}/*.tsv) to output/*.tsv.
# Independent GPU jobs (cross-encoders) run in parallel over all visible GPUs.
#
# Usage: bash scripts/run_pipeline.sh
#   CE_LIST : cross-encoders of the base run (default: the v8 set "small base canon large")
#   EXTRA_CE: models added afterwards one by one, each followed by stage 2 + variants
#             (default "qwen25_7b qwen3_4b"; EXTRA_CE="" to stop after the base run)
#   RUN     : tag for this run (default v8); the extra models give <RUN>_llm_1, <RUN>_llm_2, ...
#   START   : resume from a stage: prepare|biencoder|blocking|stage1|ce|stage2 (default prepare)
source "$(dirname "$0")/common.sh"
CE_LIST=${CE_LIST:-small base canon large}
EXTRA_CE=${EXTRA_CE-qwen25_7b qwen3_4b}
RUN=${RUN:-v8}
START=${START:-prepare}
stages=(prepare biencoder blocking stage1 ce stage2)
go=0; for s in "${stages[@]}"; do [ "$s" = "$START" ] && go=1; declare "do_$s=$go"; done
[ "$go" -eq 1 ] || { echo "unknown START=$START" >&2; exit 1; }
cd "$SRC"

if [ "$do_prepare" -eq 1 ]; then
  step "prepare: normalise records, filler words, ground truth"
  ber prepare --split train > "$LOG/prepare_train.log" 2>&1
  ber prepare --split test > "$LOG/prepare_test.log" 2>&1
fi
if [ "$do_biencoder" -eq 1 ]; then
  step "bi-encoder round 1 (in-batch negatives)"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} ber train_biencoder --out "$BER_WORK/biencoder" > "$LOG/biencoder.log" 2>&1
  step "round-1 blocking on train + hard-negative mining"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} ber candidates_v5 --split train --enc "$BER_WORK/biencoder" > "$LOG/cand_round1.log" 2>&1
  ber candidates_v5 --split train --prune 20 5 >> "$LOG/cand_round1.log" 2>&1
  ber mine_hardneg >> "$LOG/cand_round1.log" 2>&1
  step "bi-encoder round 2 (mined hard negatives)"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} ber train_biencoder --hard --init "$BER_WORK/biencoder" \
      --out "$BER_WORK/biencoder_v2" --lr 2e-5 > "$LOG/biencoder_v2.log" 2>&1
fi
if [ "$do_blocking" -eq 1 ]; then
  step "hybrid blocking: dense + reverse dense + GPU IDF-sparse (train, test)"
  for s in train test; do
    CUDA_VISIBLE_DEVICES=${GPUS[0]} ber candidates_v5 --split "$s" > "$LOG/cand_$s.log" 2>&1
    ber candidates_v5 --split "$s" --prune 20 5 >> "$LOG/cand_$s.log" 2>&1
  done
fi
if [ "$do_stage1" -eq 1 ]; then
  step "stage 1: pair features, XGBoost, p1 for every candidate"
  ber build_features subset > "$LOG/feat_subset.log" 2>&1
  CUDA_VISIBLE_DEVICES=${GPUS[0]} ber train_ranker stage1 > "$LOG/stage1.log" 2>&1
  for s in train test; do
    CUDA_VISIBLE_DEVICES=${GPUS[0]} ber build_features score --split "$s" > "$LOG/p1_$s.log" 2>&1
  done
fi
if [ "$do_ce" -eq 1 ]; then
  step "cross-encoders [$CE_LIST]: train (parallel) and score (sharded)"
  train_ces $CE_LIST
  score_ces $(ce_names $CE_LIST)
fi
step "stage 2 + inference ($RUN)"
stage2_and_infer "$RUN"
bash "$HERE/make_variants.sh" "$RUN"
step "base run done: submissions/${RUN}_*"
if [ -n "$EXTRA_CE" ]; then
  RUN="${RUN}_llm" EXTRA_CE="$EXTRA_CE" bash "$HERE/run_experiments.sh"
fi
step "done: output/matching_results.tsv, output/candidate_pairs.tsv, submissions/${RUN}*"
