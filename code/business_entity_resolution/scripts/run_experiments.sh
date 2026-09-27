#!/usr/bin/env bash
# DGX experiment run on top of existing artefacts (work/ copied from the development machine or
# produced by run_pipeline.sh): train extra cross-encoders in parallel (one per GPU), score them
# sharded over all GPUs, retrain stage 2 with every cross-encoder, infer, build variants.
#
# Usage: RUN=v10 EXTRA_CE="large2 bgem3 qwen15" bash scripts/run_experiments.sh
#   EXTRA_CE : cross-encoders to add, NAME[:N_S1[:EPOCHS]] (names in src/ber/ce_registry.py),
#              e.g. EXTRA_CE="large2:250000 bgem3:250000:1 qwen15:100000"
#   RUN      : tag for this run's outputs (p2_<RUN>.parquet, decision_<RUN>.json, submissions/<RUN>_*)
#   REF      : France reference run for the hybrid variants (default v6)
#   GPUS     : physical GPU ids to use (default: all), e.g. GPUS="0 1 2 3"
#   STAGE2_EACH : with one GPU (default 1) retrain stage 2 after every model -> <RUN>_1, <RUN>_2, ...
source "$(dirname "$0")/common.sh"
RUN=${RUN:-v10}
EXTRA_CE=${EXTRA_CE:-qwen25_7b qwen3_4b large2}
REF=${REF:-v6}
for f in train/p1.parquet test/p1.parquet train/candidates.parquet train/gt.parquet train/subset_s1.npy; do
  [ -f "$BER_WORK/$f" ] || { echo "missing $BER_WORK/$f - copy work/ first or run run_pipeline.sh" >&2; exit 1; }
done
step "== experiment $RUN on $NGPU GPU(s): extra cross-encoders [$EXTRA_CE]"
if [ "$NGPU" -eq 1 ] && [ "${STAGE2_EACH:-1}" = "1" ]; then
  # single GPU: model by model; stage 2 + variants after each one, so every finished model
  # already yields submission files (<RUN>_1_*, <RUN>_2_*, ...)
  i=0
  for item in $EXTRA_CE; do
    i=$((i + 1))
    train_ces "$item"
    score_ces "${item%%:*}"
    step "stage 2 + inference (${RUN}_$i: + ${item%%:*})"
    stage2_and_infer "${RUN}_$i"
    bash "$HERE/make_variants.sh" "${RUN}_$i" "$REF"
  done
else
  train_ces $EXTRA_CE
  score_ces $(ce_names $EXTRA_CE)
  step "stage 2 + inference ($RUN)"
  stage2_and_infer "$RUN"
  bash "$HERE/make_variants.sh" "$RUN" "$REF"
fi
step "== $RUN done: submission files in $BER_ROOT/submissions/${RUN}*"
