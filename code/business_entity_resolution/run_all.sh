#!/usr/bin/env bash
# End-to-end reproduction: data -> blocking -> matching -> output/*.tsv
# Run from code/business_entity_resolution/. Expects dataset/ at the repo root
# (override with BER_DATA / BER_WORK / BER_OUT environment variables).
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PY:-python}
$PY -m ber.prepare --split train
$PY -m ber.prepare --split test
$PY -m ber.train_biencoder                 # dense blocking encoder (E folds)
$PY -m ber.candidates --split train        # blocking on train
$PY -m ber.candidates --split test         # blocking on test -> candidate set
$PY -m ber.eval_blocking                   # recall diagnostics
$PY -m ber.build_features --split train
$PY -m ber.build_features --split test
$PY -m ber.train_ranker stage1             # GBDT stage 1 (+ OOF p1)
$PY -m ber.train_ce train                  # cross-encoder on E folds
$PY -m ber.train_ce score --split train
$PY -m ber.train_ranker stage2             # GBDT stage 2 + decision tuning
$PY -m ber.infer                           # test inference + TSVs
$PY ../../../utils/validate_submission.py --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../dataset/test
