# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline: **data → normalisation → GPU dense blocking → two-stage matcher
(XGBoost + cross-encoder) → F0.5-tuned decision → `output/*.tsv`**.
Only the provided training/test files are used (no external data or lookups).
All models are MIT/Apache-2.0 and ≤ 118M parameters.

## Environment

- Linux, Python 3.12, one CUDA GPU (developed on a 32 GB RTX PRO 4500), ~64 GB RAM, ~15 GB free disk.
- `pip install -r requirements.txt`
- The pretrained encoder `intfloat/multilingual-e5-small` (MIT) is downloaded from the
  Hugging Face hub on first use.

## Layout expected

```
<repo>/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<repo>/dataset/test/test_source{1,2,3}.tsv
<repo>/code/business_entity_resolution/        (this folder)
```
Paths can be overridden with `BER_DATA`, `BER_WORK` (intermediates, default `<repo>/work`)
and `BER_OUT` (default `<repo>/output`).

## Reproduce

```bash
cd code/business_entity_resolution/src
python -m ber.prepare --split train          # normalise + cache parquet
python -m ber.prepare --split test
python -m ber.train_biencoder                # bi-encoder v1 (in-batch negatives, E folds)
python -m ber.candidates --split train --enc ../../../work/biencoder --cap 60   # v1 candidates
# mine hard negatives from v1 candidates (see ber/train_biencoder.py --hard) then:
python -m ber.train_biencoder --hard --init ../../../work/biencoder --out ../../../work/biencoder_v2 --lr 2e-5
python -m ber.candidates --split train --cap 40 && python -m ber.candidates --split train --cap 20 --prune
python -m ber.candidates --split test  --cap 40 && python -m ber.candidates --split test  --cap 20 --prune
python -m ber.eval_blocking                  # recall vs cap on all training links
python -m ber.train_ce train                 # cross-encoder on E-fold candidates
python -m ber.build_features subset          # stage-1 training features (500K R-fold S1)
python -m ber.train_ranker stage1            # stage-1 XGBoost (4-fold OOF)
python -m ber.build_features score --split train   # p1 for every train candidate
python -m ber.train_ce score --split train
python -m ber.train_ranker stage2            # stage-2 XGBoost + decision tuning (OOF macro F0.5)
python -m ber.build_features score --split test
python -m ber.infer                          # CE on test, stage 2, decision, TSVs
python ../../../utils/validate_submission.py --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../dataset/test
```

Approximate wall time on the reference machine: ~4–5 h end to end.

## Modules (`src/ber/`)

| module | role |
|---|---|
| `config.py`, `io.py`, `splits.py` | paths, TSV I/O, S1-level folds (E = encoder folds 0–2, R = ranker folds 3–9) |
| `normalize.py` | transliteration, abbreviation/legal-form canonicalisation, DBA/domain handling, skeleton key |
| `biencoder.py`, `train_biencoder.py` | e5 bi-encoder with InfoNCE (+ mined hard negatives) |
| `retrieval.py`, `candidates.py` | GPU exact kNN per country (forward + reverse), capping, pruning |
| `features.py`, `build_features.py` | pair features (rapidfuzz, TF-IDF cosines, number logic), streamed |
| `crossencoder.py`, `train_ce.py` | e5 cross-encoder pair classifier |
| `ranker.py`, `train_ranker.py` | two-stage XGBoost with competition context, OOF training |
| `decide.py`, `metric.py` | exclusive assignment, expected-F0.5 selection, exact macro F0.5 |
| `infer.py` | test inference and submission files |
| `eval_blocking.py`, `gpu_sparse.py` | diagnostics; GPU inverted-index experiment (not in final path) |
