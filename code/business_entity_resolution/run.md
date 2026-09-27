# How to run

## Folder structure

```
business-entity-linking-engine/          <- repository root (BER_ROOT)
├── dataset/                             <- competition data (not in git)
│   ├── train/ train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/  test_source1.tsv   test_source2.tsv   test_source3.tsv
├── utils/validate_submission.py         <- official format checker
├── code/business_entity_resolution/
│   ├── src/ber/                         <- pipeline modules (python -m ber.<module>)
│   ├── scripts/                         <- entry points (this file explains them)
│   ├── reference/                       <- v6 French probabilities keyed by entity ids (for hybrids)
│   └── requirements.txt  run.md  README.md  package_submission.sh
├── work/                                <- all intermediates, created by the pipeline (not in git)
│   ├── train/ test/                     <- prepared parquet, candidates, p1, cross-encoder scores, p2
│   ├── models/                          <- stage-1/2 models, decision_<run>.json
│   ├── crossencoder_*/  biencoder*/     <- trained neural models
│   └── logs/                            <- one log per step + progress.txt (timeline)
├── output/                              <- matching_results.tsv, candidate_pairs.tsv (final)
└── submissions/<run>_*/                 <- candidate leaderboard files (+ NOTES.txt)
```

The paths can be moved with the `BER_DATA`, `BER_WORK` and `BER_OUT` environment variables.

## Route A: copy the finished artefacts with rsync (recommended)

This route skips about 10 hours of recomputation (blocking, stage 1, the existing cross-encoders).
Run these commands on the **development machine**; each asks for the DGX password.

```bash
cd /home/kartavya/business-entity-linking-engine
# 1) code, data, docs (~2.4 GB)
rsync -avP --rsync-path="mkdir -p ~/business-entity-linking-engine && rsync" \
  --exclude '__pycache__' --exclude '.venv' \
  code utils dataset architecture.md Documentation_template.md \
  d22001@dgx-login.iitmandi.ac.in:~/business-entity-linking-engine/
# 2) pipeline artefacts (~22 GB). Resumable: re-run the same command if it stops.
bash code/business_entity_resolution/scripts/pack_artifacts.sh --list > work/artifact_files.txt
rsync -avP --files-from=work/artifact_files.txt . \
  d22001@dgx-login.iitmandi.ac.in:~/business-entity-linking-engine/
```

After a code change, re-run command 1; only the changed files are sent.

## Route B: fresh clone (no artefacts)

```bash
git clone https://github.com/Kartavya728/business-entity-linking-engine && cd business-entity-linking-engine
mkdir -p dataset/train dataset/test     # then put the 7 competition TSVs there (see the tree above)
```

Then run `scripts/run_pipeline.sh`, which rebuilds everything (section 4).

## 1. Environment (once, on the DGX)

```bash
cd ~/business-entity-linking-engine/code/business_entity_resolution
bash scripts/setup_env.sh      # venv + requirements + checkpoints (~40 GB incl. Qwen2.5-7B); PYBIN=python3.12 to choose
```

It ends by printing `torch ... cuda True gpus 1`. If CUDA is `False`, install the matching wheel
(`.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`) and re-run.

If the DGX uses SLURM, first get a GPU shell
(`srun --gres=gpu:1 --cpus-per-task=32 --mem=200G --time=12:00:00 --pty bash`) or wrap the run commands
in `sbatch`. Use `tmux` or `nohup` so a dropped SSH session does not stop the run.

## 2. Run the LLM matchers on the single B200 (Route A)

```bash
cd ~/business-entity-linking-engine/code/business_entity_resolution
RUN=v11 EXTRA_CE="qwen25_7b qwen3_4b" nohup bash scripts/run_experiments.sh > ../../work/logs/v11.out 2>&1 &
tail -f ../../work/logs/progress.txt
```

On one GPU the models run one after another, and **stage 2 + variants are rebuilt after each model**:

| Step | Model | Estimated B200 time | Output |
|---|---|---|---|
| 1 | `qwen25_7b`: Qwen2.5-7B (7.6B, Apache-2.0), LoRA r=16, hard-pair band | train ~0.5 h, score ~2–3 h, stage 2 ~20 min | `submissions/v11_1_*` |
| 2 | `qwen3_4b`: Qwen3-4B-Base (4.0B, Apache-2.0), LoRA r=32, hard-pair band | train ~0.3 h, score ~1.5 h, stage 2 ~20 min | `submissions/v11_2_*` |

The times are estimates (not measured on a B200). Check `work/logs/ce_train_<model>.log` and
`ce_score_<model>_*.log` for real progress.

The **hard-pair band** (0.02 ≤ p1 < 0.995, the same rule on train and test) restricts the LLMs to the
2.4M train / 2.2M test pairs where stage 1 is not already near-certain, instead of 12M / 9.5M.

Each stage-2 log line `best OOF macro F0.5 = …` is the US/India validation score. Compare it with
**0.99077** (v8, the current best).

Model definitions live in `src/ber/ce_registry.py`, with syntax `NAME[:E-fold S1 entities[:epochs]]`.
Every backbone is MIT or Apache-2.0 and ≤ 8B parameters; Qwen3-8B (8.2B) is deliberately absent.
Other ready entries:
- `large2`: second e5-large, 560M.
- `bgem3`: BGE-M3, 568M.
- `qwen15`: Qwen2.5-1.5B, full fine-tune.
- `qwen3rr`: Qwen3-Reranker-0.6B.
- `qwen3_17`: Qwen3-1.7B-Base.

Not used: `mdeberta` diverges to NaN with transformers 5.8.

Options:
- `EXTRA_CE="qwen25_7b:60000 qwen3_4b"` trains on fewer entities.
- `FORCE_CE=1` retrains existing models.
- `STAGE2_EACH=0` runs a single stage 2 at the end.

## 3. Which files to submit

`scripts/make_variants.sh <RUN>` runs automatically after every stage 2 and writes, for each run:

| Folder | Content |
|---|---|
| `<RUN>_all` | every country from this run |
| `<RUN>_hybrid` | US/India from this run, France from v6 (best French leaderboard so far) |
| `<RUN>_hybrid_em` | hybrid + EM label-shift correction for France (Saerens et al. 2002) |
| `<RUN>_frblend` | US/India from this run, France = mean of v6 and this run |

All files are validated by `utils/validate_submission.py`. Only compare France variants against each
other when their US/India rows are identical: each leaderboard difference is then exactly
0.1498 × ΔF(France).

To build a single variant by hand (from `src/`):

```bash
../.venv/bin/python -m ber.combine --out ../../../submissions/X \
    --default p2_v11_1.parquet --default-cfg decision_v11_1.json \
    --country France --src ../reference/p2_v6_france.parquet --src-cfg ../reference/decision_v6.json --odds 0.6
```

The other options are `--em`, `--empty`, and `--src2 FILE --w2 0.5` to blend.

## 4. Full reproduction (Route B, or to rebuild)

```bash
bash scripts/run_pipeline.sh     # START=blocking|stage1|ce|stage2 resumes from that stage
```

The stages run in order:

1. **prepare:** normalisation, fillers, French locale, ground truth.
2. **Bi-encoders:** round 1, then hard-negative mining, then round 2.
3. **Hybrid blocking.**
4. **Stage 1** and p1 for every candidate.
5. **Cross-encoders** `CE_LIST` (default `small base canon large`).
6. **Stage 2** and the base-run variants, then the `EXTRA_CE` LLM matchers (default
   `qwen25_7b qwen3_4b`) one by one, each followed by stage 2 (`<RUN>_llm_1`, `<RUN>_llm_2`).

It takes about 10–14 h on the development GPU; a B200 is several times faster for the neural parts.

## 5. Final zip

```bash
bash scripts/package_final.sh ../../submissions/<best>/matching_results.tsv <team_name>
```

This writes:

| File | Content |
|---|---|
| `output/matching_results.tsv` | the chosen file |
| `output/candidate_pairs.tsv` | the stage-1 filtered blocking output (p1 ≥ 0.001) plus every predicted match: 6.4 candidates per S1 instead of 46, 99.83% train recall |

It validates both and zips them with `code/business_entity_resolution/` (without `.venv`) and
`Documentation_template.md`.

## Leaderboard history (for reading new results)

| Version | Change | Leaderboard |
|---|---|---|
| v5 | hybrid blocking, fillers, 3 cross-encoders, XGBoost + LightGBM | 0.984567 |
| v6 | French filler words | 0.985093 |
| v6b | v6 + accept French same-address near-name pairs | 0.985021 |
| v8 | e5-large field-structured cross-encoder, word-difference features, French address fix | 0.985164 |
| v9a–v9d | hybrid / EM France / strong France correction / France-empty probe | pending |

Validation (US/India, out of fold) is at 0.99077, close to the ambiguity ceiling: about 81% of the
remaining loss is empty-address records whose name is shared by 3 or more businesses. The
leaderboard gap is France (test-only, no labels), which is tuned through variants that change French
rows only.

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA out of memory | lower `bs` / `score_bs` for that model in `ce_registry.py` |
| A step failed | its log is in `work/logs/`; fix it and re-run. Trained models are skipped, and scoring is incremental. |
| Stage 2 killed (RAM) | keep `BER_S2_MIN_P1` at 0.0005 or higher; do not run other large jobs at the same time |
| No hybrid variants | `reference/p2_v6_france.parquet` or `work/test/p2_v6.parquet` is missing |
