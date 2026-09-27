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

## 1. Environment (once, on the DGX login node)

```bash
cd ~/business-entity-linking-engine/code/business_entity_resolution
nohup bash scripts/setup_env.sh > ../../work/logs/setup_env.log 2>&1 &   # conda env "ber" + checkpoints
```

Run it on the **login node**, not through sbatch. The GPU node `dgx-b200` downloads from PyPI at
~0.3 MB/s and from Hugging Face at ~10 MB/s, the login node at up to 58 and 33–39 MB/s (measured
2026-09-27; PyPI's CDN is also throttled on the login node at times). `/home` is shared, so the env
(`~/miniconda3/envs/ber`, `BER_ENV=<name>` for another name) and the Hugging Face cache are visible
to every job. The training jobs print the CUDA check on the GPU.

All GPU jobs run through SLURM on the `medium-b200` partition with one **MIG 3g.90gb slice**
(90 GB VRAM, `--gres=gpu:nvidia_b200_3g.90gb:1`). Each QoS (`qos_medium`, `qos_small`, `qos_full`)
allows at most 2 jobs / 2 GPUs per user; jobs on other partitions can also count against
`qos_medium` (the default QoS).

| Script | Wraps | Default resources |
|---|---|---|
| `scripts/setup_env.sh` | environment + checkpoints (login node) | – |
| `scripts/slurm_experiments.sh` | `run_experiments.sh` | 24 CPU, 120 GB, 24 h |
| `scripts/slurm_pipeline.sh` | `run_pipeline.sh` | 32 CPU, 200 GB, 24 h |

Override any of them on the command line, e.g. `sbatch --time=12:00:00 scripts/slurm_experiments.sh`.
Environment variables (`RUN`, `EXTRA_CE`, `START`, ...) pass through to the job. Useful commands:
`squeue -u $USER`, `scancel <jobid>`, `tail -f work/logs/progress.txt`.

## 2. Run the LLM matchers on one MIG slice (Route A)

```bash
cd ~/business-entity-linking-engine/code/business_entity_resolution
RUN=v11 EXTRA_CE="qwen3rr_4b" sbatch scripts/slurm_experiments.sh
tail -f ../../work/logs/progress.txt
```

With several models (`EXTRA_CE="a b"`) they run one after another on one GPU, and **stage 2 + variants
are rebuilt after each model**. v11 uses only the 4B reranker: the rules allow up to 8B parameters, and
counted over the whole system (e5 bi-encoder + four e5 cross-encoders = 1.19B) Qwen2.5-7B would total 8.26B:

| Step | Model | Estimated full-B200 time | Output |
|---|---|---|---|
| 1 | `qwen3rr_4b`: Qwen3-Reranker-4B (4.0B, Apache-2.0), LoRA r=32, native reranker template, hard-pair band | on a 3g.90gb slice: train 1.7 h (166K pairs, 2.4 s/step), score 3.4 h (3.9M pairs), stage 2 ~40 min | `submissions/v11_1_*` |

Training speed was measured on the slice; the scoring time assumes the benchmarked ~320 pairs/s
(merged LoRA, bf16). Lower `bs` / `score_bs` in `ce_registry.py` if a model runs out of memory
(gradient checkpointing must stay on for the 4B at batch 64: without it training needs more than 89 GB). Check `work/logs/ce_train_<model>.log` and `ce_score_<model>_*.log` for real progress.

Every LLM matcher asks a yes/no question ("Do records A and B describe the same business?", or the
Qwen3-Reranker template), and its classification head starts as the LM's own `yes − no` answer
logit (`yesno` in `ce_registry.py`), so fine-tuning starts from the model's multilingual judgement
instead of a random head (France has no labels). Scoring writes its file every 1M pairs: a job
stopped by the time limit resumes where it stopped when resubmitted (trained models are skipped).

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
~/miniconda3/envs/ber/bin/python -m ber.combine --out ../../../submissions/X \
    --default p2_v11_1.parquet --default-cfg decision_v11_1.json \
    --country France --src ../reference/p2_v6_france.parquet --src-cfg ../reference/decision_v6.json --odds 0.6
```

The other options are `--em`, `--empty`, and `--src2 FILE --w2 0.5` to blend.

## 4. Full reproduction (Route B, or to rebuild)

```bash
sbatch scripts/slurm_pipeline.sh            # full run
START=ce sbatch scripts/slurm_pipeline.sh   # resume from prepare|biencoder|blocking|stage1|ce|stage2
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

It validates both and zips them with `code/business_entity_resolution/` and
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
