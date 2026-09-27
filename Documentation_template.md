# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve every Source-1 (S1) business against ~10M noisy Source-2/3 records with a cascade that narrows the
search space at each step and spends the expensive models only where they are needed:

1. **Hybrid blocking** (sub-quadratic, GPU): a fine-tuned multilingual-e5 bi-encoder (dense kNN, forward and
   reverse) united with a GPU IDF-weighted typed-token index. **46 candidates per S1, 99.86% pair recall.**
2. **Learned candidate filter**: a stage-1 XGBoost over ~100 string / number / address / retrieval features
   keeps pairs with p1 ≥ 0.01. **The final candidate set has 4.71 records per S1 on test** (6.4 at p1 ≥ 0.001)
   with a **99.60% recall ceiling** on held-out training entities; this is exactly the set the matching model
   scores and the content of `candidate_pairs.tsv`.
3. **Matching model** on the final candidates: four fine-tuned multilingual cross-encoders (e5-small / base /
   large) and a **Qwen3-Reranker-4B LLM matcher** (LoRA) feed a stage-2 XGBoost + LightGBM ensemble with
   "competition" features that model the fact that each S2/S3 record belongs to at most one S1.
4. **Metric-aware decision**: exclusive assignment of each target to its best S1, then per-S1 top-k selection
   that maximises expected F0.5 against the score of predicting nothing (singletons).

Out-of-fold macro F0.5 on held-out training entities (US/India): **0.99077** (v8 pipeline). Every model is MIT
or Apache-2.0; **all models together have 5.2B parameters**, under the 8B limit even when counted jointly.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M-entity training set (7.64M true links) showed:

- **Scale**: 2.2M S1 × 10.3M S2/S3 (train), 1.73M × 9.97M (test, 1.7 × 10¹³ possible pairs) → blocking must be
  sub-quadratic.
- **Cardinality**: 0–11 links per S1 (mean 3.46; ≤5 from S2, ≤6 from S3); **5.59% singletons**. Every S2/S3
  record links to **at most one** S1, and **no link crosses countries**. 26.6% of train S2/S3 records are unmatched.
- **Distractors are adversarial near-duplicates** of S1 entities: same name with a different house number,
  same name with a different legal form (`UDE Service EI` vs `UDE Service SAS`), distractor words appended
  (`Ent`, `Traders`; in France `International`, `Distribution`, `Participations`, `Holding`), random names at
  the same address. Distractors almost never keep the exact address (0.05% vs 22% for true copies).
- **Name noise** on true copies: legal-suffix variants and repeats, word shuffles, junk prefixes (`--`, `>>`,
  `M/s`), accents, OCR swaps (`Heart1and`), **filler words** (`Services`, `Center`; in France `& Fils`,
  `Associés`), domain-style names, DBA forms, appended phone numbers or `(ID: 30420)`, non-Latin scripts.
- **Address noise**: reordering, abbreviations, state name vs code, zero-padded or truncated house numbers,
  missing address (~3%); in France `N° 13`, `Crs`, `Q.`, `28 B`, and département instead of région.
- **France** (15% of test S1) has no training labels, its own vocabulary (names = brand + category word + legal
  form) and 3× more look-alike candidates per S1 than the US.

### 2.2 Solution Strategy

**Approach Type:** Hybrid blocking → learned candidate filter → stacked matcher (GBDT + cross-encoders + LLM) →
metric-aware decision rule.
**Core Innovation:** a blocking cascade that reaches 4.7 candidates per S1 at a 99.6% recall ceiling; an LLM
matcher whose head starts from the model's own yes/no answer; "competition" features and exclusive assignment
that use the one-target-to-at-most-one-S1 structure; decisions optimised directly on out-of-fold per-entity F0.5.

Training S1 entities are split into 10 random folds: folds 0–2 (**E**) train the neural models (bi-encoder,
cross-encoders, LLM matcher), folds 3–9 (**R**) train and validate the GBDTs with 4-fold S1-grouped
cross-validation, so every neural score used as a GBDT feature is out-of-sample. Country is used **only as a
partition key** (never as a feature), so an unseen country label such as France flows through unchanged.

---

## 3. Candidate Generation (Blocking)

The candidate set is built in two steps: a recall-oriented hybrid retrieval, then a learned filter.

**Normalisation** (`normalize.py`, `admin_areas.py`, `fillers.py`): Unicode fold + `anyascii` transliteration,
`&`→and, merged initials (`L.L.C.`→`llc`), canonical EN/IN/FR abbreviation and legal-form maps, US/Indian state
codes and French région/département → one région code, DBA split, domain stem, phone/ID removal, a
**consonant-skeleton key** that bridges transliteration and OCR variants (`श्री गणेश ट्रेडर्स` / `Shree Ganesh
Traders` → `sr gns trdrs`), a French address locale (`N°`, `Crs`→cours, `B`→bis), and **data-driven filler
words** (tokens far more frequent in S2/S3 names than in S1 names, per split and country, no labels).

**Step 1 — hybrid retrieval** (`candidates_v5.py`, per target source, inside each country):
- **Dense**: `intfloat/multilingual-e5-small` bi-encoder (MIT, 118M) fine-tuned with symmetric InfoNCE on E-fold
  links, then a second round with mined hard negatives (top-scoring non-matches of the same S1). Exact
  inner-product kNN on GPU: S1 → target top-40 and target → S1 top-5 (reverse).
- **Sparse**: a GPU IDF-weighted index over typed tokens (filler-free name, skeleton, address words, digits,
  `house number × street word` keys), fitted on each country's own records. It recovers the tail the encoder
  misses in an unseen language (French true matches ranked ≥10 were 23× more frequent than in the US).
- **Kept**: dense rank < 20 **or** sparse rank < 5 **or** the target's best S1 (reverse rank 0).
- **Result**: 45.4 candidates per S1 at **99.862%** pair recall on all 7.64M training links; 79.6M test pairs
  (45.97 per S1). Every pair with identical filler-free name and address is present, in every country.

**Step 2 — learned filter** (`build_features.py`, `ranker.py`): stage-1 XGBoost (GPU, depth 9, 4-fold OOF,
trained on 800K R-fold entities) scores all retrieved pairs with ~100 features (Section 4) and keeps
**p1 ≥ 0.01** (`config.CAND_MIN_P1`). Everything downstream — cross-encoders, the LLM matcher, stage 2, the
decision — runs on exactly this set, which is written to `candidate_pairs.tsv`.

| Candidate set | Test candidates / S1 (US / India / France) | Train pair recall ceiling |
|---|---|---|
| Hybrid retrieval | 45.97 | 99.862% |
| Filter p1 ≥ 0.001 (previous package) | 6.35 | 99.822% |
| **Filter p1 ≥ 0.01 (final)** | **4.71** (4.62 / 4.62 / 5.22) | **99.601%** |
| Filter p1 ≥ 0.02 | 4.33 | 99.402% |

- **Why 0.01:** the v8 matcher accepted only 440 of its 5.9M test matches below p1 = 0.01 (0.007%), so the
  26% smaller candidate set costs essentially no F0.5. The average S1 has 3.46 true links, so 4.71 is within
  1.3 candidates of the minimum possible size.
- **Reduction ratio (test):** 8.16M final candidate pairs out of 1.73 × 10¹³ S1 × target pairs = 0.99999953.
- **Remaining misses** are mostly targets with an empty address and a partial or different name.
- **Scaling:** retrieval is exact GPU kNN plus sparse matrix top-k, linear in the number of records per country
  block; the filter is a streaming GBDT pass over 5M-row chunks.

---

## 4. Matching Model

**Features** (`features.py`, `adapt.py`, all country-agnostic):
- **Name**: ratio / partial / token-sort / token-set / Jaro-Winkler / Levenshtein on core, full, raw, skeleton,
  no-space, DBA-alternative and filler-free views; exact flags; missing/extra identity tokens with typo
  tolerance; word-IDF and char-3-gram TF-IDF cosines fitted per country without labels.
- **Address**: string similarities and TF-IDF cosines; empty flag; house-number logic tolerant to the observed
  noise (exact / prefix / suffix / any, "big number missing"), number-set Jaccard, postcode agreement.
- **Word difference**: within-country document frequency of the missing and extra words, one-word-swap
  Jaro-Winkler, exact same-address flag (separates filler words from distractor words).
- **Other**: legal-form agreement / conflict, domain flags, name frequency (chains), dense / sparse scores and
  ranks, retrieval-group context.

**Models:**
- **Stage 1** (candidate filter, Section 3): XGBoost on all features → p1.
- **Cross-encoders** (`crossencoder.py`, `ce_registry.py`), trained on E-fold candidates with hard negatives:
  e5-small on raw text, e5-base on raw text, e5-small on a canonical view (filler-free name + legal form |
  normalised address), and e5-large on a field-structured view (raw ‖ canonical) with side-swap augmentation.
- **LLM matcher**: **Qwen3-Reranker-4B** (Apache-2.0, 4.0B) with LoRA r = 32 (peft), trained on 166K E-fold
  pairs of the *hard band* 0.02 ≤ p1 < 0.995 — the pairs stage 1 is unsure about. The pair is given in the
  reranker's own template (query = one record, document = the other), and the classification head is
  initialised with the LM-head difference of the answer tokens `yes` − `no` (verified equal to the full LM
  head), so fine-tuning starts from the model's multilingual judgement instead of a random direction — the
  part that can carry over to France, which has no labels. Zero-shot AUC on held-out hard-band pairs is 0.68
  before fine-tuning.
- **Stage 2**: XGBoost + LightGBM average on p1, all cross-encoder and LLM scores with their per-S1 and
  per-target context (max, gap, rank, count), compact pair features, word-difference features, and
  **competition features from p1**: rank and margin within the S1's list, margin to the best *other*
  claimant of the same target, claimant count, best p1 in the other source, confident matches per S1.

**Threshold selection method:** grid search on out-of-fold stage-2 probabilities over the full evaluation
universe (singletons and entities whose links were lost in blocking included): plain thresholds, and an
expected-F0.5 rule — per S1 the top-k prefix maximising `1.25·Σp / (k + 0.25·Σp)` against the expected score of
predicting nothing (`Π(1−p)`), with a floor — each with and without exclusive assignment of every target to its
best S1. Selected: exclusive assignment + expected-F0.5 (floor 0.5, empty bias 1.3). An exact Poisson-binomial
optimiser was no better (0.99020 vs 0.99023), so the decision layer is saturated.

**France (unseen country):** countries are separable (blocking, exclusivity and decisions never cross them), so
each country can be decided from its own run. French-specific handling is label-free: région/département
canonicalisation, French address locale, excess filler words, and a same-address test that measures the role
of each word on test (validated on train, where a word's same-address rate tracks its match rate with
correlation 0.85). An EM prior re-estimation (Saerens et al., 2002) gives an odds multiplier of ~0.8 for France,
used for leaderboard-free calibration variants.

---

## 5. Results & Error Analysis

- **Blocking:** 99.862% recall at 45.4 candidates / S1 (retrieval); final candidate set 4.71 / S1 on test at a
  99.60% recall ceiling.
- **F_0.5 (macro, out-of-fold, 800K held-out US/India S1):** stage 1 alone 0.98496; stage 2 (v8: four
  cross-encoders + word-difference features) **0.99077**; with the Qwen3-Reranker-4B matcher (v11): [TBD].
- **Public leaderboard:** v5 0.984567 → v6 (French fillers) 0.985093 → v8 (e5-large cross-encoder)
  **0.985164**; v11: [TBD].
- **Where US/India still lose:** 81% of the remaining loss is missed matches; the largest block is
  empty-address targets whose exact name is shared by 3 or more S1 records, which name and address cannot
  resolve.
- **France:** the leaderboard implies French F0.5 ≈ 0.953 against a model estimate of 0.975 — the unseen
  vocabulary gives word roles that differ from training (e.g. `Groupe` is partly a filler in France while
  `Group` is a pure distractor word in the US and India).

---

## 6. Conclusion

A two-step blocking cascade — hybrid dense + sparse retrieval followed by a learned GBDT filter — keeps 4.7
candidates per S1 (from 1.73 × 10¹³ possible pairs) while retaining a 99.6% recall ceiling, and the matching
models only ever see that set. Most of the remaining error on the training countries is irreducible name
ambiguity; the open problem is the unseen country, where label-free tests of word roles and an LLM matcher that
starts from its own multilingual judgement are the levers we pursued.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `scripts/run_pipeline.sh` reproduces everything from `dataset/`
(`scripts/slurm_pipeline.sh` / `slurm_experiments.sh` run it under SLURM); see `run.md`.

| Step | Module | Output |
|---|---|---|
| normalise, fillers | `ber.prepare` | `work/<split>/source*.parquet` |
| bi-encoder (2 rounds) | `ber.train_biencoder`, `ber.mine_hardneg` | `work/biencoder_v2` |
| hybrid retrieval | `ber.candidates_v5` (+ `--prune 20 5`) | `work/<split>/candidates.parquet` |
| stage-1 filter | `ber.build_features subset`, `ber.train_ranker stage1`, `ber.build_features score` | `work/<split>/p1.parquet` |
| cross-encoders, LLM matcher | `ber.train_ce train / score` | `work/crossencoder_*`, `work/<split>/ce_*.parquet` |
| stage 2 + decision | `ber.train_ranker stage2` | `work/models/stage2_*`, `decision.json` |
| inference | `ber.infer` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| per-country variants | `ber.combine` | `submissions/<run>_*` |
| final candidate file | `ber.package_candidates` | `output/candidate_pairs.tsv` |

### B. License & Parameter Verification

| Component | License | Parameters |
|---|---|---|
| intfloat/multilingual-e5-small (bi-encoder; raw and canonical cross-encoders) | MIT | 118M each, 3 models |
| intfloat/multilingual-e5-base (cross-encoder) | MIT | 278M |
| intfloat/multilingual-e5-large (cross-encoder) | MIT | 560M |
| Qwen/Qwen3-Reranker-4B (LLM matcher, LoRA merged for inference) | Apache-2.0 | 4.02B |
| **All neural models together** | | **5.21B** |
| XGBoost, LightGBM | Apache-2.0, MIT | trees only |
| peft, transformers, torch | Apache-2.0, Apache-2.0, BSD-3 | – |
| rapidfuzz, anyascii, polars, scikit-learn, sparse_dot_topn | MIT / ISC / MIT / BSD-3 / Apache-2.0 | – |

No external data, APIs, geocoders or lookups are used; models are trained only on the provided training files,
and test records are used only without labels (TF-IDF fitting, filler statistics, label-shift estimation).
