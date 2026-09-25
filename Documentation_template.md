# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve each Source-1 business against ~10M noisy Source-2/3 records with a three-part pipeline: (1) **GPU dense blocking** with a multilingual-e5 bi-encoder fine-tuned on the training links with mined hard negatives (≈40 candidates per S1, 99.84% pair recall); (2) a **two-stage matcher** — an XGBoost model over ~70 country-agnostic string/number/legal-form features, stacked with a fine-tuned cross-encoder and *competition features* that exploit the fact that every S2/S3 record belongs to at most one S1; (3) a **per-entity decision rule** tuned directly on out-of-fold macro F0.5 (exclusive assignment + expected-F0.5 subset selection, which handles singletons explicitly).

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M-entity training set (7.64M true links) showed:

- **Scale**: 2.2M S1 × 10.3M S2/S3 (train), 1.73M × 9.97M (test) → blocking must be sub-quadratic.
- **Cardinality**: 0–11 links per S1 (median 3; ≤5 from S2, ≤6 from S3); **5.59% singletons**. Every S2/S3 record links to **at most one** S1, and **no link crosses countries**. 26.6% of S2/S3 records are unmatched distractors.
- **Distractors are adversarial near-duplicates** of S1 entities: same name with a different house number (`Premier Financial Services Corp, 8326 107` vs S1 `… Inc, 8315 107`), same name root with a different legal form (`UDE Service EI` vs `UDE Service SAS`), chain names repeated in many cities.
- **Name noise** (true matches): legal-suffix variants/duplication (`Inc Inc`, `Private Limited Limited`), word shuffles, bracket/punctuation junk (`[Corp]`, `-- `, `>> `), accents (`Nétwork`), OCR swaps (`Heart1and`, `F0os`), filler words (`Services`, `Center`, `Partners`), **domain-style names** (`ryfoods.com`), **DBA** forms (`X doing business as Y`), appended phone numbers, `M/s` prefixes, and **non-Latin scripts** (Devanagari, Bengali, Kannada…).
- **Address noise**: component reordering, abbreviations, full vs. abbreviated state names (also in local script: `दिल्ली`), zero-padded (`02610`) or truncated (`1914→191`, `3890→890`) house numbers, extra inserted numbers (`Hn 359 B-8/15`), missing address (~3% of S2/S3).
- **France** (15% of test) is unseen in training but follows the same generator (legal forms SARL/SAS/EURL/EI, `R.`=rue, `N°16`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage stacked GBDT + cross-encoder) + metric-aware decision rule
**Core Innovation:** hard-negative bi-encoder blocking; "competition" stacking features that model the 1-S2/S3-to-≤1-S1 structure globally; decision thresholds optimised directly on out-of-fold per-entity F0.5.

Training S1 entities are split into 10 random folds: folds 0–2 (**E**) train the neural encoders, folds 3–9 (**R**) train/validate the GBDT, so neural scores used as GBDT features are always out-of-sample. Country is used **only as a partition key** for retrieval (never as a feature), so any new country label (France) flows through unchanged.

---

## 3. Candidate Generation (Blocking)

- **Normalisation** (`normalize.py`): Unicode fold + `anyascii` transliteration, `&`→and, merge spelled initials (`L.L.C.`→`llc`), canonical abbreviation maps (EN/IN/FR: street/rd/ave/blvd/rue/chemin…, US & Indian state names), legal-form canonicalisation, DBA split, domain stem, phone removal, and a **consonant-skeleton key** that bridges transliteration variants (`श्री गणेश ट्रेडर्स` and `Shree Ganesh Traders` → `sr gns trdrs`).
- **Dense bi-encoder** (`biencoder.py`): `intfloat/multilingual-e5-small` (MIT, 118M params) fine-tuned with symmetric InfoNCE on 2.29M E-fold positive pairs, then a second epoch with **mined hard negatives** (top-scoring non-matches of the same S1). Exact inner-product kNN on GPU inside each country partition.
- **Blocking keys used:** forward S1→target top-40 and reverse target→S1 top-5 by cosine; a pair is kept if it is within the S1's top-20 per source **or** the target's best S1.
- **Candidate pairs generated:** ≈40 per S1 (89.1M train; test figure in Appendix).
- **How we ensured true matches were not lost:** recall-vs-cap curves on all 7.64M training links (hard-negative training raised recall@3/source from 90% to 98.97%); pair recall at the chosen cap = **99.84%**. A CPU IDF-token path was evaluated and dropped (+0.03% recall at ~25× the cost). Remaining misses are mostly targets with empty address and a partial or totally different name.

---

## 4. Matching Model

**Features used** (`features.py`, all country-agnostic):
- **Name**: ratio / partial / token-sort / token-set / Jaro-Winkler / Levenshtein on core name; full-name and raw-lowercase ratios; skeleton-key ratios & exact flag; no-space ratios (domain names); DBA-alternative similarity; token counts/common tokens; first-token match; word-IDF and char-3-gram TF-IDF cosines (fitted per country on the split's own records, unsupervised).
- **Address**: ratio / token-set / token-sort / partial / Jaro-Winkler; word-IDF and char-3-gram cosines; empty-address flag; **number logic** tolerant to the observed noise (zero padding, dropped leading/trailing digits, inserted numbers): house-number exact/prefix/suffix/any-match, "big number missing" (distractor signal), number-set Jaccard, zip exact / 3-digit prefix.
- **Other**: legal-form agreement / conflict, domain flags, name frequency within source & country (chains), bi-encoder cosine and ranks (forward/reverse), retrieval-group context (gap to best, rank within S1, number of S1 claimants of the target).

**Model type:**
- **Stage 1**: XGBoost (Apache-2.0, GPU hist, depth 9, η 0.08, early stopping) on the full feature set, 4-fold S1-grouped CV over a 500K-entity R subset.
- **Cross-encoder**: `multilingual-e5-small` fine-tuned as a pair classifier on E-fold candidates (positives + top-12 hard negatives, 3.6M pairs); scores pairs with p1 ≥ 0.003.
- **Stage 2**: XGBoost on p1 + cross-encoder score + compact pair features + **competition context** computed over *all* candidates: p1 rank/gap within the S1's list, margin to the best other claimant of the same target, number of claimants, cross-source agreement (best p1 in the other source), count of confident matches per S1.

**Threshold selection method:** grid search on out-of-fold stage-2 probabilities over the full evaluation universe (including singletons and entities whose links were lost in blocking) of (a) plain thresholds and (b) an expected-F0.5 rule — per S1 choose the top-k prefix maximising `1.25·Σp / (k + 0.25·Σp)` versus the score of predicting nothing (`Π(1−p)`) — each with/without exclusive assignment of every target to its best S1.

---

## 5. Results & Error Analysis

- **Blocking recall (train, all links):** 99.84% at ≈40 candidates/S1.
- **F_0.5 Score (macro, out-of-fold, 500K held-out S1):** stage 1 = [TBD], stage 2 = [TBD]
- **Public leaderboard:** [TBD]
- **Common false positives (wrong merges):** [TBD]
- **Common false negatives (missed matches):** [TBD]

---

## 6. Conclusion

[TBD]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `run_all.sh` reproduces everything from `dataset/`:

| Step | Module | Output |
|---|---|---|
| normalise | `ber.prepare` | `work/<split>/source*.parquet` |
| bi-encoder | `ber.train_biencoder` (+ `--hard`) | `work/biencoder_v2` |
| blocking | `ber.candidates` (+ `--prune --cap 20`) | `work/<split>/candidates.parquet` |
| stage-1 features/model | `ber.build_features subset`, `ber.train_ranker stage1` | `work/models/stage1_*` |
| p1 pass | `ber.build_features score --split <split>` | `work/<split>/p1.parquet` |
| cross-encoder | `ber.train_ce train/score` | `work/crossencoder`, `ce.parquet` |
| stage-2 + decision | `ber.train_ranker stage2` | `work/models/stage2_*`, `decision.json` |
| inference | `ber.infer` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

### B. License & Parameter Verification

| Component | License | Parameters |
|---|---|---|
| intfloat/multilingual-e5-small (bi-encoder & cross-encoder) | MIT | 118M each |
| XGBoost | Apache-2.0 | trees only |
| rapidfuzz, anyascii, polars, scikit-learn, sparse_dot_topn | MIT / ISC / MIT / BSD / Apache-2.0 | – |

No external data, APIs, geocoders or lookups are used; all models are trained only on the provided training files.
