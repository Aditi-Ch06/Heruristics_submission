# Business Entity Resolution — Approach Report
**Amazon ML Challenge 2026** · Metric: macro-averaged F0.5

## Problem
Given a primary catalog (Source 1), identify every record in two noisy external catalogs (Source 2, Source 3) that refers to the **same real-world business**. Names and addresses contain typos, abbreviations, reordering, legal-suffix noise, and cross-lingual variants across multiple countries. The metric is **macro-F0.5**, which weights precision twice as heavily as recall, so false matches are penalised harder than misses.

## Pipeline Overview
**Clean → Block (dual-channel) → Feature engineering → Two-stage ranking → Expected-F0.5 decision → 1-to-1 assignment.**

1. **Normalisation:** Unicode/accent folding, lowercase, legal-suffix and "DBA" stripping, address-abbreviation expansion (`st` → `street`, `rd` → `road`, …), and derivation of compact keys, house numbers, postal codes, and a Soundex phonetic code per record.

2. **Dual-channel blocking (recall unlock):** An inverted-index blocker generates candidates per country. Beyond standard name keys (compact string, token bigrams, sorted tokens, phonetic, postal + first-token), we add an **address-only channel** (compact-address prefix, address bigrams, long address tokens). This recovers true matches whose *name* is corrupted but whose *address* is shared — raising pair-recall from **≈87% to 96.9%** at ~90 candidates/query, setting the achievable ceiling for everything downstream.

3. **Features (49 base):** Per candidate pair: RapidFuzz string similarities (ratio, token-sort/set, partial, Jaro-Winkler, Levenshtein, Indel) on name and address; token/char Jaccard; TF-IDF cosine (word 1–2 grams); postal/house-number/number-set overlap; phonetic and compact-equality flags; length and digit-ratio gaps; and per-query competition features (gap-to-best, count-of-close-rivals).

4. **Two-stage ranking:** A first LightGBM + XGBoost ensemble (5-fold GroupKFold, grouped by Source-1 ID to prevent leakage) produces out-of-fold match probabilities. From these, we build **sibling features**: once a business has a confident match, other candidates are re-scored by their similarity to that *already-confirmed sibling*. Because all copies of one business share the same corruptions, this recovers hard matches that direct comparison misses (sibling signal fired on **89%** of query groups). A second ensemble trains on the 49 base features + sibling + competition features.

5. **Decision & assignment:** For each Source-1 business, we select the match set that **maximises expected F0.5** (rather than a fixed threshold), then enforce **one-to-one** assignment within each country by greedy highest-confidence selection.

## Results (honest offline validation)
Validated on out-of-fold predictions with a large negative pool (~1.5M records/source) to mirror the true class imbalance.

| Stage | Honest macro-F0.5 |
|---|---|
| Single-stage baseline | 0.918 |
| **Two-stage + sibling features** | **0.954** |

* Blocking recall: **96.9%**
* Candidate positive rate: 3.7%
* Base features: 49 (+2 sibling/competition)

## Compliance & Reproducibility
Fully **offline**; no external data or pretrained weights. Open-source libraries only (LightGBM, XGBoost, scikit-learn, RapidFuzz — MIT/Apache). All randomness fixed (`seed=42`); GroupKFold prevents train/test leakage across sources. The pipeline is a single reproducible script that regenerates `matching_results.tsv` and `candidate_pairs.tsv` end-to-end, with automatic GPU→CPU fallback.

## Known Limitations
Records from countries unseen or under-represented in training (e.g. cross-lingual variants) are the main residual error source and the primary avenue for further gains.