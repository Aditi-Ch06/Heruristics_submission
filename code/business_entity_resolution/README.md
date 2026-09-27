# Business Entity Resolution — Execution & Reproduction Guide
**Amazon ML Challenge 2026**

This repository contains the end-to-end pipeline for matching business entities across noisy catalogs (Source 1 to Source 2 & Source 3) under the **macro-F0.5** evaluation metric.

---

## Directory Structure
```text
.
├── Documentation_template.md
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        └── src/
            └── kaggle_winning_final.py