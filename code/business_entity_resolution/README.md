# Business Entity Resolution

This project matches each business in **Source 1** with its corresponding record or records in **Source 2** and **Source 3**. The implementation is a Kaggle notebook that builds candidate pairs, computes name and address similarity features, trains matching models, and writes the final submission files.

## How It Works

1. **Prepare the data.** Download and extract the organizer-provided dataset, then load the training and test tables.
2. **Generate candidates.** Use country-based blocking and business-name/address signals to find a manageable set of possible matches.
3. **Build features and train.** Calculate similarity features and train five-fold LightGBM and XGBoost models using grouped cross-validation.
4. **Predict and write results.** Blend the model predictions, apply a 0.70 final match threshold, and write the matches and candidate lists as TSV files.

The notebook also reports validation metrics at several thresholds. During test inference, candidates with scores of at least 0.12 are considered for assignment; the final matching output uses the 0.70 threshold. Candidate lists contain the pairs produced by blocking, including pairs below either score threshold.

## Submission Structure

```text
Heuristics_submission.zip
|-- output/
|   |-- matching_results.tsv
|   `-- candidate_pairs.tsv
|-- code/
|   `-- business_entity_resolution/
|       |-- src/
|       |   `-- amazon-ml-challenge-team-heuristics.ipynb
|       |-- README.md
|       `-- requirements.txt
`-- Documentation_template.md
```

The notebook is the only source-code implementation included in this submission. The training and test datasets are not included in the ZIP; the notebook downloads the organizer-provided dataset when it runs.

## Reproduction

### 1. Open the notebook

Extract the submission ZIP and upload this notebook to Kaggle:

```text
code/business_entity_resolution/src/amazon-ml-challenge-team-heuristics.ipynb
```

### 2. Configure the Kaggle session

Enable Internet access and select a Kaggle accelerator with GPU support. The training and prediction code configures LightGBM for GPU and XGBoost for CUDA.

The notebook downloads the dataset and extracts it under `/kaggle/working/`. It expects the data files at these paths:

```text
/kaggle/working/student_resource/dataset/train/
/kaggle/working/student_resource/dataset/test/
```

### 3. Install dependencies

Before running the notebook, execute this command in a new first cell:

```python
%pip install gdown==5.2.0 lightgbm==4.6.0 numpy==2.2.6 pandas==2.2.3 rapidfuzz==3.13.0 scikit-learn==1.6.1 Unidecode==1.3.8 xgboost==3.0.2
```

The pinned versions are also listed in `requirements.txt`.

### 4. Run the notebook

Run all notebook cells in order, from top to bottom. The notebook samples up to 50,000 Source 1 training records, builds training candidates, fits the five-fold model ensembles, evaluates several validation thresholds, and runs inference on the test records.

### 5. Collect the output files

When execution completes, retrieve these files from `/kaggle/working/output/` and place them in the `output/` directory of the submission ZIP:

```text
matching_results.tsv
candidate_pairs.tsv
```

## Output Files

Both files are tab-separated and include a header row.

### `matching_results.tsv`

Contains the final predicted matches for every Source 1 test entity.

| Column | Description |
| --- | --- |
| `source1_entity_id` | Source 1 test entity identifier. |
| `matched_entity_ids` | Comma-separated matching IDs from Source 2 and/or Source 3; empty when no match is predicted. |

### `candidate_pairs.tsv`

Contains the candidates produced by the blocking stage for each Source 1 test entity.

| Column | Description |
| --- | --- |
| `source1_entity_id` | Source 1 test entity identifier. |
| `candidate_entity_ids` | Comma-separated candidate IDs from Source 2 and/or Source 3; empty when blocking finds no candidates. |

Every final match in `matching_results.tsv` should also appear in the corresponding candidate list in `candidate_pairs.tsv`.

## Validation

Before submission, validate the generated files with the challenge-provided validation script. For example, from the challenge workspace root:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Adjust the paths if the validator or test data are located elsewhere. The script checks the output format and consistency with the test data.

## Compute Notes

The notebook uses up to 50,000 Source 1 training records, samples up to 1.5 million negative records per target source, and evaluates the test set during inference. Candidate generation, feature computation, and model training can require substantial memory, GPU support, and runtime. Execution is designed for Kaggle and uses Kaggle filesystem paths and the dataset download link configured in the notebook.
