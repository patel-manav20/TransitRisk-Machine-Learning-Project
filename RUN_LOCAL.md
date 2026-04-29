# Run TransitRisk Locally

This project can run locally without retraining any models, as long as the saved model artifacts and processed data files are available.

## Verified local setup

Tested on macOS with `Python 3.12.2`.

The dashboard entry point is:

```bash
streamlit run app/dashboard.py
```

## Minimum run path

You do not need to run the notebooks or `make all`.

The dashboard only needs:

- `data/processed/modeling_table.parquet`
- `data/processed/X_features.parquet`
- `data/processed/train_val_test_indices.json`
- `models/xgb_calibrated.joblib`

Optional, but used when present:

- `data/processed/prediction_sets.parquet`
- `data/processed/thresholds.json`
- `figures/metrics.json`

If `models/xgb_calibrated.joblib` is missing, the app will also accept `models/xgb.joblib`.

## Setup commands

From the project root:

```bash
cd "/Users/manavnayanbhaipatel/Desktop/SJSU/DATA - 245/ML- Project/transitrisk-main"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Artifact placement

The app now checks for saved artifacts in this order:

1. The repo itself, such as `transitrisk-main/data/` and `transitrisk-main/models/`
2. A custom artifact directory pointed to by `TRANSITRISK_ARTIFACTS_DIR`
3. A sibling bundle folder named `transitrisk_data_models/`

That means any of these layouts work:

### Option A: Put artifacts directly in the repo

```text
transitrisk-main/
  data/
    processed/
      modeling_table.parquet
      X_features.parquet
      train_val_test_indices.json
      ...
  models/
    xgb_calibrated.joblib
```

### Option B: Keep artifacts in the sibling bundle

```text
ML- Project/
  transitrisk-main/
  transitrisk_data_models/
    data/
      processed/
        modeling_table.parquet
        X_features.parquet
        train_val_test_indices.json
        ...
    models/
      xgb_calibrated.joblib
```

### Option C: Point to another extracted artifact folder

```bash
export TRANSITRISK_ARTIFACTS_DIR="/absolute/path/to/artifact-bundle"
```

That directory should contain `data/`, `models/`, and optionally `figures/`.

## Launch the dashboard

```bash
cd "/Users/manavnayanbhaipatel/Desktop/SJSU/DATA - 245/ML- Project/transitrisk-main"
source .venv/bin/activate
streamlit run app/dashboard.py
```

For a headless smoke test:

```bash
streamlit run app/dashboard.py --server.headless true
```

## Important note about retraining

Do not run:

```bash
make all
```

That executes the full notebook pipeline. It is only for regenerating data and retraining models from scratch.

For local dashboard use, only install dependencies and launch Streamlit with the saved artifacts already in place.

## Common errors and fixes

### "Saved artifacts not found"

Cause: the required `.parquet` or `.joblib` files are not present in any of the checked locations.

Fix:

- place the saved `data/processed/` and `models/` files in the repo, or
- place them in the sibling `transitrisk_data_models/` folder, or
- set `TRANSITRISK_ARTIFACTS_DIR` to the extracted artifact bundle location

### `ModuleNotFoundError` or package import errors

Cause: dependencies were not installed into the active environment.

Fix:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Streamlit command not found

Cause: the virtual environment is not active.

Fix:

```bash
source .venv/bin/activate
streamlit run app/dashboard.py
```

### Dashboard opens but shows no data

Cause: Streamlit started correctly, but the processed artifacts are still missing.

Fix: confirm these exact files exist:

```bash
ls data/processed/modeling_table.parquet
ls data/processed/X_features.parquet
ls data/processed/train_val_test_indices.json
ls models/xgb_calibrated.joblib
```

If you keep artifacts outside the repo, check the sibling `transitrisk_data_models/` folder or the `TRANSITRISK_ARTIFACTS_DIR` environment variable instead.
