# Walk-forward development through 2020

Run notebooks 01, 02 and then 03 with a fresh kernel for each. They all read
`ProjectConfig()` from `src/config.py`. The full profile remains the default.

The output directory is split by notebook stage. Notebook 01 also holds shared
experiment metadata under `output/01_data_exploration/`. Reusing that metadata
folder with different settings is rejected. Changed input contents still require
a fresh `run_name`; input checksums detect those changes.

The folder layout is:

```text
data/processed/
    development_features.parquet
    development_audit.json
    development_schema.json
    stock_metadata.csv
output/
    01_data_exploration/
        experiment_config.json
        source_checksums.json
        data_audit.json             # Small dataset-provenance snapshot
        figures/
        tables/
    02_time_split_and_preprocessing/
        feature_sets.json
        feature_catalog.csv
        feature_filter_fit.json
        correlated_features_removed.csv
        folds.csv
    03_gmm_grid_search/
        models/
        figures/
        tables/
        # Search reports and candidate records when modeling is implemented
```

Each notebook writes its results directly to its own folder under `output/`.
Notebook 01 does not execute walk-forward validation or create modeling results.
Existing output folders are left intact; `walk_forward_01` is not read by the
current configuration.

Notebook 01 loads corrected Project 1 features through 2020 and saves all eligible
2017-2020 rows as Parquet, preserving RowKey and excluding Target and Prediction.
The dataset audit records filtering settings, source checksums and the dataset
checksum. Stock metadata and the dataset schema are saved beside the Parquet
file in `data/processed/`. Diagnostic figures, summary tables and the small
experiment provenance snapshot go under `output/01_data_exploration/`. No notebook writes `train_rows.pkl`.
Notebooks 02 and 03 read the same prepared Parquet dataset and reject a stale cache.
An experiment that changes only modeling settings can reuse this prepared data.

Notebook 02 uses all eligible 2017 observations to freeze feature sets A-D and
saves their catalog, filtering record and correlation removals. It defines:

| Fold | Training | Validation |
|---|---|---|
| validate_2018 | 2017 | 2018 |
| validate_2019 | 2017-2018 | 2019 |
| validate_2020 | 2017-2019 | 2020 |

Call `chronological_folds(development_rows, cfg)` with the complete development
dataset. Create a fresh imputer/scaler inside each fold; fit it only on training
rows and transform validation rows. Full mode uses every eligible GMM row.
Quick mode may cap rows within an already defined fold, but never changes the
prepared dataset or the 2017 feature-filtering fit.

The current notebook 03 loads prepared data and frozen feature lists; its GMM search is still to be implemented.
Tune GMM parameters separately within each fixed feature set. Inspect yearly
validation scores, convergence and state stability. Overlapping rolling windows
make folds dependent. Raw likelihoods and BIC from different feature spaces
cannot directly rank feature sets. Do not use future returns as clustering labels.

After selecting the configuration, refit a fresh preprocessing/GMM pipeline on
all development rows through 2020. Save the selected pipeline as
`output/03_gmm_grid_search/models/<selected_model_id>.joblib` and record:

```python
# In notebook 03: OUT = ROOT / "output" / "03_gmm_grid_search"
write_json(OUT / "final_model_fit.json", {
    "selected_model_id": selected_model_id,
    "feature_set": selected_feature_set,
    "features": feature_sets[selected_feature_set],
    "training_start": cfg.development_start,
    "training_end": cfg.development_end,
    "n_rows": len(development_rows),
})
```

These dates describe the configured fit interval. Record actual observed dates
separately if needed. The final refit can permute numeric cluster IDs; interpret
its own state profiles rather than reusing fold IDs.

`finalize_model_selection()` checks the refit record and locks the model, dataset,
feature lists, configuration, code, source inputs, folds and search reports. It
takes `RUN` (`output/01_data_exploration/`, where shared metadata is stored) as
its `out` argument. It reads model records from the sibling
`output/03_gmm_grid_search/` folder and feature-selection files from
`output/02_time_split_and_preprocessing/`. Ward/stability reports are checksummed when present. It no longer
requires a separate model-selection checkpoint treating 2020 as an outer holdout.

Only after locking may the final evaluation call:

```python
daily_rows, test_audit = load_daily_until(
    ROOT, cfg, cfg.test_end, allow_holdout=True, out=RUN,
)
test_rows = daily_rows.loc[daily_rows.Date.between(cfg.test_start, cfg.test_end)]
```

The loader verifies the lock before reading 2021. Apply the saved pipeline to
`test_rows`; do not refit preprocessing, tune settings or select features on it.
The loader includes earlier history so eligibility and trading-day indices remain
consistent. No 2021 data is read by notebooks 01-02 or the protocol tests.

Document that 2021 was previously inspected in Project 1; the split does not undo
that exposure. The current changes perform no new final-test evaluation.

Verification uses actual JPX records, never generated market data:

```bash
python -m unittest discover -s tests -v
```
