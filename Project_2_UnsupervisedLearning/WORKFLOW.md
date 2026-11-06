# Walk-forward development through 2020

Run notebooks 01 and 02 in order with a fresh kernel. Both read `ProjectConfig()`
and default to the full profile and `output/walk_forward_01/`. Existing experiments
are preserved. A changed configuration or input source requires a new run name.

Notebook 01 loads corrected Project 1 features through 2020 and writes:

- `data/processed/development_features.parquet`: all eligible 2017-2020 rows,
  including the named RowKey index; no Target or Prediction.
- `data/processed/development_audit.json`: filtering settings, source checksums
  and the prepared dataset checksum. Notebook 02 rejects a stale cache.
- Experiment-specific figures, tables, metadata, schema and audit under output.

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

The existing notebooks prepare the experiment; they do not run a GMM search.
Tune GMM parameters separately within each fixed feature set. Inspect yearly
validation scores, convergence and state stability. Overlapping rolling windows
make folds dependent. Raw likelihoods and BIC from different feature spaces
cannot directly rank feature sets. Do not use future returns as clustering labels.

After selecting the configuration, refit a fresh preprocessing/GMM pipeline on
all development rows through 2020. Save the selected pipeline as
`output/<run_name>/models/<selected_model_id>.joblib` and record:

```python
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
requires `gmm_cv.csv`, `gmm_candidates.csv` and `candidate_ids.json` from the later
modeling stage. Ward/stability reports are checksummed when present. It no longer
requires a separate model-selection checkpoint treating 2020 as an outer holdout.

Only after locking may the final evaluation call:

```python
daily_rows, test_audit = load_daily_until(
    ROOT, cfg, cfg.test_end, allow_holdout=True, out=OUT,
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
