# Price-adjustment correction: 2026-11-06

The Kaggle CSV records AdjustmentFactor on the last pre-adjustment observation.
The old calculation used it on that observation. The corrected calculation uses
the previous observation's factor within the same stock for returns and gaps.
Momentum, volatility, the causal price index, moving-average features and ranks
were recomputed for all 2,332,531 rows. No clipping or row removal was used.
Target values, raw prices, keys, financial data and metadata are unchanged.

Regression tests use actual 2019 split/reverse-split observations for stocks 1414
and 7638 and check that later observations do not rewrite earlier features:

    python -m unittest discover -s tests -v

Before/after development-data diagnostics are in reports/price_adjustment_fix.json.
The feature schema remains 46 numeric features plus the existing categorical features.

Original source, feature table, notebooks, models and reports are preserved under:

    archive/price_adjustment_fix_20261106/

Old trained models, tuning scores, selected_model.json and other generated results
were moved out of active paths because they describe the earlier feature values.
Rerun notebook 03 and training/validation in 04 to obtain a valid new model choice.
Review any previously written numerical conclusions. Notebook 05 can then be used
for reevaluation, but its 2021 period has already been inspected in the original
project and cannot be described as a newly unseen test.

Project 2 reads the corrected table directly. Restart its kernel, choose a new
run_name if earlier source checksums were saved, and rerun from notebook 01.
