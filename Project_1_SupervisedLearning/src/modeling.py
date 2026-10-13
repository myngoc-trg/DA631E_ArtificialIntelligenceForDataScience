from __future__ import annotations

from ast import Tuple
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
import json
from multiprocessing import Value
from pathlib import Path
from re import split
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


from config import ProjectConfig, Candidate, TuningResults
#import lightgbm
from threadpoolctl import threadpool_limits
import time
import gc

def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    

def _purge_tail_dates(df: pd.DataFrame, n_dates: int) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    dates = pd.Index(data=df['Date'].drop_duplicates().sort_values())
    purged = list(dates[-n_dates:]) if n_dates else []
    
    '''
    dates = df['Date].drop_duplicates().sort_values()
    purged = dates.tail(n_dates).tolist()
    '''
    
    return df.loc[~df['Date'].isin(purged)].copy(), purged


def chronological_split(features_table: pd.DataFrame
                        , config: ProjectConfig
                        ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    
    '''Create train/validation/test sets and purge labels across boundaries'''
    
    df = features_table.loc[features_table['Target'].notna()].copy()
    bounds = config.timestamp_bounds
    
    train_raw = df.loc[df['Date'].le(bounds['train_end'])].copy()
    valid_raw = df.loc[df['Date'].between(left=bounds['valid_start'], right=bounds['valid_end'])].copy()
    test = df.loc[df['Date'].between(left=bounds['test_start'], right=bounds['test_end'])].copy()
    
    train, train_purged = _purge_tail_dates(df=train_raw, n_dates=config.purge_dates)
    valid, valid_purged = _purge_tail_dates(df=valid_raw, n_dates=config.purge_dates)
    splits = {'train': train, 'validation': valid, 'test':test}
    
    if any(frame.empty for frame in splits.values()):
        sizes = {name: len(frame) for name, frame in splits.items()}
        raise ValueError(f"At least one chronological split is empty: {sizes}")
    
    if train['Date'].max() >= valid['Date'].min():
        raise AssertionError('Train and validation dates overlap')
    
    if valid['Date'].max() >= test['Date'].min():
        raise AssertionError('validation and test dates overlap')

    report_rows = []
    for name, frame in splits.items():
        report_rows.append({
            'split': name
            ,'rows': len(frame)
            ,'dates': frame['Date'].nunique()
            ,'securities_codes': frame['SecuritiesCode'].nunique()
            ,'date_min': frame['Date'].min()
            ,'date_max': frame['Date'].max()
            ,'target_missing': frame['Target'].isna().sum()
        })
        
    report = pd.DataFrame(report_rows).set_index("split")
    report.attrs['train_purged_dates'] = [str(date.date()) for date in train_purged]
    report.attrs['validation_purged_dates'] = [str(date.date() for date in valid_purged)]
    return splits, report


def walk_forward_split(
    features_table: pd.DataFrame
    ,train_end: str | pd.Timestamp
    ,valid_start: str | pd.Timestamp
    ,valid_end: str | pd.Timestamp
    ,purged_dates: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    '''Create one expanding-window train/validation fold
    
    The final training dates (the last 2) are purged because their Target values use
    future prices that may belong to the validation set. This prevents data leakage.
    '''
    
    required = {'Date','SecuritiesCode','Target'}
    missing = sorted(required.difference(features_table.columns))
    if missing:
        raise ValueError(f"Features table is missing required columns: {missing}")
    
    if purged_dates < 0:
        raise ValueError(f"purged_dates must be non-negative, not {purged_dates}")
    
    train_end = pd.Timestamp(train_end)
    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)
    
    if train_end >= valid_start:
        raise ValueError(f"train_end={train_end} must be before valid_start={valid_start}")
    
    if valid_start > valid_end:
        raise ValueError(f"valid_start={valid_start} must be on or before valid_end={valid_end}")
    
    #Only labeled observations can be used for supervised learning. Unlabeled observations are ignored.
    df_labeled = features_table.loc[features_table['Target'].notna()].copy()
    train_raw = df_labeled.loc[df_labeled['Date'].le(train_end)].copy()
    valid = df_labeled.loc[df_labeled['Date'].between(left=valid_start, right=valid_end, inclusive='both')].copy()

    if train_raw.empty:
        raise ValueError(f"Train set is empty. No rows with Date <= {train_end}")
    if valid.empty:
        raise ValueError(f"Validation set is empty. No rows with {valid_start} <= Date <= {valid_end}")
    
    train, train_purged = _purge_tail_dates(df=train_raw, n_dates=purged_dates)
    
    if train.empty:
        raise ValueError(f"Train set is empty after purging {purged_dates} dates")
    
    sort_columns = ['Date','SecuritiesCode']
    train = train.sort_values(sort_columns).reset_index(drop=True)
    valid = valid.sort_values(sort_columns).reset_index(drop=True)
    
    if train['Date'].max() >= valid['Date'].min():
        raise AssertionError(f"Train and validation dates overlap: train max={train['Date'].max()}, valid min={valid['Date'].min()}")
    if train.duplicated(subset=sort_columns).any():
        raise AssertionError("Train set has duplicate (Date, SecuritiesCode) pairs")
    if valid.duplicated(subset=sort_columns).any():
        raise AssertionError("Validation set has duplicate (Date, SecuritiesCode) pairs")
    
    if train['Target'].isna().any():
        raise AssertionError("Train set has missing Target values")
    if valid['Target'].isna().any():
        raise AssertionError("Validation set has missing Target values")
    
    return train, valid, train_purged


def make_fold_metric_record(
    train_full: pd.DataFrame
    ,train_purged: pd.DataFrame
    ,valid: pd.DataFrame
    ,train_purged_dates: list[pd.Timestamp]
    ,model_name: str
    ,fold_name: str
    ,metrics: pd.Series
    ,training_seconds: float
) -> pd.DataFrame:
    '''Create a result row for one model and one fold'''
    
    record = {
        'model_name': model_name
        ,'fold_name':fold_name
        ,'train_start': train_full['Date'].min()
        ,'train_end': train_full['Date'].max()
        ,'validation_start': valid['Date'].min()
        ,'validation_end': valid['Date'].max()
        ,'train_purged_dates': [str(date.date()) for date in train_purged_dates]
        ,'train_full_rows': len(train_full)
        ,'train_after_purged_rows': len(train_purged)
        ,'validation_rows': len(valid)
        ,'train_full_dates': train_full['Date'].nunique()
        ,'train_after_purged_dates': train_purged['Date'].nunique()
        ,'validation_dates': valid['Date'].nunique()
        ,'training_seconds': training_seconds
    }
    
    record.update(metrics.to_dict())
    
    return record


def development_and_test_split(
    features_table: pd.DataFrame
    ,config: ProjectConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    '''After model selection, combine train+validation and preserve test isolation'''
    df = features_table.loc[features_table['Target'].notna()].copy()
    bounds = config.timestamp_bounds
    development_raw = df.loc[df['Date'].le(bounds['valid_end'])].copy()
    development, purged = _purge_tail_dates(development_raw,config.purge_dates)
    test = df.loc[df['Date'].ge(bounds['test_start'])].copy()
    if development['Date'].max() > test['Date'].min():
        raise AssertionError("Development and test dates overlap")
    
    return development, test, purged

def select_recent_dates(df: pd.DataFrame, max_dates: int | None) -> pd.DataFrame:
    '''Deterministic fast mode: keep complete cross-sections for recent dates'''
    if max_dates is None:
        return df.copy()
    dates = df['Date'].drop_duplicates().sort_values()
    if len(dates) <= max_dates:
        return df.copy()
    keep = set(dates.iloc[-max_dates:])
    return df.loc[df['Date'].isin(keep)].copy()
        
    
def make_xy(
    df: pd.DataFrame
    ,numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = list(numeric_features) + list(categorical_features)
    missing = sorted(set(feature_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Frame is missing model features: {missing}")
    if 'Target' in feature_columns:
        raise AssertionError("Target must never be in X")
    
    X = df[feature_columns].copy()
    
    for col in categorical_features:
        X[col] = X[col].astype("object").where(X[col].notna(), other=np.nan)
        
    return X, df['Target'].astype(float).copy()


def build_linear_preprocessor(
    numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
            ,("scaler", StandardScaler())
        ]
    )
    
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))
            ,("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True))
        ]
    )
    
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(numeric_features))
            ,("categorical", categorical, list(categorical_features))
        ]
        ,remainder="drop"
        ,sparse_threshold=0.3
        ,verbose_feature_names_out=True
    )
    
    
def build_tree_preprocessor(
    numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
            #,("scaler", StandardScaler())
        ]
    )
    
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))
            ,("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1))
        ]
    )
    
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(categorical_features))
            ,("numeric", numeric, list(numeric_features))
        ]
        ,remainder="drop"
        ,sparse_threshold=0.0
        ,verbose_feature_names_out=True
    )


def order_for_daily_ranking(
    df: pd.DataFrame    
) -> pd.DataFrame:
    ''' Order rows so that evert date forms one contiguous ranking group
        Daily ranking from Target
    '''

    required = {'Date','SecuritiesCode'}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prediction df is missing: {missing}")

    ordered = df.sort_values(['Date','SecuritiesCode'], kind="mergesort").reset_index(drop=True)
    if not ordered['Date'].is_monotonic_increasing:
        raise AssertionError('Ranking data must be ordered by Date')
    return ordered


def make_daily_relevance_labels(
    ordered_df : pd.DataFrame
    ,n_relevance_bins: int= 20
) -> np.ndarray:
    '''Convert daily continous Target values into ordinal relevance labels (Regression to Classification)
    
    Label 0: the lowest daily targets
    Label n_relevance_bins - 1: the highest daily Targets
    '''

    if n_relevance_bins < 2:
        raise ValueError("n_relevance bins must be at least 2")

    if 'Target' not in ordered_df.columns:
        raise ValueError("Target is required when creating relevance labels")

    if ordered_df['Target'].isna().any():
        raise ValueError('Ranking training data contains missing Target')

    daily_percentile = ordered_df.groupby('Date')['Target'].rank(method="average", pct=True)

    relevance = np.floor(daily_percentile.to_numpy(dtype=float) * n_relevance_bins)
    relevance = np.clip(a=relevance, a_min=0,a_max=n_relevance_bins-1) #to make sure the 1-percentile falls in the last bin

    return relevance.astype('int16')

def make_daily_relevance_sizes(
    ordered_df: pd.DataFrame
) -> np.ndarray:
    '''Return the number of stocks belonging to each trading date
    
    LightGBM uses this to identify the individual ranking problems
    '''

    if not ordered_df['Date'].is_monotonic_increasing():
        raise AssertionError("Ranking data must be ordered by Date. Order the df by Date before creating ranking group")

    group_sizes = ordered_df.groupby('Date', sort=False).size()

    if group_sizes.sum() != len(ordered_df):
        raise AssertionError("Ranking group sizes do not sum to the number of rows in df")

    return group_sizes

'''
def build_lambdarank_pipeline(
    numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
    ,portfolio_size: int = 200
    ,n_relevance_bins: int = 20
    ,random_seed:int = 42
    ,fast_mode:bool =False
):
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:
        raise ImportError(
            "LightGBM is required for LambdaRank"
            "Install it with: python -m pip install lightgbm"
        ) from error

    model = LGBMRanker(
        objective ="lambdarank"
        ,metric="ndcg" "evaluate ranking, rewarding high-relevance stocks near the top. Higher NDCG is better"

    )
    return 
'''


    
def build_model_pipeline(
    model_name: str
    ,numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
    ,random_seed: int = 42
    ,fast_mode: bool = False
    ,model_params: Mapping[str, Any] | None = None
) -> Pipeline:
    '''Return a full preprocessing pipeline
    
    Calling fit uses fit_transform on training data. Calling predict uses only transform 
    on validation/test data, preventing separate preprocessing logic.
    
    Replacement: candidate overrides are applied before fitting
    '''
    
    overrides = dict(model_params or {})
    if 'Target' in set(numeric_features) | set(categorical_features):
        raise ValueError("Target must never be a model feature.")
    if set(numeric_features) & set(categorical_features):
        raise ValueError("Numeric and categorical feature lists must not overlap")
    
    if model_name == 'ridge':
        params = {"alpha": 10.0, "solver": "lsqr"}
        params.update(overrides)
        return Pipeline(
            steps=[
                ("preprocess", build_linear_preprocessor(numeric_features, categorical_features))
                ,("model", Ridge(**params))
            ]
        )
    
    if model_name == 'hist_gradient_boosting':
        if "categorical_features" in overrides or "random_state" in overrides:
            raise ValueError("Set categorical columns and random_seed through the builder")
        params = {
            'loss': "squared_error"
            ,'learning_rate': 0.05
            ,'max_iter': 80 if fast_mode else 200
            ,'max_leaf_nodes': 15
            ,"max_depth": 6
            ,"min_samples_leaf": 100
            ,"l2_regularization": 10.0
            ,"early_stopping": False
            ,"random_state": random_seed
            ,"categorical_features": list(range(len(categorical_features))) if len(categorical_features) else None
        }
        params.update(overrides)
        return Pipeline(
            steps=[
                ("preprocess", build_tree_preprocessor(numeric_features, categorical_features))
                ,("model", HistGradientBoostingRegressor(**params))
            ]
        )
    raise ValueError(
        f"Unknown model_name={model_name!r}; choose 'ridge' or 'hist_gradient_boosting'"
    )
    
def make_hgb_stage1_candidates(include_reference: bool = True) -> List[Candidate]:
    fixed = {
        'learning_rate': 0.05
        ,'max_iter': 200
        ,"max_depth": 6
        ,"l2_regularization": 10.0
        ,"early_stopping": False
        }
    candidates = []
    if include_reference:
        candidates.append(
            Candidate("hgb_reference", "hist_gradient_boosting"
                    ,{'learning_rate': 0.05
                        ,'max_iter': 180
                        ,"max_leaf_nodes": 31
                        ,"max_depth": None
                        ,"min_samples_leaf": 40
                        ,"l2_regularization": 1.0
                        ,"early_stopping": False,
                    })
        )
    
    grid = ParameterGrid({'max_leaf_nodes':[7,15,31]
                        ,"min_samples_leaf": [100,500]})
    
    candidates.extend(Candidate(f"hgb_s1_{i:02d}", "hist_gradient_boosting"
                        ,{**fixed, **params})
                        for i, params in enumerate(grid, start=1)
                    )
    return candidates

def make_ridge_candidates() -> list[Candidate]:
    return [Candidate(f"ridge_alpha_{alpha:g}", "ridge", {"alpha": alpha})
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]
            ]

def make_hgb_stage2_candidates(
    best_params: Mapping[str, Any]
    ,tested_candidates: Sequence[Candidate] = ()
) -> list[Candidate]:
    signatures = {json.dumps(c.params,sort_keys=True) 
                for c in tested_candidates
                if c.model_name == "hist_gradient_boosting"
                }
    candidates = []
    grid = ParameterGrid({"l2_regularization": [0.0, 10.0, 100.0]
                        ,"max_iter": [100, 200, 400]})
    
    for i, val in enumerate(grid, start=1):
        params = {**dict(best_params), **val, "early_stopping": False}
        if json.dumps(params, sort_keys=True) not in signatures:
            candidates.append(Candidate(f"hgb_s2_{i:02d}", "hist_gradient_boosting", params))
    return candidates

def summarize_search(fold_results: pd.DataFrame, expected_folds: int) -> pd.DataFrame:
    """Rank complete candidates by mean validation Sharpe, with explicit tie breaks."""
    scored = fold_results.loc[~fold_results["is_baseline"]].copy()
    if scored.empty:
        raise ValueError("There are no supervised candidate results.")
    if scored.duplicated(["candidate_name", "fold_name"]).any():
        raise ValueError("A candidate has duplicate results for a fold.")
    if not np.isfinite(scored[["SpreadSharpeRatio", "MeanDailyIC"]].to_numpy()).all():
        raise ValueError("A candidate has undefined validation Sharpe or mean IC.")
    
    groups = scored.groupby("candidate_name", sort=False)
    if not groups["fold_name"].nunique().eq(expected_folds).all():
        raise ValueError("All candidates must complete the same number of folds.")
    if not groups["params_json"].nunique().eq(1).all():
        raise ValueError("A candidate's parameters changed between folds.")
    
    fold_sets = groups["fold_name"].agg(lambda values: tuple(sorted(values)))
    if fold_sets.nunique() != 1:
        raise ValueError("Candidates were evaluated on different folds.")
    
    summary = groups.agg(
        model_name=("model_name", "first"), params_json=("params_json", "first")
        ,folds=("fold_name", "nunique")
        ,mean_sharpe=("SpreadSharpeRatio", "mean")
        ,worst_sharpe=("SpreadSharpeRatio", "min")
        ,std_sharpe=("SpreadSharpeRatio", "std")
        ,mean_ic=("MeanDailyIC", "mean"), worst_ic=("MeanDailyIC", "min")
        ,mean_mse=("MSE", "mean")
        ,mean_train_sharpe=("Train_SpreadSharpeRatio", "mean")
        ,mean_train_ic=("Train_MeanDailyIC", "mean")
        ,mean_sharpe_gap=("SharpeGap", "mean")
        ,min_ic_coverage=("ValidICCoverage", "min")
        ,total_fit_seconds=("training_seconds", "sum")
    )
    return summary.sort_values(
        ["mean_sharpe", "worst_sharpe", "std_sharpe", "mean_ic"]
        ,ascending=[False, False, True, False], kind="mergesort"
    )
    

def run_walk_forward_search(
    features_table: pd.DataFrame
    ,folds: Sequence[Mapping[str, Any]]
    ,numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
    ,config: Any
    ,candidates: Sequence[Candidate]
    ,helpers: Any
    ,*
    ,include_momentum: bool = True
    ,threads: int = 4
    ,progress_csv: str | Path | None = None
) -> TuningResults:
    """Use helpers=modeling; fit a NEW pipeline for each candidate and fold.

    Each fold's final config.purge_dates validation dates are excluded, so labels
    at the end of 2020 cannot use 2021 prices under the configured horizon.
    All candidate comparisons use the same validation rows. Progress CSV, if
    supplied, is rewritten after each completed candidate for this run.
    """
    candidates, folds = list(candidates), list(folds)
    if not candidates or not folds:
        raise ValueError("Provide at least one candidate and one fold.")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique.")
    if len({f["name"] for f in folds}) != len(folds):
        raise ValueError("Fold names must be unique.")
    if threads < 1 or config.purge_dates < 2:
        raise ValueError("Use threads >= 1 and at least two purge dates for the target.")
    
    test_start = pd.Timestamp(config.timestamp_bounds["test_start"])
    dates = pd.to_datetime(features_table["Date"], errors="raise")
    
    if dates.isna().any():
        raise ValueError("Date contains missing timestamps.")
    
    development_mask = dates.lt(test_start)
    development = features_table.loc[development_mask].copy()
    development["Date"] = dates.loc[development_mask].to_numpy()
    
    for fold in folds:
        if pd.Timestamp(fold["valid_end"]) >= test_start:
            raise ValueError(f"Fold {fold['name']} reaches the held-out test period.")
    records, daily_ic_record, daily_spread_record = [], {}, {}

    def append_result(candidate,fold
                    ,train,valid
                    ,full_train, purged,valid_purged
                    ,train_metrics, valid_metrics
                    ,ic, spread,fit_seconds
                    ,iterations, is_baseline):
        record = dict(helpers.make_fold_metric_record(
            train_full=full_train, train_purged=train, valid=valid,
            train_purged_dates=purged, model_name=candidate.model_name,
            fold_name=fold["name"], metrics=valid_metrics,
            training_seconds=fit_seconds,
        ))
        record.update({f"Train_{key}": float(value)
                        for key, value in train_metrics.items()})
        record.update(candidate_name=candidate.name
                    ,is_baseline=is_baseline
                    ,params_json=json.dumps(candidate.params, sort_keys=True)
                    ,n_iter=iterations
                    ,SharpeGap=float(train_metrics["SpreadSharpeRatio"] - valid_metrics["SpreadSharpeRatio"])
                    ,ValidICCoverage=float(ic.notna().mean())
                    ,ValidICDates=int(ic.notna().sum())
                    ,validation_purged_dates=[str(pd.Timestamp(d).date())
                                            for d in valid_purged])
        records.append(record)
        key = (candidate.name, fold["name"])
        daily_ic_record[key], daily_spread_record[key] = ic, spread
        if progress_csv is not None:
            path = Path(progress_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(path, index=False)

    for fold in folds:
        print(f"\nTraining fold {fold['name']}", flush=True)
        train, valid, purged = helpers.walk_forward_split(
            features_table=development
            ,train_end=fold["train_end"]
            ,valid_start=fold["valid_start"]
            ,valid_end=fold["valid_end"]
            ,purged_dates=config.purge_dates
        )
        valid_dates = pd.Index(valid["Date"].drop_duplicates().sort_values())
        valid_purged = list(valid_dates[-config.purge_dates:])
        valid = valid.loc[~valid["Date"].isin(valid_purged)].copy()
        if valid.empty:
            raise ValueError(f"Fold {fold['name']} has no validation rows after purging.")
        full_train = development.loc[
            development["Date"].le(pd.Timestamp(fold["train_end"]))
            & development["Target"].notna()
        ]
        X_train, y_train = helpers.make_xy(train, numeric_features, categorical_features)
        X_valid, y_valid = helpers.make_xy(valid, numeric_features, categorical_features)
        if len(X_train) != len(y_train) or len(X_valid) != len(y_valid):
            raise ValueError("Feature/target lengths do not match.")

        if include_momentum:
            baseline = Candidate("baseline_momentum20", "baseline_momentum20", {})
            tr, _, _, _ = helpers.evaluate_predictions(
                train, train["Momentum20Days"].fillna(0.0).to_numpy(), config,
                strict_portfolio_size=True,
            )
            va, ranked, ic, spread = helpers.evaluate_predictions(
                valid, valid["Momentum20Days"].fillna(0.0).to_numpy(), config,
                strict_portfolio_size=True,
            )
            append_result(baseline, fold, train, valid, full_train, purged,
                        valid_purged, tr, va, ic, spread, 0.0, 0, True)
            del ranked
            
        for candidate in candidates:
            pipeline = build_model_pipeline(
                candidate.model_name, numeric_features, categorical_features,
                random_seed=config.random_seed, fast_mode=False,
                model_params=candidate.params,
            )
            model = pipeline.named_steps["model"]
            if isinstance(model, HistGradientBoostingRegressor):
                if model.early_stopping is not False:
                    raise ValueError("Use early_stopping=False for this fixed-iteration search.")
                too_large = {name: int(X_train[name].nunique(dropna=True))
                            for name in categorical_features
                            if X_train[name].nunique(dropna=True) > model.max_bins}
                if too_large:
                    raise ValueError(
                        f"Native HGB categories exceed max_bins={model.max_bins}: {too_large}. "
                        "Remove identifier features or group rare categories using training data."
                    )
            print(f"{candidate.name}: {candidate.params}", flush=True)
            
            with threadpool_limits(limits=threads):
                started = time.perf_counter()
                pipeline.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - started
                train_predictions = np.asarray(pipeline.predict(X_train), dtype=float).reshape(-1)
                valid_predictions = np.asarray(pipeline.predict(X_valid), dtype=float).reshape(-1)
            if (len(train_predictions) != len(y_train)
                    or len(valid_predictions) != len(y_valid)
                    or not np.isfinite(train_predictions).all()
                    or not np.isfinite(valid_predictions).all()):
                raise ValueError(f"Invalid predictions from {candidate.name} in {fold['name']}.")
            tr, _, _, _ = helpers.evaluate_predictions(
                train, train_predictions, config, strict_portfolio_size=True,
            )
            va, ranked, ic, spread = helpers.evaluate_predictions(
                valid, valid_predictions, config, strict_portfolio_size=True,
            )
            actual_iterations = getattr(model, "n_iter_", None)
            iterations = (int(np.asarray(actual_iterations).max())
                        if actual_iterations is not None else np.nan)
            append_result(candidate, fold, train, valid, full_train, purged,
                        valid_purged, tr, va, ic, spread, fit_seconds, iterations, False)
            print(f"fit={fit_seconds:.1f}s, train Sharpe={tr['SpreadSharpeRatio']:.4f}, "
                f"valid Sharpe={va['SpreadSharpeRatio']:.4f}, "
                f"valid IC={va['MeanDailyIC']:.4f}", flush=True)
            del pipeline, model, train_predictions, valid_predictions, ranked
            gc.collect()
        del train, valid, full_train, X_train, y_train, X_valid, y_valid
        gc.collect()
    results = pd.DataFrame(records)
    return TuningResults(results, summarize_search(results, len(folds)),
                        daily_ic_record, daily_spread_record,
                        {c.name: c for c in candidates})

def rank_prediction(prediction_df: pd.DataFrame) -> pd.DataFrame:
    '''Assign daily ranks: 0 is the highest predicted return target
    
    This function measures whether the model correctly orders stocks from high to low future return on each trading day.
    '''
    
    required = {'Date','SecuritiesCode', 'Prediction'}
    missing = sorted(required.difference(prediction_df.columns))
    if missing:
        raise ValueError(f"Prediction df is missing: {missing}")
    ranked = prediction_df.copy()
    ranked = ranked.sort_values(
        ['Date','Prediction','SecuritiesCode']
        ,ascending=[True,False,True]
        ,kind="mergesort"
        )
    ranked['Rank'] = ranked.groupby('Date',sort=False).cumcount().astype("int32")
    return ranked.sort_values(['Date','SecuritiesCode']).reset_index(drop=True)


def _spread_return_one_day(
    day: pd.DataFrame
    ,portfolio_size: int
    ,top_rank_weight_ratio: float
    ,strict: bool
) -> float:
    '''Evaluates how well the model ranked stocks for one date, using their actual future returns.
    
    Args:
        - day: pd.DataFrame; a table contains stock for one day. SHould have columns: [Target, Rank]
        - portfolio_size: number of stocks selected on EACH SIDE. With 200 to buy and 200 to short
        - top_rank_weight_ratio: how much more weight the strongest selection receives relative to the weakest selection within each side
                                THis weight makes the most extreme selection count more heavily.
                                + On the buy side, the best-ranked stock receives the largest weight
                                + On the short side, the worst-ranked stock receives the largest wegiht    
        strict: whether to raise an error when too few usable stocks are available
    '''
    day = day.dropna(subset=['Target', 'Rank']).sort_values('Rank')
    if strict and len(day) < 2 * portfolio_size:
        raise ValueError(
            f"Date {day['Date'].iloc[0]} has {len(day)} rows."
            f"at least {2 * portfolio_size} are required"
        )
    size = portfolio_size if strict else min(portfolio_size, len(day) // 2)
    if size < 1:
        return np.nan
    weights = np.linspace(top_rank_weight_ratio, 1.0, size)
    purchase = np.dot(day.head(size)['Target'], weights) / weights.mean() #scaled return contribution from the buy side.
    short = np.dot(day.tail(size).iloc[::-1]['Target'], weights) / weights.mean()
    return float(purchase - short)

def spread_return_series(
    df_rankedPreds: pd.DataFrame
    ,portfolio_size: int = 200
    ,top_rank_weight_ratio: float = 2.0
    ,strict: bool = True
) -> pd.Series:
    '''Calculate one spread return for each date in the ranked prediction table, using the actual future returns.
    In other words, the model's predictions determine the stock's [Rank]. Their actual future returns, stored in [Target], 
    are used to calculate the spread return to determine how sucessful that ranking was.
    Args:
        - df_rankedPreds: Dataframe with columns ['Date','SecuritiesCode','Target','Rank'] where Rank is the model's predicted ranking of stocks for each date.
                            It can contain multiple dates.
        - portfolio_size: number of stocks selected on EACH SIDE. With 200 to buy and 200 to short
        - top_rank_weight_ratio: how much more weight the strongest selection receives relative to the weakest selection within each side
                                THis weight makes the most extreme selection count more heavily.
                                + On the buy side, the best-ranked stock receives the largest weight
                                + On the short side, the worst-ranked stock receives the largest wegiht
        - strict: whether to raise an error when too few usable stocks are available
        
    spread return shows how the ranking ability translates into the performance of your selected portfolios.
    '''
    
    required = {'Date','Target','Rank'}
    missing = sorted(required.difference(df_rankedPreds.columns))
    if missing:
        raise ValueError(f"Prediction df is missing: {missing}")
    
    values = {
        date: _spread_return_one_day(
            day=day
            ,portfolio_size=portfolio_size
            ,top_rank_weight_ratio=top_rank_weight_ratio
            ,strict=strict
        )
        for date, day in df_rankedPreds.groupby('Date', sort=True)
    }
    
    return pd.Series(data=values, name='DailySpreadReturn', dtype=float)


def daily_information_coefficient(
    df_rankedPreds: pd.DataFrame
    ,strict: bool = True
) -> pd.Series:
    '''Calculate one information coefficient for each date in the ranked prediction table, using the actual future returns.
    In other words, the model's predictions determine the stock's [Rank]. Their actual future returns, stored in [Target], 
    are used to calculate the information coefficient to determine how sucessful that ranking was.
    Args:
        - df_rankedPreds: Dataframe with columns ['Date','SecuritiesCode','Target','Rank'] where Rank is the model's predicted ranking of stocks for each date.
                            It can contain multiple dates.
        - strict: whether to raise an error when too few usable stocks are available
        
    A consistently positive IC provides evidence of useful ranking ability
    '''
    
    values = {}
    for date, day in df_rankedPreds.groupby('Date', sort=True):
        true_pred_labels = day[['Target','Prediction']].dropna()
        values[date] = (
            true_pred_labels['Prediction'].corr(true_pred_labels['Target'], method='spearman')
            if len(true_pred_labels) >= 3 else np.nan
            #Calculate Spearman correlation only if there are at least 3 valid rows. Otherwise, return NaN.
            #With only 2 distinct observations, the correlation is either +1 or -1, which is not informative. With 1 or 0 observations, correlation is undefined.
        ) 
    return pd.Series(data=values, name='DailyInformationCoefficient', dtype=float)

def evaluate_predictions(
    df: pd.DataFrame
    ,predictions: np.ndarray | pd.Series
    ,config: ProjectConfig
    ,strict_portfolio_size: bool = True
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
#) -> tuple[pd.Series, pd.DataFrame]:
    result = df[['Date','SecuritiesCode','Target']].copy()
    result['Prediction'] = np.asarray(a=predictions, dtype=float)
    result = rank_prediction(result)
    
    y_true = result['Target'].to_numpy()
    y_pred = result['Prediction'].to_numpy()
    daily_ic = daily_information_coefficient(result, strict=strict_portfolio_size)
    daily_spread_return = spread_return_series(result, portfolio_size=config.portfolio_size, top_rank_weight_ratio=config.top_rank_weight_ratio, strict=strict_portfolio_size)
    spread_std = daily_spread_return.std(ddof=1)
    spread_mean = daily_spread_return.mean()
    
    metrics = pd.Series({
        'MAE': mean_absolute_error(y_true, y_pred)
        ,'MSE': mean_squared_error(y_true,y_pred)
        ,'R2': r2_score(y_true,y_pred)
        ,'MeanDailyIC': daily_ic.mean()
        ,'MedianDailyIC': daily_ic.median()
        ,'MeanDailySpreadReturn': spread_mean
        ,'MedianDailySpreadReturn': daily_spread_return.median()
        ,'DailySpreadReturnStd': spread_std
        ,'SpreadSharpeRatio': spread_mean / spread_std if spread_std > 0 else np.nan
        # Mean: whether the top-ranked stocks outperform the bottom-ranked stocks on average.
        # Std: how much that performance varies from day to day. A high standard deviation means that the model's performance is inconsistent.
        # Different models can have different Sharpe ratios if one is more consistent than the other. A higher Sharpe ratio indicates a more reliable model.
        # 2 models can have same mean daily spread return, but one with higher std will have a lower Sharpe ratio, indicating that it is less reliable and more volatile in its performance.
    }
    ,name="value"
    )
    
    return metrics, result, daily_ic, daily_spread_return