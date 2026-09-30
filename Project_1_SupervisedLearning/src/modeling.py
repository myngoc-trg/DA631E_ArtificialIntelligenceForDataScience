from __future__ import annotations

from ast import Tuple
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
import json
from multiprocessing import Value
from pathlib import Path
from re import split

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


from config import ProjectConfig


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


def development_and_test_split(
    features_table: pd.DataFrame
    ,config: ProjectConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    '''After model selection, combine train+validation and preserve test isolation'''


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
            ("numeric", numeric, list(numeric_features))
            ,("categorical", categorical, list(categorical_features))
        ]
        ,remainder="drop"
        ,sparse_threshold=0.0
        ,verbose_feature_names_out=True
    )
    
def build_model_pipeline(
    model_name: str
    ,numeric_features: Sequence[str]
    ,categorical_features: Sequence[str]
    ,random_seed: int = 42
    ,fast_mode: bool = True
) -> Pipeline:
    '''Return a full preprocessing pipeline
    
    Calling fit uses fit_transform on training data. Calling predict uses only transform 
    on validation/test data, preventing separate preprocessing logic.
    '''
    
    if model_name == 'ridge':
        return Pipeline(
            steps=[
                ("preprocess", build_linear_preprocessor(numeric_features, categorical_features))
                ,("model", Ridge(alpha=10.0, solver="lsqr"))
            ]
        )
    
    if model_name == 'hist_gradient_boosting':
        return Pipeline(
            steps=[
                ("preprocess", build_tree_preprocessor(numeric_features, categorical_features))
                ,("model", HistGradientBoostingRegressor(
                    learning_rate=0.05
                    ,max_iter = 80 if fast_mode else 180
                    ,max_leaf_nodes=31
                    ,min_samples_leaf=40
                    ,l2_regularization=1.0
                    ,early_stopping=True
                    ,validation_fraction=0.10
                    ,n_iter_no_change=10
                    ,random_state=random_seed
                    )
                )
            ]
        )
    raise ValueError(
        f"Unknown model_name={model_name!r}; choose 'ridge' or 'hist_gradient_boosting'"
    )
    
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

#def daily_information_coefficient

def _spread_return_one_day(
    day: pd.DataFrame
    ,portfolio_size: int
    ,top_rank_weight_ratio: float
    ,strict: bool
) -> float:
    '''Evaluates how well the model ranked stocks for one date, using their actual future returns.
    
    Args:
        - day: pd.DataFrame; a table contains stock for one day
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

def evaluate_predictions(
    df: pd.DataFrame
    ,predictions: np.ndarray | pd.Series
    ,config: ProjectConfig
    ,strict_portfolio_size: bool = True
#) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
) -> tuple[pd.Series, pd.DataFrame]:
    result = df[['Date','SecuritiesCode','Target']].copy()
    result['Prediction'] = np.asarray(a=predictions, dtype=float)
    #result = rank_prediction(result)
    
    y_true = result['Target'].to_numpy()
    y_pred = result['Prediction'].to_numpy()
    
    
    metrics = pd.Series({
        'MAE': mean_absolute_error(y_true, y_pred)
        ,'RMSE': mean_squared_error(y_true,y_pred)
        ,'R2': r2_score(y_true,y_pred)
    }
    ,name="value"
    )
    
    return metrics, result