"""Shared settings for walk-forward development through 2020 and a 2021 test."""
from dataclasses import dataclass
from datetime import date
from typing import Literal
import re


PROJECT1_ROOT = "../Project_1_SupervisedLearning"


@dataclass(frozen=True)
class ProjectConfig:
    random_seed: int = 42
    run_profile: Literal["quick", "full"] = "full"
    run_name: str | None = "01_data_exploration"
    
    train_end:str = "2019-12-31"
    
    valid_start: str = "2020-01-01"
    valid_end: str = "2020-12-31"
    
    feature_selection_end: str = "2017-12-31"
    development_start: str = "2017-01-01"
    development_end: str = "2020-12-31"
    
    test_start: str = "2021-01-01"
    test_end: str = "2021-12-31"

    stock_list_path: str = f"{PROJECT1_ROOT}/data/train/stock_list.csv"
    project1_features_path: str = f"{PROJECT1_ROOT}/data/processed/features_table.parquet"
    project1_feature_audit_path: str = f"{PROJECT1_ROOT}/reports/features_audit.json"
    development_features_path: str = "data/processed/development_features.parquet"
    development_audit_path: str = "data/processed/development_audit.json"
    pred_path: str | None = None
    pred_column: str = "Prediction"

    min_history_days: int = 60
    max_feature_missing_fraction: float = 0.8
    correlation_threshold: float = 0.95

    def __post_init__(self):
        if self.run_name is not None and (
            not isinstance(self.run_name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_name)
        ):
            raise ValueError("run_name must be None or use letters, digits, underscores and hyphens.")
        if self.run_profile not in {"quick", "full"}:
            raise ValueError("run_profile must be quick or full.")
        if type(self.min_history_days) is not int or self.min_history_days < 1:
            raise ValueError("min_history_days must be a positive integer.")
        for name in ("max_feature_missing_fraction", "correlation_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if not isinstance(self.pred_column, str) or not self.pred_column:
            raise ValueError("pred_column must be a non-empty string.")
        development_start, feature_selection_end, train_end, development_end, test_start, test_end = [
            date.fromisoformat(getattr(self, name))
            for name in (
                "development_start", "feature_selection_end", "train_end",
                "development_end", "test_start", "test_end"
            )
        ]
        if not development_start <= feature_selection_end <= train_end < development_end < test_start <= test_end:
            raise ValueError("Feature selection, development and test periods must be ordered.")
        if any((boundary.month, boundary.day) != (12, 31) for boundary in (
            feature_selection_end, train_end, development_end
        )):
            raise ValueError("Annual walk-forward folds require year-end selection and development boundaries.")


def options(cfg: ProjectConfig) -> dict:
    """Full uses all GMM rows; only the Ward benchmark remains capped.

    Quick reduces the search and row budgets for workflow checks. The reusable
    development dataset and the 2017 feature lists always retain all eligible rows.
    """
    if cfg.run_profile == "quick":
        return {"k": list(range(2, 7)), "covariance": ["diag", "full"], "n_init": 2, "max_iter": 200,
                "train_cap": 50000, "validation_cap": 50000, "benchmark_cap": 700,
                "stability_repeats": 3, "random_seeds": [24], "block_dates": 10}
    return {"k": list(range(2, 7)), "covariance": ["diag", "full"], "n_init": 5, "max_iter": 300,
            "train_cap": None, "validation_cap": None, "benchmark_cap": 2000,
            "stability_repeats": 20, "random_seeds": [4, 14, 24, 34, 44], "block_dates": 20}
