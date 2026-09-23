from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

@dataclass(frozen=True)
class ProjectConfig:
    """ Store all choices used by all notebooks (universal choices)"""
    
    random_seed: int=42
    train_end:str = "2019-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2020-12-31"
    test_start: str ="2021-01-01"
    test_end: str = "2021-12-31"
    
    purge_dates: int=2 #number of trading trades removed immediately before a train/validation or validation/test boundary. 
    # TO prevent the target calculation from crossing into the next dataset (as Target_t is calculated from Close_t+1 and Close_t+2)
    
    financial_lag_days: int=1
    market_no_trade_threshold:float = 0.9
    portfolio_size:int = 200
    
    @property
    def timestamp_bounds(self) -> dict[str, pd.Timestamp]:
        return {
            "train_end":pd.Timestamp(self.train_end)
            ,"valid_start": pd.Timestamp(self.valid_start)
            ,"valid_end": pd.Timestamp(self.valid_end)
            ,"test_start": pd.Timestamp(self.test_start)
            ,"test_end": pd.Timestamp(self.test_end)
        }
        

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    
    @property
    def train_raw(self) -> Path:
        return self.root / "data" / "train"
    
    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"
    
    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"
    
    @property
    def models(self) -> Path:
        return self.root / "models"
    
    @property
    def reports(self) -> Path:
        return self.root / "reports"
    
    @property
    def figures(self) -> Path:
        return self.reports / "figures"
    
    def ensure_output_dirs(self) -> None:
        for dir in (self.interim, self.processed, self.models, self.reports, self.figures):
            dir.mkdir(parents=True, exist_ok=True)    
            
def find_project_root(start: Path = Path.cwd()) -> Path:
    """Find the parent directory containing src/data.py."""
    for candidate in (start, *start.parents):
        if (candidate / "src" / "data.py").is_file():
            return candidate

    raise FileNotFoundError("Could not find src/data.py")