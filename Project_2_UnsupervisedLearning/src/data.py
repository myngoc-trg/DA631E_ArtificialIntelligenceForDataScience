from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io_utils import read_json, sha256, verify_lock

KEYS = ["Date", "SecuritiesCode"]
CORE = ["LogReturn1Day", "Momentum5Days", "Momentum20Days", "Volatility20Days",
        "Volatility60Days", "VolatilityRatio20To60", "CloseToMovingAverage20Days", "VolumeVs20Days"]
FORBIDDEN = {"Date", "SecuritiesCode", "RowId", "Target", "Prediction", "State", "Cluster",
            "Open", "High", "Low", "Close", "CloseForFeatures", "CausalPriceIndex", "DayNumber", "HistoryCount"}



def normalize_keys(frame):
    ''' 
    Standardize and validate 2 columns that identify an observations: Date and SecuritiesCode. 
    Return a copy of the frame with normalized values. 
    Raise an error if any duplicates are found.
    '''
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="raise").dt.normalize()
    if result["SecuritiesCode"].isna().any():
        raise ValueError("SecuritiesCode saknas i en eller flera rader.")
    result["SecuritiesCode"] = result["SecuritiesCode"].astype(str).str.replace(r"\.0$", "", regex=True)
    if result.duplicated(KEYS).any():
        raise ValueError("Date + SecuritiesCode är inte unika.")
    return result.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)


def load_stock_metadata(root, cfg):
    meta = pd.read_csv(Path(root) / cfg.stock_list_path)
    if "SecuritiesCode" not in meta:
        raise ValueError("stock_list saknar SecuritiesCode.")
    meta["SecuritiesCode"] = meta["SecuritiesCode"].astype(str).str.replace(r"\.0$", "", regex=True)
    company = next((c for c in ["Name", "CompanyName"] if c in meta), None)
    sector = next((c for c in ["17SectorName", "33SectorName", "Sector"] if c in meta), None)
    result = pd.DataFrame({"SecuritiesCode": meta["SecuritiesCode"],
                        "CompanyName": meta[company] if company else "Unknown",
                        "Sector": meta[sector] if sector else "Unknown"})
    if result.duplicated("SecuritiesCode").any():
        raise ValueError("Bolagsinformationen innehåller flera rader per SecuritiesCode. Granska innan join.")
    return result


def feature_family(name):
    if name.startswith("Fin_") or name == "FinancialAgeDays":
        return "financial"
    if any(s in name for s in ["Flag", "Dividend", "LastTrade"]):
        return "events_quality"
    if "Momentum" in name or "MovingAverage" in name:
        return "momentum"
    if "Volatility" in name or "Range" in name:
        return "risk"
    if any(s in name for s in ["Volume", "Turnover"]):
        return "activity"
    return "returns"


def load_daily_until(root, cfg, end, *, with_target=False, allow_holdout=False, out=None):
    ''' 
    Load daily features from the earliest available date through ``end`` and apply
    basic quality checks.
    Returns a DataFrame and a dictionary of summary information.
    
    Inputs:
    - root: Project 2 directory used to resolve the configured input paths
    - cfg: ProjectConfig object with universal choices
    - end: last date to include (inclusive), as a string in YYYY-MM-DD format
    - with_target: if True, include the Target column in the returned DataFrame
    - allow_holdout: Explicitly allow access to the final test period when enabled.
    - out: Experiment directory containing the saved model-selection checkpoint.
    
    * means subsequent arguments must be supplied by name, such as with_target=True
    
    A row must have:
    - Positive closing price and trading volume.
    - NoTradeFlag == 0.
    - Non-missing values for the four listed return, momentum and volatility features.
    - Enough earlier observations according to cfg.min_history_days.
    '''
    
    end = pd.Timestamp(end)
    # checks whether the requested period overlaps with the final test period, which is only allowed after model selection and locking
    if end >= pd.Timestamp(cfg.test_start):
        if not allow_holdout or out is None:
            raise RuntimeError("2021 öppnas först i final notebook efter modellval och låsning.")
        verify_lock(out, cfg)
        
    feature_audit = read_json(Path(root) / cfg.project1_feature_audit_path)
    path = Path(root) / cfg.project1_features_path
    allowed = feature_audit["numeric_features"]
    schema = pq.ParquetFile(path).schema_arrow.names # check the parquet file's columns
    required = ["Date", "SecuritiesCode", "Close", "Volume"] + allowed
    missing = sorted(set(required) - set(schema))
    if missing:
        raise ValueError(f"P1-tabellen matchar inte det granskade kontraktet: {missing}")

    optional = [c for c in ["FinancialAvailableDate"] if c in schema]
    columns = list(dict.fromkeys(required + optional + (["Target"] if with_target else [])))
    d = pd.read_parquet(path, columns=columns, filters=[("Date", ">=", pd.Timestamp(cfg.development_start)), ("Date", "<=", end)])
    d = normalize_keys(d)
    
    
    '''When FinancialAvailableDate exists, the function rejects rows whose financial information 
    became available after the observation’s date. This helps detect future information entering 
    historical features.'''
    if "FinancialAvailableDate" in d:
        if pd.to_datetime(d["FinancialAvailableDate"]).gt(d["Date"]).fillna(False).any():
            raise ValueError("En finansiell uppgift publicerades efter radens datum. Granska Projekt1-tabellen.")
    
    before = len(d)
    # En riktig observerad marknadshandelsdag finns när någon aktie har pris och positiv volym
    traded = d["Close"].gt(0) & d["Volume"].gt(0)
    market_dates = d.loc[traded, "Date"].drop_duplicates().sort_values()
    
    #DayNumber numbers observed market-trading dates in order. Later code uses it to identify consecutive trading observations
    positions = pd.Series(np.arange(len(market_dates)), index=market_dates)
    d["DayNumber"] = d["Date"].map(positions)
    
    # HistoryCount counts earlier rows for each stock in the loaded period
    d["HistoryCount"] = d.groupby("SecuritiesCode").cumcount()
    required_history = d[["LogReturn1Day", "Momentum20Days", "Volatility20Days", "Volatility60Days"]].notna().all(axis=1)
    eligible = traded & d["NoTradeFlag"].eq(0) & required_history & d["HistoryCount"].ge(cfg.min_history_days)
    d = d.loc[eligible].copy()
    if d.empty:
        raise ValueError("Inga rader återstår efter historik- och handelskriterierna. Kontrollera perioden och datan.")
    for c in allowed:
        d[c] = pd.to_numeric(d[c], errors="raise").replace([np.inf, -np.inf], np.nan)
    d["RowKey"] = d["Date"].dt.strftime("%Y-%m-%d") + "|" + d["SecuritiesCode"].astype(str)
    if d["RowKey"].duplicated().any():
        raise ValueError("Date + SecuritiesCode är inte unika.")
    d = d.set_index("RowKey").sort_index()
    info = {"rows_before_filter": before, "rows_after_filter": len(d), "stocks": d.SecuritiesCode.nunique(),
            "dates": d.Date.nunique(), "min_date": str(d.Date.min().date()), "max_date": str(d.Date.max().date()),
            "input_features_available": len(allowed), "allowed_features": allowed}
    return d, info

def balanced_sample(frame, cap: int | None, random_seed=42):
    """Urval balanserat över datum; cap=None behåller alla rader."""
    if cap is None or len(frame) <= cap:
        return frame.copy()
    
    rng = np.random.default_rng(random_seed)
    by_date = frame.groupby("Date", sort=True)
    per_date = max(1, int(np.ceil(cap / frame.Date.nunique())))
    indices = []
    # for each date, and its Dataframe group, randomly select up to per_date rows without replacement
    for _, group in by_date:
        indices.extend(rng.choice(group.index.to_numpy(), size=min(len(group), per_date), replace=False))
    if len(indices) > cap:
        indices = rng.choice(np.asarray(indices), size=cap, replace=False)
    return frame.loc[indices].sort_index().copy()


def make_feature_sets(selection_rows, allowed_features, config):
    """Build four feature sets and report feature quality and correlation removals.

    selection_rows contains the training-period rows used for feature selection.
    allowed_features lists permitted model inputs. config supplies the missingness
    and correlation thresholds. Returns feature sets, a catalog and a removal table.
    """
    if set(allowed_features) & FORBIDDEN:
        raise ValueError("Mål, nycklar eller prisnivåer har hamnat bland inputfeatures.")

    feature_catalog = pd.DataFrame(index=allowed_features)
    feature_catalog.index.name = "feature"
    feature_catalog["family"] = [feature_family(feature) for feature in allowed_features]
    feature_catalog["missing_fraction"] = selection_rows[allowed_features].isna().mean()
    feature_catalog["unique"] = selection_rows[allowed_features].nunique(dropna=True)
    # A feature is usable when it varies and passes the missingness threshold.
    feature_catalog["usable"] = (
        feature_catalog["unique"].gt(1)
        & feature_catalog["missing_fraction"].le(config.max_feature_missing_fraction)
    )
    # Set A: all usable features.
    usable_features = feature_catalog.index[feature_catalog["usable"]].tolist()

    absolute_correlations = selection_rows[usable_features].corr(method="spearman").abs()
    
    # Set B: remove features highly correlated with an already retained feature.
    correlation_filtered_features = []
    correlation_removals = []
    for feature in usable_features:
        correlated_feature = next(
            (kept_feature for kept_feature in correlation_filtered_features
            if absolute_correlations.loc[feature, kept_feature] > config.correlation_threshold),
            None,
        )
        if correlated_feature is None:
            correlation_filtered_features.append(feature)
        else:
            correlation_removals.append({
                "dropped": feature,
                "kept": correlated_feature,
                "correlation": absolute_correlations.loc[feature, correlated_feature],
            })

    # Set C: features from the market-behaviour families.
    market_features = [
        feature for feature in correlation_filtered_features
        if feature_catalog.loc[feature, "family"] in {"momentum", "risk", "activity", "returns"}
    ]
    
    # Set D: usable core features, independently of correlation filtering.
    core_features = [feature for feature in CORE if feature in usable_features]
    feature_sets = {
        "A": usable_features,
        "B": correlation_filtered_features,
        "C": market_features,
        "D": core_features,
    }
    for set_name, selected_features in feature_sets.items():
        if len(selected_features) < 2:
            raise ValueError(f"För få användbara features i set {set_name}.")

    removed_correlations = pd.DataFrame(
        correlation_removals, columns=["dropped", "kept", "correlation"]
    )
    return feature_sets, feature_catalog, removed_correlations

def make_preprocessor():
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler())])
    
def chronological_folds(development_rows, cfg):
    """Yield expanding-window folds ending in each development validation year.

    With the default configuration: 2017 -> 2018, 2017-2018 -> 2019,
    and 2017-2019 -> 2020. Every stock on a date stays in the same period.
    """
    dates = pd.to_datetime(development_rows["Date"], errors="raise")
    development_end = pd.Timestamp(cfg.development_end)
    if dates.min() < pd.Timestamp(cfg.development_start) or dates.max() > development_end:
        raise ValueError("Pass only rows from the configured development period.")
    first_validation_year = pd.Timestamp(cfg.feature_selection_end).year + 1
    for validation_year in range(first_validation_year, development_end.year + 1):
        validation_start = pd.Timestamp(validation_year, 1, 1)
        validation_end = pd.Timestamp(validation_year, 12, 31)
        training_rows = development_rows.loc[dates < validation_start]
        validation_rows = development_rows.loc[dates.between(validation_start, validation_end)]
        fold_name = f"validate_{validation_year}"
        if training_rows.empty or validation_rows.empty:
            raise ValueError(f"Missing training or validation observations for {fold_name}.")
        assert training_rows["Date"].max() < validation_rows["Date"].min()
        yield fold_name, training_rows, validation_rows


def load_development_data(root, cfg):
    """Load notebook 01's reusable Parquet dataset and verify its provenance."""
    root = Path(root)
    dataset_path = root / cfg.development_features_path
    audit_path = root / cfg.development_audit_path
    if not dataset_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Run notebook 01 to prepare the development dataset first.")
    audit = read_json(audit_path)
    expected_settings = {name: getattr(cfg, name) for name in (
        "development_start", "development_end", "min_history_days"
    )}
    if audit["preparation_config"] != expected_settings:
        raise RuntimeError("Dataset preparation settings changed. Rerun notebook 01.")
    expected_sources = {str((root / getattr(cfg, name)).resolve()) for name in (
        "project1_features_path", "project1_feature_audit_path", "stock_list_path", "pred_path"
    ) if getattr(cfg, name) is not None}
    if set(audit["source_checksums"]) != expected_sources:
        raise RuntimeError("Configured dataset sources changed. Rerun notebook 01.")
    for name, expected in audit["source_checksums"].items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise RuntimeError("A dataset source changed. Rerun notebook 01: " + name)
    if sha256(dataset_path) != audit["dataset_sha256"]:
        raise RuntimeError("The prepared dataset changed. Rerun notebook 01.")
    development_rows = pd.read_parquet(dataset_path)
    if len(development_rows) != audit["rows_after_filter"]:
        raise ValueError("The cache row count does not match its preparation audit.")
    if not development_rows.index.is_unique or {"Target", "Prediction"} & set(development_rows.columns):
        raise ValueError("Invalid development dataset keys or forbidden outcome columns.")
    if not development_rows["Date"].between(cfg.development_start, cfg.development_end).all():
        raise ValueError("The cache contains observations outside the development period.")
    return development_rows, audit
