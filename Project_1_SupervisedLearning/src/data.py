from __future__ import annotations
from ast import Tuple
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

STOCK_CATEGORY_CANDIDATES = [
    "33SectorCode",
    "17SectorCode",
    "NewMarketSegment",
    "NewIndexSeriesSizeCode",
]
MISSING_TOKENS = ['','-','nan','None','N/A']

FINANCIAL_NUMERIC_CANDIDATES = [
    "NetSales",
    "OperatingProfit",
    "OrdinaryProfit",
    "Profit",
    "EarningsPerShare",
    "TotalAssets",
    "Equity",
    "EquityToAssetRatio",
    "BookValuePerShare",
    "ResultDividendPerShareAnnual",
    "ForecastDividendPerShareAnnual",
    "ForecastNetSales",
    "ForecastOperatingProfit",
    "ForecastOrdinaryProfit",
    "ForecastProfit",
    "ForecastEarningsPerShare",
]

def find_project_root(
        start: Path = Path.cwd()
        ,project_name: str = "Project_SupervisedLearning_1"
) -> Path:
    """
    Find a project directory containing both data/ and notebooks/ folders.

    Searches:
    1. current dir
    2. current dir / Project_1
    3. parent dirs
    """

    start = start.resolve()

    candidates = [
        start
        ,start / project_name
        ,*start.parents
    ]

    #print("Candidates to search: ", candidates)

    checked = set() #unique set to store checked candidates

    for candidate in candidates:
        candidate = candidate.resolve()
        #print("Check candidate: ", candidate)
        if candidate in checked:
            #print("Already checked! Skip to next candidate")
            continue

        checked.add(candidate)

        if (candidate / "data").is_dir() and (candidate / "notebooks").is_dir():
            #print("Found ", candidate)
            return candidate

    raise FileNotFoundError(
        f"Could not find {project_name!r} containing data/ and notebooks/."
        "Set PROJECT_ROOT manually using Path()"
    )

def find_unique_file(directory: Path, filename: str) -> Path:
    matches = [path for path in directory.rglob(filename)
                      if path.is_file()]
    print("Matched files: ",matches)
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {directory}")

    if len(matches) > 1:
        formatted = "\n".join(str(path) for path in matches)
        print("Different formats of dup matches: ", formatted)
        raise RuntimeError(
            f"Found multiple copies of {filename!r}:\n{formatted}"
        )

    return matches[0]

PROJECT_ROOT = find_project_root()
print(PROJECT_ROOT)

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

print("Train Data directory: ", TRAIN_DIR)

def normalize_security_code(series, table_name):
    values = (series.astype("string").str.strip())


def load_data(train_dir = TRAIN_DIR):
    PRICE_PATH = find_unique_file(directory=train_dir, filename="stock_prices.csv")
    STOCK_LIST_PATH = find_unique_file(train_dir, "stock_list.csv")
    FINANCIALS_PATH = find_unique_file(train_dir, "financials.csv")

    prices = pd.read_csv(PRICE_PATH, dtype={'SecuritiesCode':'string'} , low_memory=False)
    stock_list = pd.read_csv(STOCK_LIST_PATH, dtype={'SecuritiesCode':'string'}, low_memory=False)
    financials = pd.read_csv(FINANCIALS_PATH, dtype={'SecuritiesCode':'string'}, low_memory=False)

    #standardize main identifier (primary key)
    for name, frame in {
        "prices": prices
        ,"stock_list": stock_list
        ,"financials": financials
    }.items():
        if "SecuritiesCode" not in frame.columns:
            raise ValueError(f"{name} has no SecuritiesCode column")
        
        frame['SecuritiesCode'] =frame['SecuritiesCode'].str.strip()
        
        codes = (frame['SecuritiesCode'].astype("string").str.strip().str.replace(r"\.0+$","",regex=True))
        missing_count = frame['SecuritiesCode'].isna().sum()
        
      
        print(f"{name}: {missing_count} rows with missing SecuritiesCode")
        frame["SecuritiesCode"] = codes
        
        '''
        if frame['SecuritiesCode'].isna().any():
                    raise ValueError(f"{name}.SecuritiesCode contains missing values.")
        invalid_codes = ~frame['SecuritiesCode'].str.fullmatch(rf"\d{4}")
        
        if invalid_codes.any():
            raise ValueError(f"{name} contains invalid SecuritiesCode values (not 4 digit code).") 
        '''
     
    
    prices['Date'] = pd.to_datetime(prices['Date'], errors='raise')
    
    for column in ['Date'
                   ,'DisclosedDate'
                   ,'CurrentPeriodEndDate'
                   ,'CurrentFiscalYearStartDate'
                   ,'CurrentFiscalYearEndDate']:
        if column in financials.columns:
            financials[column] = pd.to_datetime(financials[column], errors='coerce')
        
    for column in ['EffectiveDate'
                ,'TradeDate']:
        if column in stock_list.columns:
            stock_list[column] = pd.to_datetime(stock_list[column], errors='coerce')
        
    prices = prices.sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    return prices, stock_list, financials      

def clean_prices(
    df: pd.DataFrame
    ,market_no_trade_thresholde: float = 0.90 #to exclude those data rows > threshold    
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors="raise")

    numeric_cols = ['Open', 'High', 'Low', 'Close','Volume', 'AdjustmentFactor','Target']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    audit = {
        "input_rows": len(df)
        ,"duplicates_stock_dates": int(df.duplicated(['Date','SecuritiesCode']).sum())
        ,"missing_close": int(df['Close'].isna().sum())
        ,"missing_volume": int(df['Volume'].isna().sum())
        ,"negative_volume": int((df['Volume'] < 0).sum())
             }

    if audit['duplicates_stock_dates']:
        duplicates = df.loc[df.duplicated(['Date','SecuritiesCode'],keep=False)].sort_values(by=['Date','SecuritiesCode'])
        raise ValueError(
            "Duplicate Date-SecuritiesCode records found. "
            "Investigate before continuing"
        )
    
    ohlc =  ['Open', 'High', 'Low', 'Close']
    complete_ohlc = df[ohlc].notna().all(axis=1)
    invalid_ohlc = complete_ohlc & (
        (df['High'] < df['Low'])
        | (df['High'] < df['Open'])
        | (df['High'] < df['Close'])
        | (df['Low'] > df['Open'])
        | (df['Low'] > df['Close'])
        | (df['Close'] <= 0)
        | (df['Open'] <= 0)
    )

    audit['invalid_ohcl'] = int(invalid_ohlc.sum())

    if invalid_ohlc.any():
        raise ValueError(f"OHLC inconsistent")

    df['PartialOHLCFlag'] = (df[ohlc].isna().any(axis=1) & ~df[ohlc].isna().all(axis=1)).astype('int8')
    
    all_ohlc_missing = df[ohlc].isna().all(axis=1)
    df['NoTradeFlag'] = (all_ohlc_missing & df['Volume'].eq(0)).astype("int8")
    audit['no_trade_rows'] = int(df['NoTradeFlag'].sum())
    audit['partial_ohlc_rows'] = int(df['PartialOHLCFlag'].sum())
    
    invalid_volume = df['Volume'].isna() | df['Volume'].lt(0)
    audit['invalid_volume_rows'] = int(invalid_volume.sum())
    df = df.loc[~invalid_volume].copy()
    
    no_trade_rate = df.groupby(by='Date')['NoTradeFlag'].transform('mean')
    df['MarketWideNoTradeFlag'] = no_trade_rate.ge(market_no_trade_thresholde).astype('int8')
    df['StockSpecificNoTradeFlag'] = (df['NoTradeFlag'].eq(1) & df['MarketWideNoTradeFlag'].eq(0)).astype('int8')
    
    missing_adjustment = df['AdjustmentFactor'].isna()
    audit['missing_adjustment'] = int(missing_adjustment.sum())
    print("Fill missing adjustment with 1.0 (unchanged stock split)")
    df.loc[missing_adjustment, 'AdjustmentFactor'] = 1.0
    if (df['AdjustmentFactor'] <= 0).any():
        raise ValueError("AdjustmentFactor contains non-positive values.")
    df['AdjustmentEventFlag'] = df['AdjustmentFactor'].ne(1).astype('int')
    
    df['HasExpectedDividend'] = df['ExpectedDividend'].notna().astype('int')
    df['ExpectedDividend'] = df['ExpectedDividend'].fillna(value=0.0)
    df['SupervisionFlag'] = df['SupervisionFlag'].fillna(False).astype('int8')

    df = df.sort_values(['SecuritiesCode','Date']).reset_index(drop=True)
    
    df['LastTradeDate'] = df['Date'].where(df['Close'].notna())
    df['LastTradeDate'] = df.groupby('SecuritiesCode',sort=False)['LastTradeDate'].ffill()
    df['DaysSinceLastTrade'] = (df['Date'] - df['LastTradeDate']).dt.days
    
    df['CloseForFeatures'] = df.groupby(by='SecuritiesCode',sort=False)['Close'].ffill()

    audit['output_rows'] = len(df)
    audit['removed_rows'] = audit['output_rows'] - audit['input_rows']

    return df, audit


def clean_stock_list(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return 1 metadata row per securities code and exclude stale price-like fields"""
    
    df = df.copy()
    duplicate_codes = int(df.duplicated(['SecuritiesCode']).sum())
    if duplicate_codes:
        raise ValueError(f"Stock list has multiple rows for a securities code")
    
    selected = ['SecuritiesCode']
    selected += [col for col in STOCK_CATEGORY_CANDIDATES if col in df]
    selected += [col for col in ("IssuedShares","MarketCapitalization","Universe0","EffectiveDate","TradeDate") if col in df]
    
    metadata = df[selected].copy()
    
    for col in STOCK_CATEGORY_CANDIDATES:
        if col in metadata.columns:
            metadata[col] = metadata[col].astype("string").str.strip().replace(MISSING_TOKENS,pd.NA).fillna("__MISSING__")
    
    for col in ("IssuedShares","MarketCapitalization"):
        if col in metadata:
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
            metadata[f"{col}Log"] = np.log1p(metadata[col].clip(lower=0)) #clip(lower=0) turns all values less than 0 to 0, log1p(x) = log(1+x)
    
    for col in ("EffectiveDate","TradeDate"):
        if col in metadata:
            numeric = pd.to_numeric(metadata[col], errors="coerce").astype("Int64")
            
            parsed = pd.to_datetime(
                numeric.astype("string"),
                format="%Y%m%d",
                errors="coerce",
            )
        
            invalid = metadata[col].notna() & parsed.isna()
        
            if invalid.any():
                examples = metadata[col].loc[invalid].head().tolist()
                raise ValueError(
                    f"{col} contains invalid dates. "
                    f"Examples: {examples}"
                )
            metadata[col] = parsed
                
    audit = {
        'input_rows': len(df)
        ,'output_rows': len(metadata)
        ,'duplicate_security_code': duplicate_codes
        ,'selected_columns': len(metadata.columns)
    }
    
    return metadata, audit


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator.ne(0))
    return result.replace([np.inf, -np.inf], np.nan)


def _optional_ratio(df:pd.DataFrame, numerator: str, denominator:str) -> pd.Series:
    if numerator not in df or denominator not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return _safe_divide(df[numerator], df[denominator])


def clean_financials(df: pd.DataFrame, availability_lag_days:int =1) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build fnancials with disclosure-event features that can be joined backward in time
    
    A one day lag is the conservative defaut: a report disclosed on day t firs becomes eligible for a price row (Closed) on t+1
    Change this if necessary
    """
    
    df = df.copy()
    input_rows = len(df)
    
    df = df.dropna(subset=['SecuritiesCode', 'DisclosedDate']).copy()
    numeric_cols = [col for col in FINANCIAL_NUMERIC_CANDIDATES if col in df]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    sort_cols = ['SecuritiesCode', 'DisclosedDate']
    for tie_breaker in ('DisclosedUnixTime','DisclosureNumber'):
        if tie_breaker in df:
            sort_cols.append(tie_breaker)
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    #a revision often updates only a subset of fields. Carry earlier disclosed values
    # forward within the same security; never backward-fill future information
    
    if numeric_cols:
        df[numeric_cols] = df.groupby('SecuritiesCode', sort=False)[numeric_cols].ffill()
        
    df['Fin_OperatingMargin'] = _optional_ratio(df, 'OperatingProfit', 'NetSales')
    df['Fin_OrdinaryMargin'] = _optional_ratio(df, 'OrdinaryProfit', 'NetSales')
    df['Fin_ProfitMargin'] = _optional_ratio(df, 'Profit', 'NetSales')
    df['Fin_ReturnOnAssets'] = _optional_ratio(df, 'Profit', 'TotalAssets')
    df['Fin_ReturnOnAssets'] = _optional_ratio(df, 'Equity', 'TotalAssets')
    
    for col in numeric_cols:
        values = df[col]
        df[f'Fin_{col}_SignedLog'] = np.sign(values) * np.log1p(values.abs())
        
    df['FinancialAvailableDate'] = df['DisclosedDate'] + pd.to_timedelta(availability_lag_days, unit="D")
    
    if 'TypeOfDocument' in df:
        df['TypeOfDocument'] = df['TypeOfDocument'].astype("string").fillna("__MISSING__")
        
    if 'TypeOfCurrentPeriod' in df:
        df['TypeOfCurrentPeriod'] = df['TypeOfCurrentPeriod'].astype("string").fillna("__MISSING__")
        
    output_cols = ['SecuritiesCode', 'DisclosedDate', 'FinancialAvailableDate']

    output_cols += [col for col in df.columns if col.startswith("Fin_")]

    df = df[output_cols].copy()
    
    df = df.drop_duplicates(['SecuritiesCode', 'FinancialAvailableDate'], keep="last")
    df = df.sort_values(['FinancialAvailableDate', 'SecuritiesCode']).reset_index(drop=True)

    audit = pd.Series({
        'input_rows': input_rows,
        'rows_with_valid_key_and_date': len(df),
        'output_rows': len(df),
        'numeric_source_columns': len(numeric_cols),
        'availability_lag_days': availability_lag_days,
    })

    return df, audit