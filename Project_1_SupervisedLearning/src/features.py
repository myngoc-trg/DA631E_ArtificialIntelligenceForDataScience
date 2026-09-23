from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Value

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

BASE_NUMERIC_FEATURES = [
    "OpenToCloseReturn",
    "HighLowRange",
    "GapReturn",
    "LogReturn1D",
    "Momentum5D",
    "Momentum20D",
    "Momentum60D",
    "Volatility20D",
    "Volatility60D",
    "VolatilityRatio20To60",
    "CloseToMovingAverage20D",
    "LogVolume",
    "VolumeVs20D",
    "LogTurnover",
    "ExpectedDividend",
    "DaysSinceLastTrade",
    "NoTradeFlag",
    "MarketWideNoTradeFlag",
    "StockSpecificNoTradeFlag",
    "PartialOHLCFlag",
    "HasExpectedDividend",
    "SupervisionFlag",
    "AdjustmentEventFlag",
    "Return1DRankPct",
    "Momentum20DRankPct",
    "Volatility20DRankPct",
    "IssuedSharesLog",
    "MarketCapitalizationLog",
    "Universe0",
    "FinancialAgeDays",
]

CATEGORICAL_FEATURES = [
    "33SectorCode",
    "17SectorCode",
    "NewMarketSegment",
    "NewIndexSeriesSizeCode",
    "Fin_DocumentType",
    "Fin_PeriodType",
    "Month",
    "DayOfWeek",
]

NON_MODEL_COLUMNS = {
    "RowId",
    "Date",
    "SecuritiesCode",
    "Target",
    "Open",
    "High",
    "Low",
    "Close",
    "CloseForFeatures",
    "LastTradeDate",
    "FinancialDisclosedDate",
    "FinancialAvailableDate",
}

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator.ne(0))
    return result.replace([-np.inf, np.inf], np.nan)


def add_price_features(
    prices: pd.DataFrame
    ,windows: Sequence[int] = (5,20,60)
) -> pd.DataFrame:
    """
    Create features using information available NO LATER THAN each row's Date
    
    The split adjustment is applied to the previous close when calculating the current one-day return.
    Unlike a reverse cumulative adjusted price, appending a future split cannot rewrite historical feature value
    """
    
    required = {
        'Date'
        ,'SecuritiesCode'
        ,'Open'
        ,'High'
        ,'Low'
        ,'Close'
        ,'CloseForFeatures'
        ,'Volume'
        ,'AdjustmentFactor'
        ,'Target'
        }
    
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"Price data is missing columns: {missing}")
    
    df = prices.copy().sort_values(['SecuritiesCode', 'Date']).reset_index(drop=True)
    by_code = df.groupby('SecuritiesCode', sort=False)
    previous_close = by_code['CloseForFeatures'].shift(1)
    comparable_previous_close = previous_close * df['AdjustmentFactor'].fillna(1.0)
    
    return_raw = _safe_ratio(df['CloseForFeatures'], comparable_previous_close) - 1.0
    df['Return1Day'] = return_raw.mask(df['NotradeFlag'].eq(1), 0.0)
    df['LogReturn1Day'] = np.log1p(df['Return1Day'].where(df['Return1Day'] > -1.0))
    
    df['OpenToCloseReturn'] = _safe_ratio(df['Close'], df['Open']) - 1.0
    df['HighLowRange'] = _safe_ratio(df['High'], df['Low']) - 1.0
    
    #Overnight movement between yesterday's close and today's open
    df['GapReturn'] = _safe_ratio(df['Open'], comparable_previous_close) 
    
    #A causal relative price index supports a moving-average feature without using adjustment factors from future rows
    cummulative_log_return = df['LogReturn1Day'].fillna(0.0).groupby(df['SecuritiesCode'], sort=False).cumsum()
    df['CausalPriceIndex'] = np.exp(cummulative_log_return.clip(-20,20)) #clip(-20,20) ie. results should be in range e^-20, e^20 to avoid numerical overflow
    #This index is causal: adding a future stock split does not change previously calculated values
    #show how an investment of 1 unit stock would have grown since the beginning of the available history, after accoundting for stock split
    
    grouped_log_return = df.groupby('SecuritiesCode', sort=False)['LogReturn1Day']
    for window in windows:
        min_periods = max(2, int(np.ceil(window * 0.75))) 
        rolling_log_return = grouped_log_return.transform(
            lambda values, w=window, ,m=min_periods: values.rolling(window=w, min_periods=m).sum())
        df[f'Momentum{window}Days'] = np.expm1(rolling_log_return)
        
    
    for window in (20,60):
        min_periods = int(np.ceil(window * 0.75))
        df[f'Volatility{window}Days'] = grouped_log_return.transform()
        
    return