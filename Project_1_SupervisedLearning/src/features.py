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
    "DisclosedDate",
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
        
    
    for window in (20,60)
        min_periods = int(np.ceil(window * 0.75))
        df[f'Volatility{window}Days'] = grouped_log_return.transform(lambda values: values.rolling(window=window, min_periods=min_periods).std())
        
    df['VolatilityRatio20To60'] = _safe_ratio(df['Volatilitywindow20Days'], df['Volatility60Days'])
    
    moving_average_20 = df.groupby(df['SecuritiesCode'], sort=False)['CausalPriceIndex'].transform(
        lambda values: values.rolling(20,min_periods=15).mean())
    
    df['CloseToMovingAverage20Days'] = _safe_ratio(df['CausalPriceIndex'], moving_average_20) -1
    
    df['LogVolume'] = np.log1p(df['Volume'].clip(0))
    log_volume_avg_20 = df.groupby(df['SecuritiesCode'], sort=False)['LogVolume'].transform(
        lambda values: values.rolling(window=20, min_periods=10).mean()
    )
    df['VolumeVs20Days'] = df['LogVolume'] - log_volume_avg_20
    
    df['LogTurnOver'] = np.log1p((df['CloseForFeatures'] * df['Volume']).clip(0))
    
    #Same day cross-sectional ranks are available at the prediction time stamp 
    for source, output in (
        ('Return1Day', 'Return1DayRankPct')
        ,('Momentum20Days', 'Momentum20DaysRankPct')
        ,('Volatility20Days', 'Volatility20DaysRankPct') 
    ):
        df['output'] = df.groupby('¨Date', sort=False)['source'].rank(pct=True) #percentile rank across share on the same day
        
    df['Month'] = df['Date'].dt.month.astype('string')
    df['DaysOfWeek'] = df['Date'].dt.day_of_week.astype('string')
    
    df = df.replace([np.inf, -np.inf],np.nan)
    
    return df.sort_values(['Date', 'SecuritiesCode']).reset_index(drop=True)


def join_stock_metadata(prices_features: pd.DataFrame, stock_meta: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(prices_features)
    result = prices_features.merge(stock_meta, on='SecuritiesCode', validate='many_to_one')
    rows_after = len(result)
    if rows_after != rows_before:
        print("Length of prices dataframe before join with stock_list metadata: ", rows_before)
        print("Length after join: ", rows_after)
        raise AssertionError("Metadata join changed the price-table rows")
    return result


def join_financials_asof(prices_features: pd.DataFrame, financial_disclosure: pd.DataFrame) -> pd.DataFrame:
    """ 
    For each price-row on Date, join with the most recent financial disclosure 
    that was already available on or before that date
    
    merge_asof() = 'join each row with the nearest relevant observation in time,
                    usually the latest one available up to that point'
    """
    
    if financial_disclosure.empty:
        print("DataFrame financials is empty! Return prices DataFrame without join.")
        result = prices_features.copy()
        result['FinancialAgeDays'] = np.nan
        return result
    
    rows_before = len(prices_features)
    left = prices_features.copy()
    left['_OriginalOrderDate'] = np.arange(len(left))
    left = left.sort_values(['Date', 'SecuritiesCode'])
    right = financial_disclosure.sort_values(['FinancialAvailableDate','SecuritiesCode'])
    
    result = pd.merge_asof(left=left, right=right,
                           left_on='Date'
                           ,right_on='FinancialAvailableDate'
                           ,by='SecuritiesCode'
                           ,direction='backward'
                           ,allow_exact_matches=True)
    
    rows_after = len(result)
    if rows_after != rows_before:
        print("Length of prices dataframe before join with financials disclosure: ", rows_before)
        print("Length after join: ", rows_after)
        raise AssertionError("Financials join changed the price-table rows")

    future_report = result['FinancialAvailableDate'].gt(result['Date'])
    if future_report.fillna(False).any():
        result.loc[future_report, ['FinancialAvailableDate', 'Date']].head(20)
        raise AssertionError("Future financial disclosure leaked into a price row")
    
    result['FinancialAgeDays'] = (result['Date'] - result['DisclosedDate']).dt.days
    
    return result.sort_values('_OriginalOrder').drop(columns='_OriginalOrder').reset_index(drop=True)
    
    


def infer_feature_column(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """ 
    Create an explicit feature contract; to avoid using every numeric columns blindly
    
    Exactly which columns are allowed to enter the model as features?
    """
    
    numeric = [col for col in BASE_NUMERIC_FEATURES if col in df]
    numeric += [
        col
        for col in df
        if isinstance(col, str)
        and col.startswith("Fin_")
        and col not in CATEGORICAL_FEATURES
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    categorical = [col for col in CATEGORICAL_FEATURES if col in df]
    
    forbidden = (set(numeric) | set(categorical)).intersection(NON_MODEL_COLUMNS)
    if forbidden:
        raise AssertionError("Leakage/key columns entered the feature list: ", sorted(forbidden))
    
    if not numeric:
        raise ValueError("No numeric model features were found.")
    
    return numeric, categorical     


def assert_causal_feature_stability(clean_prices: pd.DataFrame
                                    ,cutoff: pd.Timestamp
                                    ,columns: Sequence[str] =(
                                        'LogReturn1Day'
                                        ,'Momentum20Days'
                                        ,'Volatility20Days'
                                        ,'CloseToMovingAverage20Days'
                                    )
                                    ) -> None:
    
    """
    Verify that appending future rows does not rewrite historical features
    
    If I add future rows to my dataset, do any feature values in the past change?
    """
    
    # Pretend the future does not exist. If throw away everything after cutoff
    historical_raw = clean_prices.loc[clean_prices['Date'].le(cutoff)].copy()
    # Calculate features using only information available until cutoff
    historical_features = add_price_features(historical_raw)
    
    # Now give the feature function the entire future (whole dataset of prices)
    full_features = add_price_features(clean_prices)
    # Calculate the same added features using the enitre dataset. Then throw away everything after cutoff
    full_historical = full_features.loc[full_features['Date'].le(cutoff)].copy()
    
    
    # If the features are CAUSAL, then A == B 
    
    keys = ['Date', 'SecuritiesCode']
    expected = historical_features[keys + list(columns)].sort_values(keys).reset_index(drop=True)
    actual = full_historical[keys + list(columns)].sort_values(keys).reset_index(drop=True)
    assert_frame_equal(left=expected, right=actual, check_dtype=False, rtol=1e-10, atol=1e-12)
    
    

def feature_missingness(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                'missing_count': df[list(columns)].isna().sum()
                ,'missing_rate': df[list(columns)].isna().mean()
                ,'dtype': df[list(columns)].dtype.astype('string')
            }
        ).sort_values('missing_rate', ascending=False)
    )