"""Regression checks against actual split and reverse-split observations in Project 1."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import add_price_features, assert_causal_feature_stability


class PriceAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = pd.read_parquet(ROOT / "data/interim/stock_prices_clean.parquet", filters=[
            ("SecuritiesCode", "in", ["1414", "7638"]),
            ("Date", ">=", pd.Timestamp("2019-01-01")),
            ("Date", "<=", pd.Timestamp("2019-12-31")),
        ])
        cls.features = add_price_features(cls.prices).set_index(["Date", "SecuritiesCode"])

    def test_reverse_split_uses_factor_on_next_observation(self):
        before = self.features.loc[(pd.Timestamp("2019-09-26"), "7638")]
        after = self.features.loc[(pd.Timestamp("2019-09-27"), "7638")]
        self.assertAlmostEqual(before.Return1Day, 43 / 44 - 1)
        self.assertAlmostEqual(after.Return1Day, 907 / (43 * 20) - 1)
        self.assertAlmostEqual(after.LogReturn1Day, np.log(907 / (43 * 20)))
        self.assertAlmostEqual(after.GapReturn, 848 / (43 * 20) - 1)

    def test_split_uses_factor_on_next_observation(self):
        before = self.features.loc[(pd.Timestamp("2019-06-25"), "1414")]
        after = self.features.loc[(pd.Timestamp("2019-06-26"), "1414")]
        self.assertAlmostEqual(before.Return1Day, 7650 / 7710 - 1)
        self.assertAlmostEqual(after.Return1Day, 3700 / (7650 * .5) - 1)
        self.assertAlmostEqual(after.GapReturn, 3750 / (7650 * .5) - 1)

    def test_ordinary_day_remains_unadjusted(self):
        row = self.features.loc[(pd.Timestamp("2019-09-24"), "7638")]
        self.assertAlmostEqual(row.Return1Day, 44 / 43 - 1)

    def test_saved_feature_table_matches_corrected_returns(self):
        columns = ["Date", "SecuritiesCode", "Return1Day", "LogReturn1Day", "GapReturn"]
        stored = pd.read_parquet(ROOT / "data/processed/features_table.parquet", columns=columns, filters=[
            ("SecuritiesCode", "in", ["1414", "7638"]),
            ("Date", "in", pd.to_datetime(["2019-06-25", "2019-06-26", "2019-09-26", "2019-09-27"]).tolist()),
        ]).set_index(["Date", "SecuritiesCode"]).sort_index()
        pd.testing.assert_frame_equal(stored, self.features.loc[stored.index, columns[2:]])

    def test_future_rows_do_not_change_existing_features(self):
        assert_causal_feature_stability(self.prices, pd.Timestamp("2019-09-26"))
