"""Protocol checks using real Project 1 observations; no generated market data."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ProjectConfig, options
from src.data import CORE, chronological_folds, load_daily_until, load_development_data, make_feature_sets, make_preprocessor
from src.io_utils import finalize_model_selection, prepare_run, read_json, record_input_checksums, sha256, verify_lock, write_json


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = tempfile.TemporaryDirectory()
        input_dir = Path(cls.inputs.name)
        defaults = ProjectConfig()
        source = ROOT / defaults.project1_features_path
        cls.feature_audit = read_json(ROOT / defaults.project1_feature_audit_path)
        codes = pd.read_parquet(source, columns=["SecuritiesCode"], filters=[
            ("Date", "<=", pd.Timestamp(defaults.feature_selection_end))
        ])["SecuritiesCode"].drop_duplicates().head(24).tolist()
        columns = ["Date", "SecuritiesCode", "Close", "Volume", "FinancialAvailableDate"] + cls.feature_audit["numeric_features"]
        raw = pd.read_parquet(source, columns=columns, filters=[
            ("Date", "<=", pd.Timestamp(defaults.development_end)), ("SecuritiesCode", "in", codes)
        ])
        raw.to_parquet(input_dir / "real_features.parquet", index=False)
        write_json(input_dir / "features_audit.json", cls.feature_audit)
        cls.config = replace(defaults,
            project1_features_path=str(input_dir / "real_features.parquet"),
            project1_feature_audit_path=str(input_dir / "features_audit.json"),
            stock_list_path=str((ROOT / defaults.stock_list_path).resolve()))
        cls.rows, cls.audit = load_daily_until(ROOT, cls.config, cls.config.development_end)

    @classmethod
    def tearDownClass(cls):
        cls.inputs.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shutil.copytree(ROOT / "src", self.root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        self.out = prepare_run(self.root, self.config)
        record_input_checksums(self.root, self.config, self.out)
        dataset_path = self.root / self.config.development_features_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.to_parquet(dataset_path, index=True)
        audit = dict(self.audit)
        audit["preparation_config"] = {name: getattr(self.config, name) for name in (
            "development_start", "development_end", "min_history_days"
        )}
        audit["source_checksums"] = read_json(self.out / "source_checksums.json")
        audit["dataset_sha256"] = sha256(dataset_path)
        write_json(self.root / self.config.development_audit_path, audit)
        write_json(self.out / "data_audit.json", audit)

    def test_three_complete_chronological_folds(self):
        folds = list(chronological_folds(self.rows, self.config))
        self.assertEqual([name for name, _, _ in folds], ["validate_2018", "validate_2019", "validate_2020"])
        for year, (_, training, validation) in zip((2018, 2019, 2020), folds):
            self.assertLess(training.Date.max(), validation.Date.min())
            self.assertEqual(set(validation.Date.dt.year), {year})
            self.assertEqual(len(validation), int(self.rows.Date.dt.year.eq(year).sum()))
            self.assertEqual(len(training), int(self.rows.Date.dt.year.lt(year).sum()))
            self.assertTrue(set(training.Date).isdisjoint(validation.Date))
        self.assertIsNone(options(self.config)["train_cap"])
        self.assertIsNone(options(self.config)["validation_cap"])

    def test_missing_2020_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "validate_2020"):
            list(chronological_folds(self.rows.loc[self.rows.Date.dt.year.lt(2020)], self.config))

    def test_parquet_roundtrip_and_cache_guards(self):
        loaded, _ = load_development_data(self.root, self.config)
        pd.testing.assert_frame_equal(loaded, self.rows, check_exact=True)
        changed = replace(self.config, min_history_days=self.config.min_history_days + 1)
        with self.assertRaisesRegex(RuntimeError, "settings changed"):
            load_development_data(self.root, changed)
        with (self.root / self.config.development_features_path).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "dataset changed"):
            load_development_data(self.root, self.config)

    def test_preprocessing_is_fitted_on_each_training_fold(self):
        for _, training, validation in chronological_folds(self.rows, self.config):
            preprocessor = make_preprocessor()
            preprocessor.fit(training[CORE])
            imputed_training = preprocessor.named_steps["imputer"].transform(training[CORE])
            np.testing.assert_allclose(preprocessor.named_steps["scale"].mean_, imputed_training.mean(axis=0))
            self.assertEqual(len(preprocessor.transform(validation[CORE])), len(validation))

    def test_holdout_is_checked_before_reading(self):
        with patch("src.data.pd.read_parquet", side_effect=AssertionError("Holdout read before locking")):
            with self.assertRaises(RuntimeError):
                load_daily_until(self.root, self.config, self.config.test_start)
            with self.assertRaises(FileNotFoundError):
                load_daily_until(self.root, self.config, self.config.test_start, allow_holdout=True, out=self.out)

    def test_configuration_reuse_is_checked(self):
        with self.assertRaises(RuntimeError):
            prepare_run(self.root, replace(self.config, correlation_threshold=0.9))
        with self.assertRaises(ValueError):
            replace(self.config, development_end=self.config.test_start)

    def test_final_lock_requires_full_development_refit(self):
        selection = self.rows.loc[self.rows.Date.le(self.config.feature_selection_end)]
        feature_sets, catalog, removed = make_feature_sets(selection, self.feature_audit["numeric_features"], self.config)
        write_json(self.out / "feature_sets.json", feature_sets)
        catalog.to_csv(self.out / "feature_catalog.csv")
        write_json(self.out / "feature_filter_fit.json", {"end": str(selection.Date.max()), "n_rows": len(selection)})
        scores = []
        periods = []
        for name, training, validation in chronological_folds(self.rows, self.config):
            pipeline = Pipeline([("preprocess", make_preprocessor()),
                                 ("model", GaussianMixture(n_components=2, covariance_type="diag",
                                                           n_init=1, max_iter=300, random_state=42))])
            pipeline.fit(training[feature_sets["D"]])
            scores.append({"fold": name, "validation_log_likelihood": pipeline.score(validation[feature_sets["D"]])})
            periods.append({"fold": name, "training_end": training.Date.max(), "validation_end": validation.Date.max()})
        pd.DataFrame(scores).to_csv(self.out / "gmm_cv.csv", index=False)
        pd.DataFrame(periods).to_csv(self.out / "folds.csv", index=False)
        model_id = "D_protocol_check"
        pd.DataFrame([{"candidate_id": model_id, "feature_set": "D",
                       "mean_validation_log_likelihood": pd.DataFrame(scores).validation_log_likelihood.mean()}]
                    ).to_csv(self.out / "gmm_candidates.csv", index=False)
        write_json(self.out / "candidate_ids.json", {"D": model_id})
        pipeline.fit(self.rows[feature_sets["D"]])
        joblib.dump(pipeline, self.out / "models" / f"{model_id}.joblib")
        fit_record = {"selected_model_id": model_id, "feature_set": "D", "features": feature_sets["D"],
                      "training_start": self.config.development_start, "training_end": "2019-12-31", "n_rows": len(self.rows)}
        write_json(self.out / "final_model_fit.json", fit_record)
        with self.assertRaisesRegex(ValueError, "all development rows"):
            finalize_model_selection(self.root, self.out, self.config, model_id, "Protocol verification using actual JPX observations.")
        fit_record["training_end"] = self.config.development_end
        write_json(self.out / "final_model_fit.json", fit_record)
        finalize_model_selection(self.root, self.out, self.config, model_id, "Protocol verification using actual JPX observations.")
        self.assertEqual(verify_lock(self.out, self.config)["selected_model_id"], model_id)
        with (self.root / "src/data.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed after locking\n")
        with self.assertRaisesRegex(RuntimeError, "locked file changed"):
            verify_lock(self.out, self.config)


if __name__ == "__main__":
    unittest.main()
