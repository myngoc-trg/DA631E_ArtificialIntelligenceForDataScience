"""Diagnostic contracts checked with small samples of real development rows."""
from pathlib import Path
from dataclasses import replace
import ast
import json
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ProjectConfig
from src.modeling import (
    assignment_stability, check_development_rows, compare_validation_coverage
    ,covariance_diagnostics, describe_assignments, feature_shift_report
    ,fit_candidates, likelihood_tail_diagnostics, refit_recorded_fold
    ,run_remedy_cv, save_candidate_bundle, predict_states
    ,run_gmm_cv, summarize_cv
)


class GMMDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ProjectConfig()
        cls.features = [
            "LogReturn1Day", "Momentum5Days", "Volatility20Days"
            ,"HasExpectedDividend", "SupervisionFlag", "AdjustmentEventFlag"
            ,"Fin_OperatingMargin"
        ]
        columns = ["Date", "SecuritiesCode", "DayNumber"] + cls.features
        periods = []
        for year in (2017, 2018, 2019, 2020):
            period = pd.read_parquet(
                ROOT / cls.cfg.development_features_path
                ,columns=columns
                ,filters=[
                    ("Date", ">=", pd.Timestamp(year, 7, 3))
                    ,("Date", "<=", pd.Timestamp(year, 7, 7))
                ]
            )
            if period.empty:
                raise AssertionError(f"No real fixture observations for {year}.")
            periods.append(period.sample(n=min(500, len(period)), random_state=42).sort_index())
        cls.rows = pd.concat(periods).sort_index()
        cls.training_rows = cls.rows.loc[cls.rows.Date.dt.year.eq(2017)]
        cls.anchor_rows = cls.rows.loc[cls.rows.Date.dt.year.eq(2018)]
        cls.feature_sets = {"D": cls.features}
        cls.specifications = [{"feature_set": "D", "k": 2, "covariance": "full"}]
        cls.fit_options = {"train_cap": 300, "n_init": 1, "max_iter": 200, "reg_covar": 1e-4}
        cls.artifacts = fit_candidates(
            cls.training_rows
            ,cls.feature_sets
            ,None
            ,cls.cfg
            ,specifications=cls.specifications
            ,fit_options=cls.fit_options
            ,model_seed=24
            ,compute_metrics=False
            ,require_convergence=False
        )[0]
        cls.artifact = next(iter(cls.artifacts.values()))
        cls.scores = predict_states(
            cls.artifact
            ,cls.anchor_rows
            ,cls.cfg
            ,batch_size=73
            ,include_diagnostics=True
        )

    def test_holdout_rejected_before_scoring(self):
        invalid_rows = self.anchor_rows.iloc[:1].copy()
        invalid_rows["Date"] = pd.Timestamp("2021-01-04")
        with self.assertRaisesRegex(ValueError, "exclude 2021"):
            check_development_rows(invalid_rows, self.cfg)
        with self.assertRaises(ValueError):
            predict_states(
                self.artifact
                ,invalid_rows
                ,self.cfg
                ,include_diagnostics=True
            )

    def test_batched_scores_match_direct_model_and_preserve_keys(self):
        matrix = self.artifact["preprocessor"].transform(self.anchor_rows[self.features])
        model = self.artifact["model"]
        self.assertTrue(self.scores.index.equals(self.anchor_rows.index))
        np.testing.assert_allclose(self.scores.log_likelihood, model.score_samples(matrix))
        np.testing.assert_array_equal(self.scores.State, model.predict(matrix))
        self.assertTrue(self.scores.normalized_entropy.between(-1e-12, 1 + 1e-12).all())
        self.assertAlmostEqual(self.scores.log_likelihood.mean(), model.score(matrix))

    def test_legacy_prediction_columns_are_preserved(self):
        predictions = predict_states(self.artifact, self.anchor_rows)
        self.assertEqual(
            list(predictions.columns)
            ,["Date", "SecuritiesCode", "DayNumber", "State", "StateProbability"]
        )
        np.testing.assert_array_equal(predictions.State, self.scores.State)
        np.testing.assert_allclose(predictions.StateProbability, self.scores.StateProbability)

    def test_batched_cv_matches_original_scoring_and_checkpoint(self):
        search_options = {
            **self.fit_options, "k": [2], "covariance": ["full"]
            ,"random_seeds": [24], "validation_cap": None
        }
        direct = run_gmm_cv(
            self.rows, self.feature_sets, self.cfg, search_options=search_options
        )
        callback_records = []
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "cv.csv"
            batched = run_gmm_cv(
                self.rows, self.feature_sets, self.cfg
                ,search_options={**search_options, "score_batch_size": 73}
                ,on_result=callback_records.append, checkpoint_path=path
            )
            self.assertEqual(len(pd.read_csv(path)), 3)
        self.assertEqual(len(callback_records), 3)
        np.testing.assert_allclose(direct.valid_log_likelihood, batched.valid_log_likelihood)
        np.testing.assert_allclose(direct.train_bic, batched.train_bic)

    def test_original_four_argument_candidate_selection_still_works(self):
        cv = run_gmm_cv(
            self.rows, self.feature_sets, self.cfg
            ,search_options={
                **self.fit_options, "k": [2], "covariance": ["full"]
                ,"random_seeds": [24], "validation_cap": 200
            }
        )
        quick_cfg = replace(self.cfg, run_profile="quick")
        artifacts, table, ids = fit_candidates(
            self.training_rows, self.feature_sets, cv, quick_cfg
        )
        self.assertEqual(ids, {"D": "D_k2_full"})
        self.assertEqual(len(artifacts), 1)
        self.assertIn("silhouette", table.columns)
        self.assertAlmostEqual(table.iloc[0].cv_ll_within_set, summarize_cv(cv).iloc[0].mean_valid_ll)

    def test_profiles_and_tail_audit_handle_overlapping_flag_columns(self):
        profiles = describe_assignments(self.artifact, self.anchor_rows, self.scores, self.cfg)
        self.assertEqual(profiles["sizes"]["count"].sum(), len(self.anchor_rows))
        self.assertAlmostEqual(profiles["sizes"]["share"].sum(), 1)
        tail = likelihood_tail_diagnostics(
            self.artifact, self.anchor_rows, self.scores, tail_fraction=0.01, worst_count=7
        )
        self.assertEqual(int(tail["audit"].in_worst_tail.sum()), 5)
        self.assertEqual(len(tail["worst_rows"]), 7)
        self.assertTrue(set(self.features).issubset(tail["worst_rows"].columns))
        self.assertTrue(tail["worst_rows"].columns.is_unique)
        self.assertAlmostEqual(
            tail["tail_summary"].contribution_to_full_mean.sum()
            ,self.scores.log_likelihood.mean()
        )
        self.assertIn("any_financial_missing", tail["group_reports"])

    def test_coverage_subsets_reconstruct_full_score_without_refitting(self):
        coverage, keys = compare_validation_coverage(self.anchor_rows, self.scores, cap=100)
        self.assertEqual(len(keys), 100)
        self.assertTrue(keys.isin(self.anchor_rows.index).all())
        parts = coverage.loc[coverage.validation_subset.ne("complete_year")]
        weighted = (parts["count"] * parts.mean_ll).sum() / parts["count"].sum()
        self.assertAlmostEqual(weighted, self.scores.log_likelihood.mean())

    def test_covariance_and_shift_reports_match_feature_space(self):
        covariance = covariance_diagnostics(self.artifact)
        self.assertTrue((covariance.minimum_eigenvalue > 0).all())
        self.assertAlmostEqual(covariance.weight.sum(), 1)
        sample = self.training_rows.loc[self.artifact["sample_keys"]]
        shifts = feature_shift_report(self.artifact, sample, self.anchor_rows)
        self.assertEqual(set(shifts.index), set(self.features))

    def test_reference_stability_and_same_seed_reproduction(self):
        duplicate = fit_candidates(
            self.training_rows
            ,self.feature_sets
            ,None
            ,self.cfg
            ,specifications=self.specifications
            ,fit_options=self.fit_options
            ,model_seed=24
            ,compute_metrics=False
            ,require_convergence=False
        )[0]
        duplicate_scores = predict_states(
            next(iter(duplicate.values()))
            ,self.anchor_rows
            ,self.cfg
            ,include_diagnostics=True
        )
        np.testing.assert_allclose(duplicate_scores.log_likelihood, self.scores.log_likelihood)
        stability = assignment_stability(
            self.training_rows, self.anchor_rows, self.feature_sets, self.specifications
            ,self.artifacts, self.cfg, self.fit_options, model_seeds=(4,), sampling_seeds=(142,)
        )
        self.assertEqual(len(stability), 2)
        self.assertEqual(set(stability.sensitivity), {"initialization", "training_sample"})
        self.assertTrue(stability.ari_to_reference.between(-1, 1).all())

    def test_remedies_use_all_folds_and_pass_options_to_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            result = run_remedy_cv(
                self.rows, self.features, self.cfg, self.fit_options
                ,{"baseline": {}, "regularized": {"fit_options": {"reg_covar": 0.001}}}
                ,model_seeds=(24,), k=2, checkpoint_path=Path(directory) / "cv.csv"
            )
            self.assertEqual(len(result), 6)
            self.assertEqual(result.valid_ll.notna().sum(), 6)
            self.assertEqual(len(pd.read_csv(Path(directory) / "cv.csv")), 6)
            self.assertTrue((pd.to_datetime(result.train_end) < pd.to_datetime(result.valid_start)).all())
            self.assertTrue(result.n_train.eq(300).all())
        options = {**self.fit_options, "reg_covar": 0.001}
        artifacts = fit_candidates(
            self.training_rows
            ,self.feature_sets
            ,None
            ,self.cfg
            ,specifications=self.specifications
            ,fit_options=options
            ,compute_metrics=False
            ,require_convergence=False
        )[0]
        self.assertEqual(next(iter(artifacts.values()))["model"].reg_covar, 0.001)

    def test_recorded_fold_and_saved_bundle_round_trip(self):
        artifact, sample, valid = refit_recorded_fold(
            self.rows, self.feature_sets, self.cfg, self.fit_options
            ,feature_set="D", k=2, model_seed=24
        )
        self.assertLess(sample.Date.max(), valid.Date.min())
        self.assertEqual(len(sample), 300)
        self.assertTrue(artifact["converged"])
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            ids = save_candidate_bundle(
                {artifact["id"]: artifact}, pd.DataFrame([{"candidate_id": artifact["id"]}]), directory
            )
            restored = joblib.load(Path(directory) / "models" / f"{ids['D']}.joblib")
            restored_scores = predict_states(
                restored
                ,valid
                ,self.cfg
                ,include_diagnostics=True
            )
            reference_scores = predict_states(
                artifact
                ,valid
                ,self.cfg
                ,include_diagnostics=True
            )
            np.testing.assert_allclose(restored_scores.log_likelihood, reference_scores.log_likelihood)

    def test_notebook_cells_parse(self):
        notebook = json.loads((ROOT / "notebooks/03b_GMM_Candidate_Diagnostics.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell_{index}")


if __name__ == "__main__":
    unittest.main()
