import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.advanced_pipeline import sym_predict

ROOT = Path(__file__).resolve().parents[1]


class AdvancedPipelineValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = pd.read_csv(ROOT / "data/cleaned/advanced_modeling_features.csv", parse_dates=["date"])
        cls.audit = json.loads((ROOT / "data/cleaned/advanced_pipeline_audit.json").read_text())
        cls.pred = pd.read_csv(ROOT / "outputs/final_match_predictions_with_explanations.csv")
        cls.sim = pd.read_csv(ROOT / "outputs/final_tournament_probabilities.csv")

    def test_exactly_one_row_per_match(self):
        self.assertEqual(len(self.features), self.features.match_id.nunique())
        self.assertEqual(0, self.audit["duplicate_match_ids"])

    def test_orientation_labels(self):
        decisive = self.features.dropna(subset=["target"])
        expected = (decisive.winner == decisive.team_a).astype(float)
        self.assertTrue(np.array_equal(decisive.target.to_numpy(), expected.to_numpy()))
        self.assertEqual(0, self.audit["orientation_errors"])

    def test_pre_match_update_contract(self):
        self.assertEqual("snapshot before current-match update", self.audit["feature_update_order"])
        self.assertAlmostEqual(0.0, self.features.iloc[0].diff_elo)

    def test_no_team_centric_duplicates_in_training(self):
        self.assertEqual(0, self.audit["recent_team_centric_rows_used_in_training"])
        self.assertEqual(0, self.audit["tournament_summary_rows_used_in_training"])

    def test_served_predictions_are_order_invariant(self):
        self.assertLess(self.audit["served_prediction_mean_order_error"], 1e-12)

    def test_probabilities_and_intervals(self):
        self.assertTrue(self.pred.team_a_win_probability.between(0, 1).all())
        self.assertTrue(self.sim.win_probability.between(0, 1).all())
        self.assertTrue((self.sim.win_lower <= self.sim.win_upper).all())
        self.assertAlmostEqual(1.0, self.sim.win_probability.sum(), places=6)
        self.assertAlmostEqual(2.0, self.sim.final_probability.sum(), places=6)
        self.assertAlmostEqual(4.0, self.sim.semi_final_probability.sum(), places=6)

    def test_era_normalized_features_are_finite(self):
        cols = [c for c in self.features if c.endswith("_z")]
        self.assertTrue(np.isfinite(self.features[cols].to_numpy()).all())

    def test_venue_mapping_is_complete_or_explicitly_unresolved(self):
        mapping = pd.read_csv(ROOT / "data/cleaned/match_host_country_mapping.csv")
        unresolved = mapping.country.isna() | mapping.country.eq("")
        self.assertEqual(2617, len(mapping))
        self.assertEqual(142, int(unresolved.sum()))

    def test_final_explanations_and_uncertainty(self):
        pred = pd.read_csv(ROOT / "outputs/final_match_predictions_with_explanations.csv")
        self.assertEqual(374, len(pred))
        self.assertTrue(np.allclose(pred.team_a_win_probability + pred.team_b_win_probability, 1))
        self.assertTrue((pred.model_probability_low <= pred.model_probability_high).all())
        self.assertFalse(pred.plain_language_explanation.isna().any())

    def test_final_tournament_bounds_contain_points(self):
        final = pd.read_csv(ROOT / "outputs/final_tournament_probabilities.csv")
        for stem in ["semi_final", "final", "win"]:
            self.assertTrue((final[f"{stem}_lower"] <= final[f"{stem}_probability"]).all())
            self.assertTrue((final[f"{stem}_probability"] <= final[f"{stem}_upper"]).all())
        self.assertAlmostEqual(1.0, final.win_probability.sum(), places=6)


if __name__ == "__main__":
    unittest.main()
