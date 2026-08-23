import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nnue_ab_sprt as sprt  # noqa: E402


class NnueAbSprtTests(unittest.TestCase):
    def test_probabilities_preserve_declared_draw_rate_and_score(self):
        probabilities = sprt.result_probabilities(0.0, 0.4)
        self.assertAlmostEqual(probabilities["draw"], 0.4)
        self.assertAlmostEqual(probabilities["win"], probabilities["loss"])
        self.assertAlmostEqual(
            probabilities["win"] + 0.5 * probabilities["draw"], 0.5
        )

    def test_positive_candidate_result_moves_llr_up(self):
        win = sprt.llr_increment("win", 0.0, 30.0, 0.35)
        loss = sprt.llr_increment("loss", 0.0, 30.0, 0.35)
        self.assertGreater(win, 0.0)
        self.assertLess(loss, 0.0)

    def test_bounds_and_decisions(self):
        lower, upper = sprt.sprt_bounds(0.05, 0.05)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertEqual(sprt.sprt_decision(lower - 0.01, lower, upper), "reject_candidate")
        self.assertEqual(sprt.sprt_decision(upper + 0.01, lower, upper), "accept_candidate")
        self.assertEqual(sprt.sprt_decision(0.0, lower, upper), "continue")

    def test_candidate_score_respects_color(self):
        self.assertEqual(sprt.candidate_score("1-0", True), ("win", 1.0))
        self.assertEqual(sprt.candidate_score("1-0", False), ("loss", 0.0))
        self.assertEqual(sprt.candidate_score("0-1", False), ("win", 1.0))
        self.assertEqual(sprt.candidate_score("1/2-1/2", True), ("draw", 0.5))


if __name__ == "__main__":
    unittest.main()
