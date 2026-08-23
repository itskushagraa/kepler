import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import eval_quality  # noqa: E402


class EvalQualityTests(unittest.TestCase):
    def test_position_classification_uses_material(self):
        self.assertEqual(eval_quality.classify_position(chess.Board())[0], "opening")
        phase, material = eval_quality.classify_position(
            chess.Board("8/8/8/8/8/8/4P3/4K2k w - - 0 60")
        )
        self.assertEqual((phase, material), ("endgame", "pawn_only"))

    def test_summary_reports_bias_and_error(self):
        report = eval_quality.summarize(
            [
                {"engine_cp": 100, "teacher_cp": 80},
                {"engine_cp": -40, "teacher_cp": -20},
            ]
        )
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["mae_cp"], 20.0)
        self.assertAlmostEqual(report["bias_cp"], 0.0)
        self.assertEqual(report["sign_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
