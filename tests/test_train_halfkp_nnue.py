import sys
import tempfile
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nnue_cycle  # noqa: E402
import train_halfkp_nnue as trainer  # noqa: E402


class HalfKPTrainingTests(unittest.TestCase):
    def test_grouped_split_never_leaks_a_game(self):
        samples = [
            trainer.HalfKPSample([1], [2], float(index), f"game-{index // 3}")
            for index in range(30)
        ]
        train, validation, train_groups, validation_groups = trainer.split_samples_by_group(
            samples, 0.2, 42
        )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(set(train_groups).isdisjoint(validation_groups))
        self.assertTrue({sample.group for sample in train}.isdisjoint(
            {sample.group for sample in validation}
        ))

    def test_current_dataset_metadata_is_parsed(self):
        row = (
            "0\t25\t4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\t0\t-50_to_50\t3\t"
            "unterminated\tgame-17\ttactical\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tsv"
            path.write_text(row, encoding="utf-8")
            samples = trainer.parse_samples(str(path), 0.0, 400.0, 100.0)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].group, "game-17")
        self.assertEqual(samples[0].kind, "tactical")
        self.assertEqual(samples[0].bucket, "-50_to_50")
        self.assertEqual(samples[0].material, "pawn_only")
        self.assertEqual(samples[0].teacher_cp, 25)

    def test_score_bucket_weights_raise_scarce_samples(self):
        samples = [
            trainer.HalfKPSample([1], [2], 0.0, f"game-{index}", bucket="common")
            for index in range(9)
        ]
        samples.append(
            trainer.HalfKPSample([1], [2], 0.0, "rare-game", bucket="rare")
        )
        weights = trainer.sample_weights(samples, 1.0, 0.5)
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_segmented_metrics_expose_phase_error(self):
        samples = [
            trainer.HalfKPSample([1], [2], 10.0, "a", phase="opening"),
            trainer.HalfKPSample([1], [2], -10.0, "b", phase="endgame"),
        ]
        import numpy as np

        report = trainer.segmented_prediction_summary(
            np.array([12.0, -6.0]), np.array([10.0, -10.0]), samples, "phase"
        )
        self.assertEqual(report["opening"]["mae"], 2.0)
        self.assertEqual(report["endgame"]["mae"], 4.0)

    def test_centipawn_target_preserves_units_and_blends_wdl(self):
        row = (
            "1\t400\t4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\t0\tge_301\t3\t"
            "checkmate\tgame-18\tregular\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tsv"
            path.write_text(row, encoding="utf-8")
            samples = trainer.parse_samples(
                str(path), 0.25, 400.0, 100.0, "centipawn", 2000.0, 600.0
            )
        self.assertEqual(samples[0].target, 450.0)

    def test_tactical_classifier_distinguishes_forcing_positions(self):
        self.assertFalse(nnue_cycle.is_tactical_position(chess.Board()))
        checked = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
        self.assertTrue(nnue_cycle.is_tactical_position(checked))
        capture = chess.Board("4k3/8/8/8/8/8/4q3/4R1K1 w - - 0 1")
        self.assertTrue(
            nnue_cycle.is_tactical_position(capture, chess.Move.from_uci("e1e2"))
        )
        self.assertFalse(
            nnue_cycle.is_tactical_position(chess.Board(), chess.Move.from_uci("e2e4"))
        )


if __name__ == "__main__":
    unittest.main()
