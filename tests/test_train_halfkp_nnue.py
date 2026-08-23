import sys
import tempfile
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nnue_cycle  # noqa: E402
import train_halfkp_nnue as trainer  # noqa: E402

try:
    torch = trainer.require_torch()
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except RuntimeError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


class HalfKPTrainingTests(unittest.TestCase):
    def test_grouped_split_never_leaks_a_game(self):
        samples = [
            trainer.HalfKPSample([1], [2], float(index), f"game-{index // 3}")
            for index in range(30)
        ]
        train, validation, train_groups, validation_groups = (
            trainer.split_samples_by_group(samples, 0.2, 42)
        )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(set(train_groups).isdisjoint(validation_groups))
        self.assertTrue(
            {sample.group for sample in train}.isdisjoint(
                {sample.group for sample in validation}
            )
        )

    def test_grouped_split_is_deterministic(self):
        samples = [
            trainer.HalfKPSample([1], [2], float(index), f"game-{index // 2}")
            for index in range(20)
        ]
        first = trainer.split_samples_by_group(samples, 0.25, 17)
        second = trainer.split_samples_by_group(samples, 0.25, 17)
        self.assertEqual(first[2:], second[2:])

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
        samples.append(trainer.HalfKPSample([1], [2], 0.0, "rare-game", bucket="rare"))
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

    @unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is unavailable")
    def test_numpy_and_torch_cpu_forward_parity(self):
        import numpy as np

        rng = np.random.default_rng(9)
        hidden_size = 8
        feature_weights = rng.normal(
            0.0, 0.1, (trainer.HALFKP_INPUT_SIZE, hidden_size)
        ).astype(np.float32)
        hidden_bias = rng.normal(0.1, 0.01, hidden_size).astype(np.float32)
        output_weights = rng.normal(0.0, 0.05, hidden_size).astype(np.float32)
        output_bias = 3.25
        us = np.array([[1, 5, 9, -1], [2, 7, -1, -1], [3, 4, 8, 12]], dtype=np.int32)
        them = np.array([[2, 6, -1], [1, 9, 10], [5, -1, -1]], dtype=np.int32)
        expected = trainer.predict(
            feature_weights, hidden_bias, output_weights, output_bias, us, them
        )
        model = trainer.create_torch_model(
            hidden_size, feature_weights, hidden_bias, output_weights, output_bias
        )
        actual = trainer.torch_predict(model, us, them, "cpu", batch_size=2)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    @staticmethod
    def _tiny_torch_data():
        import numpy as np

        train_us = np.array(
            [[1, 2, -1], [2, 3, -1], [4, 5, 6], [1, 6, -1], [7, 8, -1], [2, 9, -1]],
            dtype=np.int32,
        )
        train_them = np.array(
            [[4, 5, -1], [1, 7, -1], [2, 8, -1], [3, 9, -1], [1, 4, -1], [5, 8, -1]],
            dtype=np.int32,
        )
        targets = np.array([20.0, -15.0, 35.0, -30.0, 10.0, -5.0], dtype=np.float32)
        weights = np.ones(len(targets), dtype=np.float32)
        return train_us, train_them, targets, weights

    @unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is unavailable")
    def test_torch_cpu_training_quantization_and_checkpoint_resume(self):
        import numpy as np

        train_us, train_them, targets, weights = self._tiny_torch_data()
        validation_us = train_us[:2]
        validation_them = train_them[:2]
        validation_targets = targets[:2]
        validation_weights = weights[:2]
        arguments = (
            train_us,
            train_them,
            targets,
            weights,
            validation_us,
            validation_them,
            validation_targets,
            validation_weights,
            8,
        )
        continuous = trainer.train_torch(
            *arguments, 2, 2, 0.002, 0.0, 11, 0, 1, 0.0, device="cpu"
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            trainer.train_torch(
                *arguments,
                1,
                2,
                0.002,
                0.0,
                11,
                0,
                1,
                0.0,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )
            latest = checkpoint_dir / "latest.pt"
            self.assertTrue(latest.is_file())
            resumed = trainer.train_torch(
                *arguments,
                2,
                2,
                0.002,
                0.0,
                11,
                0,
                1,
                0.0,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                resume_checkpoint=latest,
            )

            np.testing.assert_allclose(
                resumed["feature_weights"],
                continuous["feature_weights"],
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                resumed["output_weights"],
                continuous["output_weights"],
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(len(resumed["history"]), len(continuous["history"]))
            self.assertAlmostEqual(
                resumed["history"][-1]["validation_rmse"],
                continuous["history"][-1]["validation_rmse"],
                places=3,
            )

            quantized = trainer.quantize(
                resumed["feature_weights"],
                resumed["hidden_bias"],
                resumed["output_weights"],
                float(resumed["output_bias"]),
                train_us,
                train_them,
            )
            model_path = Path(directory) / "torch-smoke.nnue"
            trainer.write_numpy_model(model_path, *quantized)
            self.assertEqual(model_path.read_bytes()[:8], b"KNNUEv1\x00")

    @unittest.skipUnless(TORCH_CUDA_AVAILABLE, "CUDA PyTorch is unavailable")
    def test_torch_cuda_forward_matches_cpu(self):
        import numpy as np

        rng = np.random.default_rng(23)
        hidden_size = 8
        feature_weights = rng.normal(
            0.0, 0.1, (trainer.HALFKP_INPUT_SIZE, hidden_size)
        ).astype(np.float32)
        hidden_bias = rng.normal(0.1, 0.01, hidden_size).astype(np.float32)
        output_weights = rng.normal(0.0, 0.05, hidden_size).astype(np.float32)
        us = np.array([[1, 3, 5], [2, 4, -1], [8, 13, 21]], dtype=np.int32)
        them = np.array([[2, 4, 6], [1, 7, 9], [3, -1, -1]], dtype=np.int32)
        model = trainer.create_torch_model(
            hidden_size, feature_weights, hidden_bias, output_weights, -2.5
        )
        cpu = trainer.torch_predict(model, us, them, "cpu")
        cuda = trainer.torch_predict(model, us, them, "cuda")
        np.testing.assert_allclose(cuda, cpu, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(TORCH_CUDA_AVAILABLE, "CUDA PyTorch is unavailable")
    def test_torch_cuda_training_smoke(self):
        import numpy as np

        train_us, train_them, targets, weights = self._tiny_torch_data()
        trained = trainer.train_torch(
            train_us,
            train_them,
            targets,
            weights,
            train_us[:2],
            train_them[:2],
            targets[:2],
            weights[:2],
            8,
            1,
            2,
            0.002,
            0.0,
            31,
            0,
            1,
            0.0,
            device="cuda",
        )
        self.assertEqual(trained["device"], "cuda")
        self.assertTrue(np.isfinite(trained["feature_weights"]).all())


if __name__ == "__main__":
    unittest.main()
