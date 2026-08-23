import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import reference_ladder as ladder  # noqa: E402


class ReferenceLadderTests(unittest.TestCase):
    def test_config_requires_two_unique_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(
                json.dumps(
                    [
                        {"name": "A", "path": sys.executable},
                        {"name": "B", "path": sys.executable},
                    ]
                ),
                encoding="utf-8",
            )
            specs = ladder.parse_engine_config(path)
        self.assertEqual([spec.name for spec in specs], ["A", "B"])

    def test_equal_results_recover_anchor_rating(self):
        specs = [
            ladder.EngineSpec("Anchor", Path(sys.executable)),
            ladder.EngineSpec("Reference", Path(sys.executable)),
        ]
        games = []
        for index in range(20):
            if index % 2 == 0:
                games.append(ladder.CalibrationGame("Anchor", "Reference", 1.0, 0, "test", 20))
            else:
                games.append(ladder.CalibrationGame("Reference", "Anchor", 1.0, 0, "test", 20))
        fitted = ladder.fit_ratings(specs, games, "Anchor", 2500.0)
        self.assertEqual(fitted["status"], "converged")
        self.assertAlmostEqual(fitted["ratings"]["Reference"]["rating"], 2500.0, places=5)

    def test_one_sided_reference_results_are_marked_boundary(self):
        specs = [
            ladder.EngineSpec("Anchor", Path(sys.executable)),
            ladder.EngineSpec("Reference", Path(sys.executable)),
        ]
        games = [
            ladder.CalibrationGame("Anchor", "Reference", 1.0, 0, "test", 20)
            for _ in range(20)
        ]
        fitted = ladder.fit_ratings(specs, games, "Anchor", 2500.0)
        self.assertEqual(fitted["ratings"]["Reference"]["status"], "boundary")

    def test_target_is_excluded_from_generated_opponent_config(self):
        specs = [
            ladder.EngineSpec("Anchor", Path("/anchor")),
            ladder.EngineSpec("Target", Path("/target")),
        ]
        games = [
            ladder.CalibrationGame("Anchor", "Target", 1.0, 0, "test", 20),
            ladder.CalibrationGame("Target", "Anchor", 0.0, 0, "test", 20),
        ]
        fitted = ladder.fit_ratings(specs, games, "Anchor", 2500.0)
        config = ladder.build_reference_config(specs, fitted, "Target")
        self.assertEqual([entry["name"] for entry in config["opponents"]], ["Anchor"])
        self.assertEqual(config["target_name"], "Target")


if __name__ == "__main__":
    unittest.main()
