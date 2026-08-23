import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import search_quality


class SearchQualityTests(unittest.TestCase):
    def test_collect_positions_and_grouped_loss(self) -> None:
        pgn = """[Event \"test\"]
[White \"Kepler\"]
[Black \"Opponent\"]
[Result \"*\"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 *
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.pgn"
            path.write_text(pgn, encoding="utf-8")
            positions = search_quality.collect_positions(path, "kepler")

        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0]["pgn_move"], "e1g1")
        self.assertIn(positions[0]["phase"], {"opening", "middlegame", "endgame"})
        report = search_quality.grouped_loss(
            [
                {"phase": "opening", "cp_loss": 10.0},
                {"phase": "opening", "cp_loss": 30.0},
            ],
            "phase",
        )
        self.assertEqual(report["opening"]["average_cp_loss"], 20.0)


if __name__ == "__main__":
    unittest.main()
