import argparse
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rated_gauntlet as gauntlet  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.options = {
            "Hash": object(),
            "Threads": object(),
            "UCI_LimitStrength": object(),
            "UCI_Elo": object(),
        }
        self.configured = None

    def configure(self, options):
        self.configured = dict(options)


class FakeQualityEngine:
    def analyse(self, board, limit, root_moves=None):
        if root_moves:
            move = root_moves[0]
            score = chess.engine.PovScore(chess.engine.Cp(0), board.turn)
        else:
            move = chess.Move.from_uci("e2e4") if chess.Move.from_uci("e2e4") in board.legal_moves else next(board.legal_moves)
            score = chess.engine.PovScore(chess.engine.Cp(100), board.turn)
        return {"score": score, "pv": [move]}


def benchmark_args(mode):
    return argparse.Namespace(
        hash=64,
        opponent_threads=1,
        mode=mode,
    )


class RatedGauntletTests(unittest.TestCase):
    def test_full_mode_forces_limiter_off(self):
        engine = FakeEngine()
        spec = gauntlet.OpponentSpec("Stockfish", Path("/tmp/stockfish"), options={"UCI_Elo": 1200})
        gauntlet.configure_opponent(engine, spec, benchmark_args("full"), None)
        self.assertFalse(engine.configured["UCI_LimitStrength"])
        self.assertNotIn("UCI_Elo", engine.configured)

    def test_limited_mode_forces_requested_level(self):
        engine = FakeEngine()
        spec = gauntlet.OpponentSpec("Stockfish", Path("/tmp/stockfish"), options={"UCI_Elo": 1200})
        gauntlet.configure_opponent(engine, spec, benchmark_args("limited"), 2200)
        self.assertTrue(engine.configured["UCI_LimitStrength"])
        self.assertEqual(engine.configured["UCI_Elo"], 2200)

    def test_full_mode_accepts_reference_without_strength_options(self):
        engine = FakeEngine()
        engine.options = {"Hash": object(), "Threads": object()}
        spec = gauntlet.OpponentSpec("Historical Stockfish", Path("/tmp/stockfish"))
        gauntlet.configure_opponent(engine, spec, benchmark_args("full"), None)
        self.assertEqual(engine.configured, {"Hash": 64, "Threads": 1})

    def test_opening_pairing_reuses_opening_and_swaps_color_by_contract(self):
        first_index, first_opening = gauntlet.select_opening(4, 17)
        second_index, second_opening = gauntlet.select_opening(5, 17)
        self.assertEqual(first_index, second_index)
        self.assertEqual(first_opening, second_opening)

    def test_reference_config_parses_ratings_and_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "name": "fake",
                            "path": sys.executable,
                            "rating": 2200,
                            "options": {"Threads": 2},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            references = gauntlet.parse_reference_config(path)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].name, "fake")
        self.assertEqual(references[0].rating, 2200.0)
        self.assertEqual(references[0].options["Threads"], 2)

    def test_unanchored_result_does_not_claim_rating(self):
        games = [
            gauntlet.GameResult("Stockfish", None, 1.0, True, 0, "checkmate", 20),
            gauntlet.GameResult("Stockfish", None, 0.0, False, 0, "checkmate", 20),
        ]
        fitted = gauntlet.fit_rating(games)
        self.assertIsNone(fitted["rating"])
        self.assertEqual(fitted["status"], "unanchored")

    def test_rating_fit_recovers_even_score_at_anchor(self):
        games = []
        for index in range(20):
            games.append(
                gauntlet.GameResult(
                    "reference", 2200.0, 1.0 if index % 2 == 0 else 0.0,
                    index % 2 == 0, 0, "checkmate", 30,
                )
            )
        fitted = gauntlet.fit_rating(games)
        self.assertEqual(fitted["status"], "converged")
        self.assertAlmostEqual(fitted["rating"], 2200.0, places=5)
        self.assertLess(fitted["ci95"][0], 2200.0)
        self.assertGreater(fitted["ci95"][1], 2200.0)

    def test_one_sided_results_are_marked_as_boundary(self):
        games = [
            gauntlet.GameResult("reference", 2200.0, 1.0, True, 0, "checkmate", 30)
            for _ in range(8)
        ]
        self.assertEqual(gauntlet.fit_rating(games)["status"], "boundary")

    def test_limit_requires_exactly_one_search_budget(self):
        with self.assertRaises(ValueError):
            gauntlet.build_limit(100, 4, 0)
        with self.assertRaises(ValueError):
            gauntlet.build_limit(0, 0, 0)
        self.assertEqual(gauntlet.build_limit(100, 0, 0).nodes, 100)
        self.assertEqual(gauntlet.build_limit(0, 4, 0).depth, 4)

    def test_quality_samples_replay_only_keplers_moves(self):
        game = chess.pgn.Game()
        game.headers["White"] = "Kepler"
        game.headers["Black"] = "Stockfish"
        game.headers["Opponent"] = "Stockfish"
        game.headers["OpeningIndex"] = "3"
        node = game
        for move_text in ("e2e4", "e7e5", "g1f3", "b8c6"):
            node = node.add_variation(chess.Move.from_uci(move_text))

        samples = gauntlet.collect_quality_samples([game], 0, random.Random(7))
        self.assertEqual([sample.played_move for sample in samples], ["e2e4", "g1f3"])
        self.assertTrue(all(sample.opponent == "Stockfish" for sample in samples))
        self.assertTrue(all(sample.opening_index == 3 for sample in samples))

    def test_quality_audit_reports_kepler_move_loss(self):
        sample = gauntlet.QualitySample(
            fen=chess.Board().fen(),
            played_move="a2a3",
            opponent="Stockfish",
            opening_index=0,
            ply=0,
            phase="opening",
        )
        audit = gauntlet.run_move_quality_audit(
            FakeQualityEngine(),
            [sample],
            chess.engine.Limit(nodes=1),
        )
        self.assertEqual(audit["samples"], 1)
        self.assertEqual(audit["analyzed"], 1)
        self.assertEqual(audit["different_best_move"], 1)
        self.assertEqual(audit["average_cp_loss"], 100)
        self.assertEqual(audit["by_phase"]["opening"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
