import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import curate_dataset as curate  # noqa: E402


def row(ordinal, game, score, ply, kind):
    fen = f"4k3/8/8/8/8/8/4P3/4K3 {'w' if ply % 2 == 0 else 'b'} - - 0 {ply // 2 + 1}"
    return curate.Row(
        ordinal=ordinal,
        result_stm=0,
        score_cp=score,
        fen=fen,
        ply=ply,
        bucket=curate.score_bucket(score),
        opening_id="0",
        termination="draw",
        game_id=game,
        kind=kind,
        phase=curate.phase_from_ply(ply, 20, 60),
        fields=[],
        material="queens",
    )


class DatasetCurationTests(unittest.TestCase):
    def test_position_key_ignores_only_move_clocks(self):
        first = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        later = "4k3/8/8/8/8/8/4P3/4K3 w - - 37 92"
        different_side = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"
        self.assertEqual(curate.position_key(first), curate.position_key(later))
        self.assertNotEqual(curate.position_key(first), curate.position_key(different_side))

    def test_explicit_game_ids_are_namespaced_per_source(self):
        line = "0\t0\t4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\t0\t-50_to_50\t0\tdraw\tgame-1\tregular\n"
        line2 = "0\t0\t4k3/8/8/8/8/8/3P4/4K3 w - - 0 1\t0\t-50_to_50\t0\tdraw\tgame-1\tregular\n"
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.tsv"
            second = Path(directory) / "b.tsv"
            first.write_text(line, encoding="utf-8")
            second.write_text(line2, encoding="utf-8")
            rows, stats = curate.load_rows([first, second], 20, 60)
        self.assertEqual(stats["unique"], 2)
        self.assertEqual(len({sample.game_id for sample in rows}), 2)

    def test_balancing_hits_kind_target_and_spreads_score_buckets(self):
        rows = []
        ordinal = 0
        for kind in ("regular", "tactical"):
            for score in (-500, -200, -100, 0, 100, 200, 500):
                for index in range(10):
                    rows.append(row(ordinal, f"{kind}-{score}-{index}", score, 30, kind))
                    ordinal += 1
        selected = curate.balance_rows(rows, 70, 0.5, 7)
        kinds = {kind: sum(sample.balance_kind == kind for sample in selected) for kind in ("regular", "tactical")}
        buckets = {bucket: sum(sample.bucket == bucket for sample in selected) for bucket in curate.SCORE_BUCKETS}
        self.assertEqual(kinds, {"regular": 35, "tactical": 35})
        self.assertEqual(set(buckets.values()), {10})

    def test_game_split_has_no_leakage(self):
        rows = [row(index, f"game-{index // 3}", 0, 24 + index, "regular") for index in range(30)]
        train, validation = curate.split_games(rows, 0.2, 42)
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, {sample.game_id for sample in rows})

    def test_phase_and_material_use_board_contents(self):
        opening = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ending = "8/8/8/8/8/8/4P3/4K2k w - - 0 60"
        self.assertEqual(curate.phase_from_position(opening, 0, 20, 60), "opening")
        self.assertEqual(curate.phase_from_position(ending, 118, 20, 60), "endgame")
        self.assertEqual(curate.material_class_from_fen(ending), "pawn_only")

    def test_summary_exposes_scale_and_diversity_metrics(self):
        rows = [row(index, f"game-{index}", 400 if index % 2 else 0, 30, "regular") for index in range(4)]
        summary = curate.row_summary(rows)
        self.assertEqual(summary["games"], 4)
        self.assertEqual(summary["extreme_score_fraction"], 0.5)
        self.assertEqual(summary["largest_game_fraction"], 0.25)
        self.assertIn("material_counts", summary)

    def test_extreme_fraction_is_enforced_when_replacements_exist(self):
        rows = [row(index, f"game-{index}", 500, 30, "regular") for index in range(8)]
        rows += [row(8 + index, f"quiet-{index}", 0, 30, "regular") for index in range(8)]
        selected = curate.balance_rows(rows, 10, 0.0, 42, max_extreme_fraction=0.5)
        self.assertLessEqual(sum(abs(sample.score_cp) >= 301 for sample in selected), 5)


if __name__ == "__main__":
    unittest.main()
