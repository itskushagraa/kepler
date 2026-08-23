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


if __name__ == "__main__":
    unittest.main()
