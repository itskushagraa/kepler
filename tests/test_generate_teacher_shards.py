import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_teacher_shards as shards  # noqa: E402


class TeacherShardTests(unittest.TestCase):
    def test_completed_shard_requires_matching_summary_and_nonempty_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.tsv"
            data.write_text("row\n", encoding="utf-8")
            (root / "summary_1.json").write_text(
                json.dumps({"games": 10, "data_path": str(data)}), encoding="utf-8"
            )
            self.assertEqual(shards.completed_shard(root, 10), data.resolve())
            self.assertIsNone(shards.completed_shard(root, 11))


if __name__ == "__main__":
    unittest.main()
