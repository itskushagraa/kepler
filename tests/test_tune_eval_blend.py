import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import tune_eval_blend  # noqa: E402


class TuneEvalBlendTests(unittest.TestCase):
    def test_integer_list(self):
        self.assertEqual(tune_eval_blend.integer_list("0, 25,100"), [0, 25, 100])
        with self.assertRaises(Exception):
            tune_eval_blend.integer_list("")


if __name__ == "__main__":
    unittest.main()
