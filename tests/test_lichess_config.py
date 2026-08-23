import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "deploy" / "lichess" / "config.yml"
SPEC = importlib.util.spec_from_file_location("kepler_lichess_bot", ROOT / "tools" / "lichess_bot.py")
assert SPEC and SPEC.loader
LICHESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LICHESS)


class LichessDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_is_secret_free_and_uses_environment_placeholder(self):
        self.assertEqual(self.config["token"], "set-via-LICHESS_BOT_TOKEN")
        self.assertNotIn("Bearer ", CONFIG_PATH.read_text(encoding="utf-8"))

    def test_first_live_policy_is_standard_casual_five_plus_three(self):
        challenge = self.config["challenge"]
        self.assertEqual(challenge["concurrency"], 1)
        self.assertEqual(challenge["variants"], ["standard"])
        self.assertEqual(challenge["time_controls"], ["blitz"])
        self.assertEqual(challenge["modes"], ["casual"])
        self.assertEqual((challenge["min_base"], challenge["max_base"]), (300, 300))
        self.assertEqual((challenge["min_increment"], challenge["max_increment"]), (3, 3))
        self.assertFalse(challenge["accept_bot"])
        self.assertFalse(self.config["matchmaking"]["allow_matchmaking"])

    def test_engine_configuration_matches_keplers_uci_options(self):
        engine = self.config["engine"]
        self.assertEqual(engine["protocol"], "uci")
        self.assertFalse(engine["ponder"])
        self.assertEqual(engine["working_dir"], "../..")
        self.assertEqual(
            engine["uci_options"],
            {"Hash": 256, "Threads": 4, "MoveOverhead": 150, "UseBaseline": True},
        )

    def test_token_is_required_for_network_commands(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                LICHESS.require_token()

    def test_bridge_version_is_pinned(self):
        self.assertEqual(len(LICHESS.BRIDGE_COMMIT), 40)
        int(LICHESS.BRIDGE_COMMIT, 16)

    def test_irreversible_upgrade_requires_explicit_confirmation(self):
        args = LICHESS.argparse.Namespace(confirm_irreversible=False, verbose=False)
        with self.assertRaises(SystemExit):
            LICHESS.command_upgrade(args)


if __name__ == "__main__":
    unittest.main()
