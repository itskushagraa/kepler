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
ACCOUNT_SPEC = importlib.util.spec_from_file_location(
    "kepler_lichess_account", ROOT / "tools" / "lichess_account.py"
)
assert ACCOUNT_SPEC and ACCOUNT_SPEC.loader
ACCOUNT = importlib.util.module_from_spec(ACCOUNT_SPEC)
ACCOUNT_SPEC.loader.exec_module(ACCOUNT)


class LichessDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_is_secret_free_and_uses_environment_placeholder(self):
        self.assertEqual(self.config["token"], "set-via-LICHESS_BOT_TOKEN")
        self.assertNotIn("Bearer ", CONFIG_PATH.read_text(encoding="utf-8"))

    def test_live_policy_is_standard_casual_one_plus_zero_or_slower(self):
        challenge = self.config["challenge"]
        self.assertEqual(challenge["concurrency"], 1)
        self.assertEqual(challenge["variants"], ["standard"])
        self.assertEqual(challenge["time_controls"], ["bullet", "blitz", "rapid", "classical"])
        self.assertNotIn("correspondence", challenge["time_controls"])
        self.assertEqual(challenge["modes"], ["casual"])
        self.assertEqual((challenge["min_base"], challenge["max_base"]), (60, 10800))
        self.assertEqual((challenge["min_increment"], challenge["max_increment"]), (0, 180))
        self.assertFalse(challenge["accept_bot"])
        self.assertFalse(challenge["only_bot"])
        self.assertFalse(self.config["matchmaking"]["allow_matchmaking"])

    def test_engine_configuration_matches_keplers_uci_options(self):
        engine = self.config["engine"]
        self.assertEqual(engine["protocol"], "uci")
        self.assertFalse(engine["ponder"])
        self.assertEqual(engine["working_dir"], "../..")
        self.assertEqual(
            engine["uci_options"],
            {"Hash": 256, "Threads": 4, "MoveOverhead": 900, "UseBaseline": True},
        )
        self.assertEqual(self.config["move_overhead"], 3000)

    def test_pinned_bridge_has_bullet_first_move_patch(self):
        self.assertTrue(LICHESS.BRIDGE_PATCHES)
        for patch in LICHESS.BRIDGE_PATCHES:
            self.assertTrue(patch.is_file())
            contents = patch.read_text(encoding="utf-8")
            self.assertIn(
                "+    search_time = max(msec(100), min(seconds(1), game.clock_initial / 100))",
                contents,
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

    def test_account_check_only_allows_zero_game_or_existing_bot_accounts(self):
        eligible, safe = ACCOUNT.summarize_profile(
            {"username": "Kepler", "count": {"all": 0}}
        )
        self.assertTrue(safe)
        self.assertEqual(eligible["upgrade_state"], "eligible_zero_games")

        played, safe = ACCOUNT.summarize_profile(
            {"username": "Kepler", "count": {"all": 1}}
        )
        self.assertFalse(safe)
        self.assertEqual(played["upgrade_state"], "blocked_has_played_games")

        existing, safe = ACCOUNT.summarize_profile(
            {"username": "Kepler", "title": "BOT", "count": {"all": 12}}
        )
        self.assertTrue(safe)
        self.assertEqual(existing["upgrade_state"], "already_bot")


if __name__ == "__main__":
    unittest.main()
