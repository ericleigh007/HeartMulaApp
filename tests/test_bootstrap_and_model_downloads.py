from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.ai.setup_comparison_model_repos import download_comparison_model_repos
from tools.common.bootstrap_aimusicapp import _load_model_selection


class BootstrapAndModelDownloadsTests(unittest.TestCase):
    def test_load_model_selection_uses_defaults(self) -> None:
        selected = _load_model_selection(None, None, include_audiox=False)
        self.assertIn("ace_step_v15_turbo", selected)
        self.assertIn("ace_step_v15_sft", selected)
        self.assertNotIn("audiox", selected)

    def test_load_model_selection_reads_boolean_config_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "model_support.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "heartmula_hny": True,
                            "heartmula_base": False,
                            "ace_step_v15": True,
                            "ace_step_v15_sft": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            selected = _load_model_selection(str(config_path), None, include_audiox=False)

        self.assertEqual(["heartmula_hny", "ace_step_v15_turbo", "ace_step_v15_sft"], selected)

    def test_load_model_selection_can_add_audiox(self) -> None:
        selected = _load_model_selection(None, ["melodyflow"], include_audiox=True)
        self.assertEqual(["melodyflow", "audiox"], selected)

    def test_download_comparison_model_repos_plans_turbo_and_sft_dependencies(self) -> None:
        result = download_comparison_model_repos(
            "C:/tmp/models/comparison",
            models=["ace_step_v15_turbo", "ace_step_v15_sft"],
            dry_run=True,
        )

        self.assertTrue(result["ok"])
        plan = result["download_plan"]
        self.assertEqual(3, len(plan))
        self.assertTrue(any(item["repo_id"] == "ACE-Step/Ace-Step1.5" and item["allow_patterns"] for item in plan))
        self.assertTrue(any(item["repo_id"] == "ACE-Step/acestep-v15-sft" for item in plan))
        self.assertTrue(any(item["local_dir"].endswith("ace-step-1.5\\checkpoints") or item["local_dir"].endswith("ace-step-1.5/checkpoints") for item in plan))


if __name__ == "__main__":
    unittest.main()