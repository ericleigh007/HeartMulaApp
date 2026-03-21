from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.ai.music_compare_gui import (
    detect_default_backend_settings,
    expected_backend_output_path,
    live_generated_audio_seconds,
    load_gui_settings,
    save_gui_settings,
)


class MusicCompareGuiSettingsTests(unittest.TestCase):
    def test_detect_default_backend_settings_prefers_repo_specific_envs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "third_party" / "heartlib").mkdir(parents=True)
            (repo_root / "models" / "heartmula" / "happy-new-year").mkdir(parents=True)
            (repo_root / "models" / "heartmula" / "base").mkdir(parents=True)
            (repo_root / "models" / "comparison" / "melodyflow" / "melodyflow-t24-30secs").mkdir(parents=True)
            (repo_root / "models" / "comparison" / "ace-step" / "ACE-Step-v1-3.5B").mkdir(parents=True)

            for env_dir in [".venv-heartmula", ".venv-melodyflow", ".venv-acestep"]:
                python_path = repo_root / env_dir / "Scripts" / "python.exe"
                python_path.parent.mkdir(parents=True)
                python_path.write_text("", encoding="utf-8")

            defaults = detect_default_backend_settings(repo_root, {}, fallback_python="C:/fallback/python.exe")

        self.assertTrue(defaults["HEARTMULA_ROOT"].endswith("third_party\\heartlib"))
        self.assertTrue(defaults["HEARTMULA_HNY_CKPT_DIR"].endswith("models\\heartmula\\happy-new-year"))
        self.assertTrue(defaults["HEARTMULA_BASE_CKPT_DIR"].endswith("models\\heartmula\\base"))
        self.assertEqual(defaults["HEARTMULA_CKPT_DIR"], defaults["HEARTMULA_HNY_CKPT_DIR"])
        self.assertTrue(defaults["HEARTMULA_PYTHON"].endswith(".venv-heartmula\\Scripts\\python.exe"))
        self.assertEqual("1.5", defaults["HEARTMULA_CFG_SCALE"])
        self.assertEqual("true", defaults["HEARTMULA_LAZY_LOAD"])
        self.assertTrue(defaults["MELODYFLOW_PYTHON"].endswith(".venv-melodyflow\\Scripts\\python.exe"))
        self.assertTrue(defaults["ACESTEP_PYTHON"].endswith(".venv-acestep\\Scripts\\python.exe"))

    def test_gui_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            payload = {
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
                "HEARTMULA_HNY_CKPT_DIR": "C:/heart/hny",
            }
            save_gui_settings(payload, settings_path)
            loaded = load_gui_settings(settings_path)

        self.assertEqual(payload, loaded)

    def test_expected_backend_output_path_uses_backend_filename(self) -> None:
        output_path = expected_backend_output_path("C:/tmp/demo", "heartmula_hny")
        self.assertIsNotNone(output_path)
        self.assertTrue(str(output_path).endswith("heartmula_hny_output.wav"))

    def test_live_generated_audio_seconds_ignores_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "heartmula_output.wav"
            sf.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
            old_epoch = time.time() - 120
            os.utime(audio_path, (old_epoch, old_epoch))

            measured = live_generated_audio_seconds(audio_path, started_epoch=time.time())

        self.assertIsNone(measured)


if __name__ == "__main__":
    unittest.main()