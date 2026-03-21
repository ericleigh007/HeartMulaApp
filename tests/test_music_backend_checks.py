from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ai.music_backend_checks import collect_preflight_issues, find_missing_python_modules, get_python_runtime_info


class MusicBackendChecksTests(unittest.TestCase):
    def test_collect_preflight_issues_reports_missing_configuration(self) -> None:
        issues = collect_preflight_issues(
            ["heartmula_hny", "heartmula_base", "audiox", "melodyflow", "ace_step"],
            {
                "HEARTMULA_ROOT": "",
                "HEARTMULA_CKPT_DIR": "",
                "HEARTMULA_HNY_CKPT_DIR": "",
                "HEARTMULA_BASE_CKPT_DIR": "",
                "HEARTMULA_PYTHON": "",
                "AUDIOX_PYTHON": "",
                "MELODYFLOW_PYTHON": "",
                "MELODYFLOW_MODEL_DIR": "",
                "ACESTEP_COMMAND_TEMPLATE": "",
            },
        )
        self.assertTrue(any("HeartMuLa Root" in issue for issue in issues))
        self.assertTrue(any("HeartMuLa Happy New Year" in issue for issue in issues))
        self.assertTrue(any("HeartMuLa Base 3B" in issue for issue in issues))
        self.assertTrue(any(issue.startswith("AudioX:") for issue in issues))
        self.assertTrue(any("MelodyFlow" in issue for issue in issues))
        self.assertTrue(any("ACE-Step" in issue for issue in issues))

    def test_collect_preflight_issues_accepts_native_melodyflow_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            melodyflow_dir = root / "melodyflow"
            melodyflow_space_dir = root / "MelodyFlowSpace"
            melodyflow_dir.mkdir()
            (melodyflow_space_dir / "audiocraft" / "models").mkdir(parents=True)
            (melodyflow_space_dir / "audiocraft" / "models" / "melodyflow.py").write_text("", encoding="utf-8")
            with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
                with patch("tools.ai.music_backend_checks.get_python_runtime_info", return_value={"major": 3, "minor": 10, "micro": 11, "platform": "win32", "system": "Windows"}):
                    with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=[]):
                        issues = collect_preflight_issues(
                            ["melodyflow"],
                            {
                                "MELODYFLOW_PYTHON": sys.executable,
                                "MELODYFLOW_MODEL_DIR": str(melodyflow_dir),
                                "MELODYFLOW_SPACE_DIR": str(melodyflow_space_dir),
                            },
                        )
            self.assertEqual([], issues)

    def test_collect_preflight_issues_reports_missing_melodyflow_space_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            melodyflow_dir = Path(temp_dir) / "melodyflow"
            melodyflow_dir.mkdir()
            with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
                with patch("tools.ai.music_backend_checks.get_python_runtime_info", return_value={"major": 3, "minor": 10, "micro": 11, "platform": "win32", "system": "Windows"}):
                    with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_backend_checks.find_melodyflow_space_repo", return_value=None):
                            issues = collect_preflight_issues(
                                ["melodyflow"],
                                {
                                    "MELODYFLOW_PYTHON": sys.executable,
                                    "MELODYFLOW_MODEL_DIR": str(melodyflow_dir),
                                    "MELODYFLOW_SPACE_DIR": str(Path(temp_dir) / "missing-space"),
                                },
                            )
        self.assertTrue(any("MELODYFLOW_SPACE_DIR" in issue for issue in issues))

    def test_find_missing_python_modules_reports_unknown_module(self) -> None:
        missing = find_missing_python_modules(sys.executable, ["json", "definitely_missing_test_module"])
        self.assertEqual(["definitely_missing_test_module"], missing)

    def test_get_python_runtime_info_reports_current_interpreter(self) -> None:
        info = get_python_runtime_info(sys.executable)
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(sys.version_info.major, info["major"])
        self.assertEqual(sys.version_info.minor, info["minor"])

    def test_collect_preflight_issues_reports_missing_heartmula_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            heart_ckpt = root / "ckpt"
            heart_root.mkdir()
            heart_ckpt.mkdir()
            with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=["soundfile"]):
                issues = collect_preflight_issues(
                    ["heartmula_hny"],
                    {
                        "HEARTMULA_ROOT": str(heart_root),
                        "HEARTMULA_HNY_CKPT_DIR": str(heart_ckpt),
                        "HEARTMULA_PYTHON": sys.executable,
                    },
                )
        self.assertTrue(any("soundfile" in issue for issue in issues))

    def test_collect_preflight_issues_accepts_base_heartmula_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            heart_ckpt = root / "ckpt"
            heart_root.mkdir()
            heart_ckpt.mkdir()
            with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
                with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=[]):
                    issues = collect_preflight_issues(
                        ["heartmula_base"],
                        {
                            "HEARTMULA_ROOT": str(heart_root),
                            "HEARTMULA_BASE_CKPT_DIR": str(heart_ckpt),
                            "HEARTMULA_PYTHON": sys.executable,
                        },
                    )
        self.assertEqual([], issues)

    def test_collect_preflight_issues_warns_for_audiox_python_312(self) -> None:
        with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
            with patch("tools.ai.music_backend_checks.get_python_runtime_info", return_value={"major": 3, "minor": 12, "micro": 0, "platform": "win32", "system": "Windows"}):
                with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=[]):
                    issues = collect_preflight_issues(["audiox"], {"AUDIOX_PYTHON": sys.executable})
        self.assertTrue(any("Python 3.12+" in issue for issue in issues))

    def test_collect_preflight_issues_reports_missing_audiox_soundfile(self) -> None:
        with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
            with patch("tools.ai.music_backend_checks.get_python_runtime_info", return_value={"major": 3, "minor": 10, "micro": 11, "platform": "win32", "system": "Windows"}):
                with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=["soundfile"]):
                    issues = collect_preflight_issues(["audiox"], {"AUDIOX_PYTHON": sys.executable})
        self.assertTrue(any("soundfile" in issue for issue in issues))

    def test_collect_preflight_issues_reports_missing_melodyflow_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            melodyflow_dir = Path(temp_dir) / "melodyflow"
            melodyflow_dir.mkdir()
            with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
                with patch("tools.ai.music_backend_checks.get_python_runtime_info", return_value={"major": 3, "minor": 10, "micro": 11, "platform": "win32", "system": "Windows"}):
                    with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=["xformers"]):
                        issues = collect_preflight_issues(
                            ["melodyflow"],
                            {"MELODYFLOW_PYTHON": sys.executable, "MELODYFLOW_MODEL_DIR": str(melodyflow_dir)},
                        )
        self.assertTrue(any("xformers" in issue for issue in issues))

    def test_collect_preflight_issues_accepts_native_acestep_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = Path(temp_dir) / "ace-step"
            ckpt_dir.mkdir()
            with patch("tools.ai.music_backend_checks.python_exists", return_value=True):
                with patch("tools.ai.music_backend_checks.get_python_runtime_info", return_value={"major": 3, "minor": 12, "micro": 0, "platform": "win32", "system": "Windows"}):
                    with patch("tools.ai.music_backend_checks.find_missing_python_modules", return_value=[]):
                        issues = collect_preflight_issues(
                            ["ace_step"],
                            {
                                "ACESTEP_PYTHON": sys.executable,
                                "ACESTEP_CKPT_DIR": str(ckpt_dir),
                                "ACESTEP_COMMAND_TEMPLATE": "",
                            },
                        )
        self.assertEqual([], issues)


if __name__ == "__main__":
    unittest.main()
