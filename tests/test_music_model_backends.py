from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ai.music_model_backends import ACEStep15Backend, ACEStepBackend, AudioXBackend, HeartMuLaBackend, MelodyFlowBackend, MusicGenRequest


class MusicModelBackendsTests(unittest.TestCase):
    def test_audiox_backend_reports_missing_modules(self) -> None:
        with patch.dict(os.environ, {"AUDIOX_PYTHON": "C:/fake/python.exe"}, clear=False):
            with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=["audiox", "einops", "soundfile"]):
                    result = AudioXBackend().run(MusicGenRequest(prompt="test", output_dir=".tmp/test-audiox"))
        self.assertFalse(result.success)
        self.assertIn("audiox, einops, soundfile", result.error)

    def test_audiox_backend_uses_configured_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "audiox_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            with patch.dict(os.environ, {"AUDIOX_PYTHON": "C:/custom/python.exe"}, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = AudioXBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/custom/python.exe", recorded["command"][0])

    def test_heartmula_backend_uses_configured_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            ckpt_dir = root / "ckpt"
            heart_root.mkdir()
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "heartmula_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_CKPT_DIR": str(ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = HeartMuLaBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/heart/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_heartmula_backend.py"))
        self.assertIn("--cfg-scale", recorded["command"])
        self.assertIn("1.5", recorded["command"])

    def test_heartmula_variant_backend_uses_variant_checkpoint_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            base_ckpt_dir = root / "ckpt-base"
            heart_root.mkdir()
            base_ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "heartmula_base_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_BASE_CKPT_DIR": str(base_ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
            }
            backend = HeartMuLaBackend(name="heartmula_base", checkpoint_env="HEARTMULA_BASE_CKPT_DIR")
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = backend.run(request)

        self.assertTrue(result.success)
        self.assertIn(str(base_ckpt_dir), recorded["command"])

    def test_heartmula_backend_reads_latency_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            ckpt_dir = root / "ckpt"
            heart_root.mkdir()
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "heartmula_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_CKPT_DIR": str(ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
                "HEARTMULA_CFG_SCALE": "1.0",
                "HEARTMULA_LAZY_LOAD": "false",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = HeartMuLaBackend().run(request)

        self.assertTrue(result.success)
        cfg_scale_index = recorded["command"].index("--cfg-scale") + 1
        self.assertEqual("1.0", recorded["command"][cfg_scale_index])
        lazy_load_index = recorded["command"].index("--lazy-load") + 1
        self.assertEqual("false", recorded["command"][lazy_load_index])

    def test_heartmula_backend_writes_descriptor_and_lyrics_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            ckpt_dir = root / "ckpt"
            heart_root.mkdir()
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="night drive",
                output_dir=str(output_dir),
                duration_seconds=1.0,
                tags="synthwave, neon bass",
                lyrics=" line one\nline two\n",
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                tags_path = Path(command[command.index("--tags") + 1])
                lyrics_path = Path(command[command.index("--lyrics") + 1])
                recorded["tags_text"] = tags_path.read_text(encoding="utf-8")
                recorded["lyrics_text"] = lyrics_path.read_text(encoding="utf-8")
                (output_dir / "heartmula_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_CKPT_DIR": str(ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = HeartMuLaBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("night drive. synthwave, neon bass\n", recorded["tags_text"])
        self.assertEqual("line one\nline two", recorded["lyrics_text"])
        self.assertEqual("night drive. synthwave, neon bass", result.metadata["effective_descriptor_text"])
        self.assertEqual("synthwave, neon bass", result.metadata["tags"])
        self.assertEqual(" line one\nline two\n", result.metadata["lyrics"])

    def test_acestep_backend_uses_configured_python_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt_dir = root / "ace-step"
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "ACESTEP_CKPT_DIR": str(ckpt_dir),
                "ACESTEP_PYTHON": "C:/ace/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = ACEStepBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/ace/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_acestep_backend.py"))
        self.assertIn(str(ckpt_dir.resolve()), recorded["command"])

    def test_acestep15_backend_passes_tags_lyrics_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "ace-step-1.5"
            checkpoints_dir = root / "checkpoints"
            source_root.mkdir()
            checkpoints_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="cinematic rise",
                output_dir=str(output_dir),
                duration_seconds=2.0,
                tags="orchestral, trailer",
                lyrics="hold the line",
                seed=1234,
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_v15_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "ACESTEP15_ROOT": str(source_root),
                "ACESTEP15_CKPT_DIR": str(checkpoints_dir),
                "ACESTEP15_PYTHON": "C:/ace15/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = ACEStep15Backend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/ace15/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_acestep15_backend.py"))
        self.assertIn("cinematic rise", recorded["command"])
        self.assertIn("orchestral, trailer", recorded["command"])
        self.assertIn("hold the line", recorded["command"])
        self.assertIn("1234", recorded["command"])
        self.assertEqual("orchestral, trailer", result.metadata["tags"])
        self.assertEqual("hold the line", result.metadata["lyrics"])

    def test_melodyflow_backend_uses_configured_python_and_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "melodyflow"
            model_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0, seed=7)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "melodyflow_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "MELODYFLOW_MODEL_DIR": str(model_dir),
                "MELODYFLOW_PYTHON": "C:/melody/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = MelodyFlowBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/melody/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_melodyflow_backend.py"))
        self.assertIn(str(model_dir.resolve()), recorded["command"])
        self.assertIn("--seed", recorded["command"])


if __name__ == "__main__":
    unittest.main()
