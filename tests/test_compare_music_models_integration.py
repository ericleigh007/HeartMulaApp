from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ai.compare_music_models import run_comparison
from tools.ai.music_model_backends import MusicBackendResult


class CompareMusicModelsIntegrationTests(unittest.TestCase):
    def test_run_comparison_records_prompt_tags_lyrics_seed_and_output(self) -> None:
        class CapturingBackend:
            def __init__(self, name: str):
                self.name = name
                self.requests = []

            def run(self, request):
                self.requests.append(request)
                output_dir = Path(request.output_dir)
                output_path = output_dir / f"{self.name}_output.wav"
                output_path.write_bytes(b"RIFFtest")
                return MusicBackendResult(
                    backend=self.name,
                    success=True,
                    output_path=str(output_path),
                    elapsed_seconds=0.01,
                    command=["fake-backend", self.name],
                    metadata={
                        "prompt": request.prompt,
                        "tags": request.tags,
                        "lyrics": request.lyrics,
                        "seed": request.seed,
                    },
                )

        backend = CapturingBackend("fake")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "compare"
            with patch("tools.ai.compare_music_models.get_backend_registry", return_value={"fake": backend}):
                result = run_comparison(
                    prompt="steady pulse",
                    models=["fake"],
                    output_dir=str(output_dir),
                    duration=1.25,
                    lyrics="we don't let go",
                    tags="electropop, duet",
                    seed=99,
                )

            self.assertTrue(result["success"])
            self.assertEqual(1, len(backend.requests))
            captured_request = backend.requests[0]
            self.assertEqual("steady pulse", captured_request.prompt)
            self.assertEqual("we don't let go", captured_request.lyrics)
            self.assertEqual("electropop, duet", captured_request.tags)
            self.assertEqual(99, captured_request.seed)

            summary_path = Path(result["summary_path"])
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["success"])
            self.assertEqual("steady pulse", summary["prompt"])
            self.assertEqual("we don't let go", summary["lyrics"])
            self.assertEqual("electropop, duet", summary["tags"])
            self.assertEqual(99, summary["seed"])
            self.assertEqual(["fake"], summary["models"])
            self.assertTrue(Path(summary["results"][0]["output_path"]).exists())

    def test_run_comparison_executes_native_melodyflow_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "compare"
            env = {
                "MELODYFLOW_PYTHON": sys.executable,
                "MELODYFLOW_MODEL_DIR": str(Path(temp_dir) / "melodyflow"),
            }
            Path(env["MELODYFLOW_MODEL_DIR"]).mkdir()

            def fake_run_subprocess(command, **_kwargs):
                output_index = command.index("--output") + 1
                Path(command[output_index]).write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            with unittest.mock.patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = run_comparison(
                                prompt="steady pulse",
                                models=["melodyflow"],
                                output_dir=str(output_dir),
                                duration=0.25,
                            )

            self.assertTrue(result["success"])
            self.assertEqual(1, len(result["results"]))
            summary_path = Path(result["summary_path"])
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(["melodyflow"], summary["models"])
            for item in result["results"]:
                self.assertTrue(Path(item["output_path"]).exists())

    def test_run_comparison_executes_named_heartmula_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "compare"
            heart_root = Path(temp_dir) / "heartlib"
            ckpt_dir = Path(temp_dir) / "heartmula-hny"
            heart_root.mkdir()
            ckpt_dir.mkdir()

            def fake_run_subprocess(command, **_kwargs):
                output_index = command.index("--save-path") + 1
                Path(command[output_index]).write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            with unittest.mock.patch.dict(
                os.environ,
                {
                    "HEARTMULA_ROOT": str(heart_root),
                    "HEARTMULA_HNY_CKPT_DIR": str(ckpt_dir),
                    "HEARTMULA_PYTHON": sys.executable,
                },
                clear=False,
            ):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = run_comparison(
                            prompt="steady pulse",
                            models=["heartmula_hny"],
                            output_dir=str(output_dir),
                            duration=0.25,
                        )

            self.assertTrue(result["success"])
            self.assertEqual(1, len(result["results"]))
            self.assertTrue(Path(result["results"][0]["output_path"]).exists())

    def test_headless_setup_checker_matches_gui_preflight(self) -> None:
        checker = Path("tools/ai/check_music_backends.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            stub_root = temp_root / "stub_modules"
            model_dir = temp_root / "melodyflow"
            melodyflow_space_dir = temp_root / "MelodyFlowSpace"
            model_dir.mkdir()
            (melodyflow_space_dir / "audiocraft" / "models").mkdir(parents=True)
            (melodyflow_space_dir / "audiocraft" / "models" / "melodyflow.py").write_text("", encoding="utf-8")

            for package in ["torch", "audiocraft", "xformers", "transformers"]:
                package_dir = stub_root / package
                package_dir.mkdir(parents=True)
                (package_dir / "__init__.py").write_text("", encoding="utf-8")
            for module_name in ["soundfile", "omegaconf", "sentencepiece", "torchdiffeq"]:
                (stub_root / f"{module_name}.py").write_text("", encoding="utf-8")

            env = os.environ.copy()
            env["MELODYFLOW_PYTHON"] = sys.executable
            env["MELODYFLOW_MODEL_DIR"] = str(model_dir)
            env["MELODYFLOW_SPACE_DIR"] = str(melodyflow_space_dir)
            env["PYTHONPATH"] = str(stub_root)

            completed = subprocess.run(
                [sys.executable, str(checker), "--models", "melodyflow"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

        self.assertEqual(0, completed.returncode)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual([], payload["issues"])


if __name__ == "__main__":
    unittest.main()
