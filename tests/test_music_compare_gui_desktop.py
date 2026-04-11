from __future__ import annotations

import json
import os
import sys
import sysconfig
import tempfile
import time
import tkinter as tk
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from tools.ai.music_compare_gui import MusicCompareGui
from tools.ai.music_model_backends import MusicBackendResult


def _write_test_audio(path: Path, *, seconds: float = 1.0, sample_rate: int = 16_000) -> None:
    sample_count = max(int(seconds * sample_rate), 1)
    timeline = np.linspace(0.0, seconds, sample_count, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 220.0 * timeline)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _pump_until(root: tk.Tk, condition, *, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update_idletasks()
        root.update()
        if condition():
            return
        time.sleep(0.01)
    raise TimeoutError("Timed out waiting for desktop GUI condition.")


class _FakeBackend:
    def __init__(self, backend_name: str, *, duration_seconds: float = 1.0) -> None:
        self.backend_name = backend_name
        self.duration_seconds = duration_seconds

    def run(self, request):
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.backend_name}_output.wav"
        _write_test_audio(output_path, seconds=self.duration_seconds)
        return MusicBackendResult(
            backend=self.backend_name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=0.15,
            command=["fake-backend", self.backend_name],
            metadata={
                "prompt": request.prompt,
                "tags": request.tags,
                "lyrics": request.lyrics,
            },
        )


class _CapturingBackend(_FakeBackend):
    def __init__(self, backend_name: str, *, duration_seconds: float = 1.0) -> None:
        super().__init__(backend_name, duration_seconds=duration_seconds)
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return super().run(request)


class _InlineThread:
    def __init__(self, *, target=None, args=(), kwargs=None, daemon=None) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


@unittest.skipUnless(sys.platform == "win32", "Desktop GUI tests are intended for Windows Tkinter runs.")
class MusicCompareGuiDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._configure_tk_environment()
        cls._master_root = tk.Tk()
        cls._master_root.withdraw()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if cls._master_root.winfo_exists():
                cls._master_root.destroy()
        except tk.TclError:
            pass

    @staticmethod
    def _configure_tk_environment() -> None:
        installed_base = Path(sysconfig.get_config_var("installed_base") or sys.base_prefix)
        os.environ["TCL_LIBRARY"] = str(installed_base / "tcl" / "tcl8.6")
        os.environ["TK_LIBRARY"] = str(installed_base / "tcl" / "tk8.6")

    def _build_gui(self, *, registry: dict[str, object] | None = None) -> tuple[tk.Tk, MusicCompareGui]:
        self._configure_tk_environment()
        self.exit_stack = ExitStack()
        self.addCleanup(self.exit_stack.close)
        self.save_settings_mock = Mock()
        self.showerror_mock = Mock()
        self.showwarning_mock = Mock()
        self.showinfo_mock = Mock()
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.load_gui_settings", return_value={}))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.save_gui_settings", new=self.save_settings_mock))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.collect_preflight_issues", return_value=[]))
        self.exit_stack.enter_context(patch.object(MusicCompareGui, "_start_gpu_monitor", lambda self: None))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.threading.Thread", _InlineThread))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.messagebox.showerror", new=self.showerror_mock))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.messagebox.showwarning", new=self.showwarning_mock))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.messagebox.showinfo", new=self.showinfo_mock))
        self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.messagebox.askyesno", return_value=True))
        if registry is not None:
            self.exit_stack.enter_context(patch("tools.ai.music_compare_gui.get_backend_registry", return_value=registry))

        root = tk.Toplevel(self._master_root)
        gui = MusicCompareGui(root)
        self.addCleanup(self._destroy_gui, gui, root)
        root.update_idletasks()
        root.update()
        return root, gui

    @staticmethod
    def _destroy_gui(gui: MusicCompareGui, root: tk.Tk) -> None:
        try:
            if root.winfo_exists():
                gui._on_close()
        except tk.TclError:
            return

    def test_desktop_comparison_run_updates_cards_and_summary(self) -> None:
        registry = {
            "ace_step_v15_turbo": _FakeBackend("ace_step_v15_turbo"),
            "ace_step_v15_sft": _FakeBackend("ace_step_v15_sft"),
        }
        root, gui = self._build_gui(registry=registry)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "compare"
            gui.output_dir_var.set(str(output_dir))
            gui.duration_var.set("1")
            gui.tags_var.set("electronic, test")
            gui.prompt_text.delete("1.0", "end")
            gui.prompt_text.insert("1.0", "deterministic desktop comparison")
            gui.lyrics_text.delete("1.0", "end")
            gui.lyrics_text.insert("1.0", "[instrumental]")

            for backend_name in gui.model_vars:
                gui.model_vars[backend_name].set(backend_name in {"ace_step_v15_turbo", "ace_step_v15_sft"})
            gui._refresh_card_visibility()
            root.update_idletasks()
            root.update()

            gui.run_comparison_button.invoke()
            _pump_until(root, lambda: not gui.running and gui.current_summary_path is not None)

            self.assertIsNotNone(gui.current_summary_path)
            self.assertTrue(gui.current_summary_path.exists())
            self.assertTrue(gui.status_var.get().startswith("Done. Summary:"))

            summary = json.loads(gui.current_summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["success"])
            self.assertEqual(["ace_step_v15_turbo", "ace_step_v15_sft"], summary["models"])

            for backend_name in ("ace_step_v15_turbo", "ace_step_v15_sft"):
                card = gui.model_cards[backend_name]
                self.assertEqual("Completed", card.status_var.get())
                self.assertTrue(card.preview_button.instate(["!disabled"]))
                self.assertTrue(card.open_button.instate(["!disabled"]))
                self.assertIsNotNone(card.result)
                self.assertTrue(Path(card.result["output_path"]).exists())
                self.assertIsNotNone(card.waveform_photo)
                self.assertIsNotNone(card.spectrogram_photo)

    def test_desktop_transcription_run_updates_panel(self) -> None:
        root, gui = self._build_gui(registry={})

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            audio_path = temp_root / "sample_vocals.wav"
            output_dir = temp_root / "transcription"
            output_dir.mkdir()
            heart_python = temp_root / "heart_python.exe"
            checkpoint_root = temp_root / "heart_ckpt"
            summary_path = output_dir / "transcription_summary.json"
            vocal_path = output_dir / "vocals.wav"
            heart_python.write_text("", encoding="utf-8")
            checkpoint_root.mkdir()
            _write_test_audio(audio_path, seconds=0.5)
            _write_test_audio(vocal_path, seconds=0.5)
            summary_path.write_text("{}", encoding="utf-8")

            payload = {
                "summary_path": str(summary_path),
                "vocal_file": str(vocal_path),
                "transcription": {
                    "result": {
                        "text": "desktop transcription result",
                    }
                },
            }

            completed = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
            with patch("tools.ai.music_compare_gui.subprocess.run", return_value=completed):
                gui.transcription_skip_separation_var.set(True)
                gui.transcription_audio_var.set(str(audio_path))
                gui.transcription_output_dir_var.set(str(output_dir))
                gui.heart_python_var.set(str(heart_python))
                gui.heart_hny_ckpt_var.set(str(checkpoint_root))

                gui.transcribe_button.invoke()
                _pump_until(root, lambda: not gui.running and gui.transcription_status_var.get() == "Transcript: completed")

            self.assertEqual("Transcription completed", gui.status_var.get())
            self.assertIn("desktop transcription result", gui.transcription_text.get("1.0", "end"))
            self.assertIn(str(summary_path), gui.transcription_summary_var.get())
            self.assertIn(str(vocal_path), gui.transcription_vocal_var.get())
            self.assertTrue(gui.play_vocals_button.instate(["!disabled"]))
            self.assertTrue(gui.open_transcription_summary_button.instate(["!disabled"]))
            self.assertTrue(gui.open_transcription_dir_button.instate(["!disabled"]))

    def test_preflight_report_success_and_warning_paths(self) -> None:
        root, gui = self._build_gui(registry={})

        gui.model_vars["ace_step_v15_turbo"].set(True)
        gui._show_preflight_report()
        self.showinfo_mock.assert_called_once()
        self.showwarning_mock.assert_not_called()

        self.showinfo_mock.reset_mock()
        self.showwarning_mock.reset_mock()
        with patch.object(gui, "_collect_preflight_issues", return_value=["Missing test dependency"]):
            gui._show_preflight_report()

        self.showinfo_mock.assert_not_called()
        self.showwarning_mock.assert_called_once()
        self.assertIn("Missing test dependency", self.showwarning_mock.call_args.args[1])
        root.update_idletasks()
        root.update()

    def test_single_backend_generate_uses_selected_card_and_persists_ratings(self) -> None:
        backend = _CapturingBackend("ace_step_v15_turbo")
        root, gui = self._build_gui(registry={"ace_step_v15_turbo": backend})

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "single"
            gui.output_dir_var.set(str(output_dir))
            gui.duration_var.set("1")
            gui.seed_var.set("321")
            gui.tags_var.set("solo, regression")
            gui.prompt_text.delete("1.0", "end")
            gui.prompt_text.insert("1.0", "single backend regression")
            gui.lyrics_text.delete("1.0", "end")
            gui.lyrics_text.insert("1.0", "[instrumental]")

            gui._start_single_backend_run("ace_step_v15_turbo")
            _pump_until(root, lambda: not gui.running and gui.current_summary_path is not None)

            self.assertEqual(1, len(backend.requests))
            request = backend.requests[0]
            self.assertEqual("single backend regression", request.prompt)
            self.assertEqual("solo, regression", request.tags)
            self.assertEqual("[instrumental]", request.lyrics)
            self.assertEqual(321, request.seed)

            turbo_card = gui.model_cards["ace_step_v15_turbo"]
            self.assertEqual("Completed", turbo_card.status_var.get())
            self.assertEqual("Idle", gui.model_cards["ace_step_v15_sft"].status_var.get())

            turbo_card.rating_var.set("5")
            turbo_card.notes_text.insert("1.0", "Strong output")
            gui._save_all_ratings_to_summary()

            summary = json.loads(gui.current_summary_path.read_text(encoding="utf-8"))
            self.assertEqual("5", summary["ratings"]["ace_step_v15_turbo"]["rating"])
            self.assertEqual("Strong output", summary["ratings"]["ace_step_v15_turbo"]["notes"])

    def test_clear_text_theme_and_presets_update_state(self) -> None:
        root, gui = self._build_gui(registry={})

        gui.prompt_text.insert("1.0", "hello prompt")
        gui.lyrics_text.insert("1.0", "hello lyrics")
        gui.tags_var.set("tagged")
        gui._clear_text_inputs()
        self.assertEqual("", gui.prompt_text.get("1.0", "end").strip())
        self.assertEqual("", gui.lyrics_text.get("1.0", "end").strip())
        self.assertEqual("", gui.tags_var.get())
        self.assertEqual("Cleared prompt, lyrics, and tags", gui.status_var.get())

        gui._apply_heartmula_fast_preset()
        self.assertEqual("1.5", gui.heart_cfg_scale_var.get())
        self.assertFalse(gui.heart_stage_codec_var.get())
        self.assertFalse(gui.heart_lazy_load_var.get())

        gui._apply_heartmula_low_memory_preset()
        self.assertEqual("1.0", gui.heart_cfg_scale_var.get())
        self.assertEqual("bfloat16", gui.heart_codec_dtype_var.get())
        self.assertTrue(gui.heart_stage_codec_var.get())
        self.assertTrue(gui.heart_lazy_load_var.get())

        gui.theme_var.set("light")
        gui._on_theme_selected()
        self.assertEqual("Theme set to light", gui.status_var.get())
        root.update_idletasks()
        root.update()

    def test_on_close_persists_settings(self) -> None:
        root, gui = self._build_gui(registry={})
        gui.theme_var.set("light")
        gui.prompt_text.delete("1.0", "end")
        gui.prompt_text.insert("1.0", "saved prompt")
        gui._on_close()
        self.save_settings_mock.assert_called_once()
        saved_payload = self.save_settings_mock.call_args.args[0]
        self.assertEqual("light", saved_payload["APP_THEME"])
        self.assertEqual("saved prompt", saved_payload["PROMPT_TEXT"])