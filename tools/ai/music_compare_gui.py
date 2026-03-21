"""
tools/ai/music_compare_gui.py
Simple desktop GUI for comparing multiple local music model backends.

The UI uses a single shared prompt area and one output card per backend.
Each card shows live runtime, peak VRAM, output-vs-runtime metrics,
preview controls, waveform and spectrogram views, and user ratings.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
import warnings
import winsound
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageTk
from scipy import signal

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_SETTINGS_PATH = REPO_ROOT / ".tmp" / "music_compare_gui_settings.json"


def _resolve_existing_path(path: Path) -> str:
    return str(path.resolve()) if path.exists() else ""


def _pick_python_path(preferred_paths: list[Path], fallback: str) -> str:
    for candidate in preferred_paths:
        if candidate.exists():
            return str(candidate.resolve())
    return fallback


def detect_default_backend_settings(
    repo_root: Path,
    environ: Mapping[str, str] | None = None,
    *,
    fallback_python: str | None = None,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    fallback = fallback_python or sys.executable
    defaults = {
        "HEARTMULA_ROOT": env.get("HEARTMULA_ROOT", "").strip(),
        "HEARTMULA_CKPT_DIR": env.get("HEARTMULA_CKPT_DIR", "").strip(),
        "HEARTMULA_HNY_CKPT_DIR": env.get("HEARTMULA_HNY_CKPT_DIR", "").strip(),
        "HEARTMULA_BASE_CKPT_DIR": env.get("HEARTMULA_BASE_CKPT_DIR", "").strip(),
        "HEARTMULA_PYTHON": env.get("HEARTMULA_PYTHON", "").strip(),
        "HEARTMULA_CFG_SCALE": env.get("HEARTMULA_CFG_SCALE", "1.5").strip(),
        "HEARTMULA_LAZY_LOAD": env.get("HEARTMULA_LAZY_LOAD", "true").strip(),
        "HEARTMULA_CODEC_DTYPE": env.get("HEARTMULA_CODEC_DTYPE", "float32").strip(),
        "HEARTMULA_MAX_VRAM_GB": env.get("HEARTMULA_MAX_VRAM_GB", "").strip(),
        "HEARTMULA_STAGE_CODEC": env.get("HEARTMULA_STAGE_CODEC", "false").strip(),
        "AUDIOX_MODEL_ID": env.get("AUDIOX_MODEL_ID", "HKUSTAudio/AudioX-MAF").strip(),
        "AUDIOX_PYTHON": env.get("AUDIOX_PYTHON", "").strip(),
        "MELODYFLOW_MODEL_DIR": env.get("MELODYFLOW_MODEL_DIR", "").strip(),
        "MELODYFLOW_PYTHON": env.get("MELODYFLOW_PYTHON", "").strip(),
        "ACESTEP_CKPT_DIR": env.get("ACESTEP_CKPT_DIR", "").strip(),
        "ACESTEP_PYTHON": env.get("ACESTEP_PYTHON", "").strip(),
        "ACESTEP15_ROOT": env.get("ACESTEP15_ROOT", "").strip(),
        "ACESTEP15_CKPT_DIR": env.get("ACESTEP15_CKPT_DIR", "").strip(),
        "ACESTEP15_PYTHON": env.get("ACESTEP15_PYTHON", "").strip(),
    }

    if not defaults["HEARTMULA_ROOT"]:
        defaults["HEARTMULA_ROOT"] = _resolve_existing_path(repo_root / "third_party" / "heartlib")
    if not defaults["HEARTMULA_HNY_CKPT_DIR"]:
        defaults["HEARTMULA_HNY_CKPT_DIR"] = _resolve_existing_path(repo_root / "models" / "heartmula" / "happy-new-year")
    if not defaults["HEARTMULA_HNY_CKPT_DIR"]:
        defaults["HEARTMULA_HNY_CKPT_DIR"] = defaults["HEARTMULA_CKPT_DIR"] or _resolve_existing_path(repo_root / "models" / "heartmula" / "ckpt")
    if not defaults["HEARTMULA_BASE_CKPT_DIR"]:
        defaults["HEARTMULA_BASE_CKPT_DIR"] = _resolve_existing_path(repo_root / "models" / "heartmula" / "base")
    if not defaults["HEARTMULA_CKPT_DIR"]:
        defaults["HEARTMULA_CKPT_DIR"] = defaults["HEARTMULA_HNY_CKPT_DIR"]
    if not defaults["HEARTMULA_PYTHON"]:
        defaults["HEARTMULA_PYTHON"] = _pick_python_path([repo_root / ".venv-heartmula" / "Scripts" / "python.exe"], fallback)

    if not defaults["AUDIOX_PYTHON"]:
        defaults["AUDIOX_PYTHON"] = _pick_python_path(
            [
                repo_root / ".venv-audiox310" / "Scripts" / "python.exe",
                repo_root / ".venv-audiox" / "Scripts" / "python.exe",
            ],
            fallback,
        )

    if not defaults["MELODYFLOW_MODEL_DIR"]:
        defaults["MELODYFLOW_MODEL_DIR"] = _resolve_existing_path(repo_root / "models" / "comparison" / "melodyflow" / "melodyflow-t24-30secs")
    if not defaults["MELODYFLOW_PYTHON"]:
        defaults["MELODYFLOW_PYTHON"] = _pick_python_path([repo_root / ".venv-melodyflow" / "Scripts" / "python.exe"], fallback)

    if not defaults["ACESTEP_CKPT_DIR"]:
        defaults["ACESTEP_CKPT_DIR"] = _resolve_existing_path(repo_root / "models" / "comparison" / "ace-step" / "ACE-Step-v1-3.5B")
    if not defaults["ACESTEP_PYTHON"]:
        defaults["ACESTEP_PYTHON"] = _pick_python_path([repo_root / ".venv-acestep" / "Scripts" / "python.exe"], fallback)
    if not defaults["ACESTEP15_ROOT"]:
        defaults["ACESTEP15_ROOT"] = _resolve_existing_path(repo_root / "third_party" / "ACE-Step-1.5")
    if not defaults["ACESTEP15_CKPT_DIR"]:
        defaults["ACESTEP15_CKPT_DIR"] = _resolve_existing_path(repo_root / "models" / "comparison" / "ace-step-1.5" / "checkpoints")
    if not defaults["ACESTEP15_CKPT_DIR"] and defaults["ACESTEP15_ROOT"]:
        defaults["ACESTEP15_CKPT_DIR"] = _resolve_existing_path(Path(defaults["ACESTEP15_ROOT"]) / "checkpoints")
    if not defaults["ACESTEP15_PYTHON"]:
        defaults["ACESTEP15_PYTHON"] = _pick_python_path([repo_root / ".venv-acestep15" / "Scripts" / "python.exe"], fallback)

    return defaults


def load_gui_settings(settings_path: Path = GUI_SETTINGS_PATH) -> dict[str, str]:
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if isinstance(value, (str, int, float))}


def save_gui_settings(settings: Mapping[str, str], settings_path: Path = GUI_SETTINGS_PATH) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(dict(settings), indent=2), encoding="utf-8")


def resolve_ffmpeg_binary() -> str | None:
    env_value = os.environ.get("FFMPEG_BINARY", "").strip()
    if env_value and Path(env_value).exists():
        return env_value

    repo_root = Path(__file__).resolve().parents[2]
    bundled_candidates = [
        repo_root / "bin" / "ffmpeg" / "ffmpeg.exe",
        repo_root / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for candidate in bundled_candidates:
        if candidate.exists():
            return str(candidate)

    system_ffmpeg = which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def configure_ffmpeg() -> str | None:
    ffmpeg_binary = resolve_ffmpeg_binary()
    if not ffmpeg_binary:
        return None

    os.environ["FFMPEG_BINARY"] = ffmpeg_binary
    ffmpeg_dir = str(Path(ffmpeg_binary).parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work",
                category=RuntimeWarning,
            )
            from pydub import AudioSegment

        AudioSegment.converter = ffmpeg_binary
        ffprobe_candidate = Path(ffmpeg_binary).with_name("ffprobe.exe")
        if ffprobe_candidate.exists():
            AudioSegment.ffprobe = str(ffprobe_candidate)
    except Exception:
        pass

    return ffmpeg_binary


FFMPEG_BINARY = configure_ffmpeg()

from tools.ai.music_backend_checks import collect_preflight_issues
from tools.ai.music_model_backends import MusicGenRequest, get_backend_registry


DEFAULT_OUTPUT_DIR = ".tmp/music_compare/gui"
DEFAULT_TRANSCRIPTION_OUTPUT_DIR = ".tmp/music_compare/transcriptions"
DEFAULT_MELODYFLOW_MODEL_DIR = str((REPO_ROOT / "models" / "comparison" / "melodyflow" / "melodyflow-t24-30secs").resolve())
DEFAULT_ACESTEP_MODEL_DIR = str((REPO_ROOT / "models" / "comparison" / "ace-step" / "ACE-Step-v1-3.5B").resolve())
DEFAULT_ACESTEP15_ROOT = str((REPO_ROOT / "third_party" / "ACE-Step-1.5").resolve())
DEFAULT_ACESTEP15_CKPT_DIR = str((REPO_ROOT / "models" / "comparison" / "ace-step-1.5" / "checkpoints").resolve())
DEFAULT_SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"
HEARTMULA_CODEC_DTYPE_OPTIONS = ["float32", "bfloat16"]
THEME_OPTIONS = ["dark", "light"]
MODEL_OPTIONS = [
    ("heartmula_hny", "HeartMuLa Happy New Year"),
    ("heartmula_base", "HeartMuLa Base 3B"),
    ("melodyflow", "MelodyFlow"),
    ("ace_step", "ACE-Step v1-3.5B"),
    ("ace_step_v15", "ACE-Step 1.5"),
]
BACKEND_FIELD_HINTS = {
    "heartmula_hny": (
        "Valid inputs: prompt/chat and/or tags and/or lyrics. Prompt/chat and tags are merged into HeartMuLa's descriptor input. "
        "Reference audio is ignored."
    ),
    "heartmula_base": (
        "Valid inputs: prompt/chat and/or tags and/or lyrics. Prompt/chat and tags are merged into HeartMuLa's descriptor input. "
        "Reference audio is ignored."
    ),
    "melodyflow": "Valid inputs: prompt/chat. Reference audio editing is not exposed here yet. Lyrics and tags are ignored.",
    "ace_step": "Valid inputs: prompt/chat and/or tags and/or lyrics. Prompt/chat remains primary and tags are appended to the descriptor input. Reference audio is ignored.",
    "ace_step_v15": "Valid inputs: prompt/chat and/or tags and/or lyrics. Prompt/chat remains primary and tags are appended to the descriptor input. Reference audio is ignored.",
}
BACKEND_OUTPUT_FILENAMES = {
    "heartmula_hny": "heartmula_hny_output.wav",
    "heartmula_base": "heartmula_base_output.wav",
    "melodyflow": "melodyflow_output.wav",
    "ace_step": "ace_step_output.wav",
    "ace_step_v15": "ace_step_v15_output.wav",
}
RATING_OPTIONS = ["", "1", "2", "3", "4", "5"]
VISUAL_PANEL_SIZE = (360, 180)
WAVEFORM_SIZE = (VISUAL_PANEL_SIZE[0], 82)
SPECTROGRAM_SIZE = (VISUAL_PANEL_SIZE[0], VISUAL_PANEL_SIZE[1] - WAVEFORM_SIZE[1] - 8)
READY_BORDER_COLOR = "#2e8b57"
RESIZING_BORDER_COLOR = "#d18f00"
DEFAULT_BORDER_COLOR = "#4b5563"
RESIZE_HIT_ZONE = 10
APP_MIN_WIDTH = 860
APP_MIN_HEIGHT = 620
CARD_SINGLE_COLUMN_WIDTH = 980
CARD_TWO_COLUMN_WIDTH = 1320
CARD_THREE_COLUMN_WIDTH = 1820
CARD_FOUR_COLUMN_WIDTH = 2360
CARD_MIN_WRAP = 220
CARD_MAX_WRAP = 520
THEME_PALETTES = {
    "dark": {
        "root_bg": "#101418",
        "surface": "#171d24",
        "surface_alt": "#1e2630",
        "text": "#edf2f7",
        "muted": "#a7b4c5",
        "entry_bg": "#0f1419",
        "entry_fg": "#edf2f7",
        "border": "#2b3744",
        "accent": "#4fb3ff",
        "accent_active": "#75c5ff",
        "button_bg": "#24303b",
        "button_active": "#2c3946",
        "canvas_bg": "#11171d",
        "border_ready": READY_BORDER_COLOR,
        "spectrogram_bg": (11, 15, 20),
        "spectrogram_grid": (58, 72, 89),
        "spectrogram_text": (225, 232, 240),
    },
    "light": {
        "root_bg": "#edf2f7",
        "surface": "#ffffff",
        "surface_alt": "#f4f7fb",
        "text": "#18212b",
        "muted": "#556270",
        "entry_bg": "#ffffff",
        "entry_fg": "#18212b",
        "border": "#c7d1db",
        "accent": "#0b79d0",
        "accent_active": "#1f8de0",
        "button_bg": "#e7eef5",
        "button_active": "#d9e4ef",
        "canvas_bg": "#eef3f8",
        "border_ready": READY_BORDER_COLOR,
        "spectrogram_bg": (245, 248, 252),
        "spectrogram_grid": (186, 197, 209),
        "spectrogram_text": (31, 41, 55),
    },
}


def active_theme_palette(theme_name: str) -> dict[str, object]:
    return THEME_PALETTES.get(theme_name, THEME_PALETTES["dark"])


def format_frequency_label(frequency_hz: float) -> str:
    if frequency_hz >= 1000:
        value = frequency_hz / 1000.0
        return f"{value:.0f}k" if value >= 10 else f"{value:.1f}k"
    return f"{int(round(frequency_hz))}"


def build_frequency_ticks(max_frequency_hz: float) -> list[float]:
    candidates = [0, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000]
    ticks = [tick for tick in candidates if tick <= max_frequency_hz]
    if not ticks:
        ticks = [0.0, max_frequency_hz]
    elif ticks[-1] < max_frequency_hz:
        ticks.append(max_frequency_hz)
    return sorted(set(float(tick) for tick in ticks))


def default_separator_python_path(repo_root: Path, fallback: str) -> str:
    return _pick_python_path([repo_root / ".venv-separator310" / "Scripts" / "python.exe"], fallback)


def discover_sample_audio_files(root: Path) -> list[str]:
    search_root = root / "assets" / "music samples"
    if not search_root.exists():
        return []
    paths = [
        candidate.resolve()
        for candidate in search_root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    ]
    return [str(path) for path in sorted(paths, key=lambda item: item.name.lower())]


def extract_json_payload(stdout: str, stderr: str = "") -> dict:
    content = stdout.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start != -1:
            try:
                return json.loads(content[start:])
            except json.JSONDecodeError:
                pass
    detail = stderr.strip() or content or "Command did not return JSON."
    raise RuntimeError(detail)


def extract_transcription_text(payload: Mapping[str, object] | None) -> str:
    if not payload:
        return ""
    transcription = payload.get("transcription")
    if not isinstance(transcription, Mapping):
        return ""
    result = transcription.get("result")
    if isinstance(result, Mapping):
        text = result.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def query_gpu_stats() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"used_mb": "N/A", "total_mb": "N/A", "util": "N/A"}
        first = result.stdout.strip().splitlines()[0]
        used_mb, total_mb, util = [part.strip() for part in first.split(",")[:3]]
        return {"used_mb": used_mb, "total_mb": total_mb, "util": util}
    except Exception:
        return {"used_mb": "N/A", "total_mb": "N/A", "util": "N/A"}


def safe_audio_duration_seconds(audio_path: str) -> float | None:
    try:
        info = sf.info(audio_path)
        if info.duration:
            return float(info.duration)
    except Exception:
        pass

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(audio_path)
        return len(segment) / 1000.0
    except Exception:
        return None


def expected_backend_output_path(output_dir: str | Path, backend: str) -> Path | None:
    filename = BACKEND_OUTPUT_FILENAMES.get(backend)
    if not filename:
        return None
    return Path(output_dir).resolve() / filename


def live_generated_audio_seconds(audio_path: str | Path | None, *, started_epoch: float | None) -> float | None:
    if not audio_path or started_epoch is None:
        return None

    candidate = Path(audio_path)
    if not candidate.exists():
        return None

    try:
        modified_epoch = candidate.stat().st_mtime
    except OSError:
        return None

    if modified_epoch + 0.01 < started_epoch:
        return None

    return safe_audio_duration_seconds(str(candidate))


def load_audio_mono(audio_path: str) -> tuple[np.ndarray, int] | None:
    try:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    except Exception:
        pass

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(audio_path)
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels)).mean(axis=1)
        max_value = float(1 << (8 * segment.sample_width - 1))
        if max_value > 0:
            samples = samples / max_value
        return samples, int(segment.frame_rate)
    except Exception:
        return None


def generate_spectrogram_image(audio_path: str, theme_name: str = "dark") -> Image.Image | None:
    loaded = load_audio_mono(audio_path)
    if not loaded:
        return None

    samples, sample_rate = loaded
    if samples.size == 0:
        return None

    max_samples = sample_rate * 45
    if samples.size > max_samples:
        samples = samples[:max_samples]

    nperseg = min(2048, max(256, samples.size // 8))
    noverlap = min(max(nperseg // 2, 128), nperseg - 1)
    frequencies, _times, spectrum = signal.spectrogram(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    if spectrum.size == 0:
        return None

    magnitude = np.log10(np.maximum(spectrum, 1e-10))
    magnitude -= magnitude.min()
    peak = magnitude.max()
    if peak > 0:
        magnitude /= peak
    magnitude = np.flipud(magnitude)

    rgb = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(magnitude * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(np.sqrt(magnitude) * 220, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((1.0 - magnitude) * 180 + magnitude * 40, 0, 255).astype(np.uint8)

    palette = active_theme_palette(theme_name)
    plot_width = max(SPECTROGRAM_SIZE[0] - 52, 40)
    plot_height = max(SPECTROGRAM_SIZE[1] - 20, 40)
    image = Image.fromarray(rgb, mode="RGB").resize((plot_width, plot_height), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", SPECTROGRAM_SIZE, palette["spectrogram_bg"])
    canvas.paste(image, (46, 8))
    draw = ImageDraw.Draw(canvas)
    nyquist = float(frequencies[-1]) if frequencies.size else sample_rate / 2.0
    ticks = build_frequency_ticks(nyquist)
    top = 8
    left = 46
    right = left + plot_width
    bottom = top + plot_height

    for tick_hz in ticks:
        ratio = 0.0 if nyquist <= 0 else min(max(tick_hz / nyquist, 0.0), 1.0)
        y = int(round(bottom - ratio * plot_height))
        draw.line((left, y, right, y), fill=palette["spectrogram_grid"], width=1)
        label = format_frequency_label(tick_hz)
        draw.text((4, max(y - 7, 0)), label, fill=palette["spectrogram_text"])

    draw.rectangle((left, top, right, bottom), outline=palette["spectrogram_grid"], width=1)
    draw.text((left, 0), f"0-{format_frequency_label(nyquist)}Hz", fill=palette["spectrogram_text"])
    return canvas


def generate_waveform_image(audio_path: str) -> Image.Image | None:
    loaded = load_audio_mono(audio_path)
    if not loaded:
        return None

    samples, _sample_rate = loaded
    if samples.size == 0:
        return None

    max_columns = WAVEFORM_SIZE[0]
    if samples.size < max_columns:
        padded = np.zeros(max_columns, dtype=np.float32)
        padded[: samples.size] = samples
        samples = padded

    bins = np.array_split(samples, max_columns)
    envelope = np.array([np.max(np.abs(chunk)) if len(chunk) else 0.0 for chunk in bins], dtype=np.float32)
    peak = float(envelope.max()) if envelope.size else 0.0
    if peak > 0:
        envelope = envelope / peak

    height = WAVEFORM_SIZE[1]
    center = height // 2
    rgb = np.zeros((height, max_columns, 3), dtype=np.uint8)
    rgb[:] = np.array([20, 24, 30], dtype=np.uint8)

    for x, magnitude in enumerate(envelope):
        half_span = int(round(magnitude * max(center - 4, 1)))
        top = max(center - half_span, 0)
        bottom = min(center + half_span + 1, height)
        rgb[top:bottom, x] = np.array([99, 210, 255], dtype=np.uint8)

    rgb[max(center - 1, 0):min(center + 1, height), :] = np.array([52, 61, 72], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize(WAVEFORM_SIZE, Image.Resampling.BILINEAR)


def play_audio_preview(audio_path: str) -> None:
    suffix = Path(audio_path).suffix.lower()
    if os.name == "nt" and suffix == ".wav":
        winsound.PlaySound(audio_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
        return
    if os.name == "nt":
        os.startfile(audio_path)  # type: ignore[attr-defined]
        return
    raise RuntimeError("Audio preview is only implemented for Windows in this GUI.")


def stop_audio_preview() -> None:
    if os.name == "nt":
        winsound.PlaySound(None, winsound.SND_PURGE)


@dataclass
class ModelCardState:
    backend: str
    label: str
    frame: ttk.LabelFrame
    status_var: tk.StringVar
    runtime_var: tk.StringVar
    output_var: tk.StringVar
    output_label: ttk.Label
    progress_var: tk.StringVar
    vram_var: tk.StringVar
    rating_var: tk.StringVar
    notes_text: tk.Text
    progressbar: ttk.Progressbar
    waveform_label: ttk.Label
    spectrogram_label: ttk.Label
    generate_button: ttk.Button
    preview_button: ttk.Button
    open_button: ttk.Button
    waveform_photo: ImageTk.PhotoImage | None = None
    spectrogram_photo: ImageTk.PhotoImage | None = None
    result: dict | None = None
    started_at: float | None = None
    started_epoch: float | None = None
    target_audio_seconds: float = 0.0
    output_path_hint: str | None = None
    completed: bool = False
    peak_vram_mb: int = 0


@dataclass
class ScrollAreaState:
    canvas: tk.Canvas
    container: ttk.Frame
    scrollbar: ttk.Scrollbar
    window_id: int


class HoverTooltip:
    def __init__(self, root: tk.Tk, text: str):
        self.root = root
        self.text = text
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None
        self._label: tk.Label | None = None

    def schedule(self, event=None) -> None:
        self.cancel()
        x_root = event.x_root if event is not None else self.root.winfo_pointerx()
        y_root = event.y_root if event is not None else self.root.winfo_pointery()
        self._after_id = self.root.after(300, lambda x=x_root, y=y_root: self.show(x, y))

    def cancel(self, _event=None) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.hide()

    def show(self, x_root: int, y_root: int) -> None:
        self.hide()
        self._after_id = None
        window = tk.Toplevel(self.root)
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        label = tk.Label(
            window,
            text=self.text,
            justify="left",
            wraplength=340,
            padx=10,
            pady=6,
            bg="#fff6d8",
            fg="#1f2933",
            relief="solid",
            bd=1,
        )
        label.pack()
        window.geometry(f"+{x_root + 14}+{y_root + 18}")
        window.deiconify()
        self._window = window
        self._label = label

    def hide(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None
            self._label = None


class MusicCompareGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Music Model Compare")
        self.root.geometry("1440x980")
        self.root.minsize(APP_MIN_WIDTH, APP_MIN_HEIGHT)
        self.style = ttk.Style(self.root)

        detected_settings = detect_default_backend_settings(REPO_ROOT, os.environ, fallback_python=sys.executable)
        saved_settings = load_gui_settings()
        if saved_settings.get("HEARTMULA_CKPT_DIR", "").strip() and not saved_settings.get("HEARTMULA_HNY_CKPT_DIR", "").strip():
            saved_settings["HEARTMULA_HNY_CKPT_DIR"] = saved_settings["HEARTMULA_CKPT_DIR"]
        initial_settings = {**detected_settings, **{key: value for key, value in saved_settings.items() if value.strip()}}

        self.running = False
        self.monitor_stop = threading.Event()
        self.global_peak_vram_mb = 0
        self.current_backend_name: str | None = None
        self.current_summary_path: Path | None = None

        self.duration_var = tk.StringVar(value="15")
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.reference_audio_var = tk.StringVar()
        self.seed_var = tk.StringVar()
        self.tags_var = tk.StringVar()
        self.heart_root_var = tk.StringVar(value=initial_settings.get("HEARTMULA_ROOT", ""))
        self.heart_hny_ckpt_var = tk.StringVar(value=initial_settings.get("HEARTMULA_HNY_CKPT_DIR", initial_settings.get("HEARTMULA_CKPT_DIR", "")))
        self.heart_base_ckpt_var = tk.StringVar(value=initial_settings.get("HEARTMULA_BASE_CKPT_DIR", ""))
        self.heart_python_var = tk.StringVar(value=initial_settings.get("HEARTMULA_PYTHON", sys.executable))
        self.heart_cfg_scale_var = tk.StringVar(value=initial_settings.get("HEARTMULA_CFG_SCALE", "1.5"))
        self.heart_lazy_load_var = tk.BooleanVar(value=initial_settings.get("HEARTMULA_LAZY_LOAD", "true").strip().lower() in {"1", "true", "yes", "on"})
        self.heart_codec_dtype_var = tk.StringVar(value=initial_settings.get("HEARTMULA_CODEC_DTYPE", "float32"))
        self.heart_max_vram_gb_var = tk.StringVar(value=initial_settings.get("HEARTMULA_MAX_VRAM_GB", ""))
        self.heart_stage_codec_var = tk.BooleanVar(value=initial_settings.get("HEARTMULA_STAGE_CODEC", "false").strip().lower() in {"1", "true", "yes", "on"})
        self.melodyflow_model_var = tk.StringVar(value=initial_settings.get("MELODYFLOW_MODEL_DIR", DEFAULT_MELODYFLOW_MODEL_DIR))
        self.melodyflow_python_var = tk.StringVar(value=initial_settings.get("MELODYFLOW_PYTHON", sys.executable))
        self.acestep_ckpt_var = tk.StringVar(value=initial_settings.get("ACESTEP_CKPT_DIR", DEFAULT_ACESTEP_MODEL_DIR))
        self.acestep_python_var = tk.StringVar(value=initial_settings.get("ACESTEP_PYTHON", sys.executable))
        self.acestep15_root_var = tk.StringVar(value=initial_settings.get("ACESTEP15_ROOT", DEFAULT_ACESTEP15_ROOT))
        self.acestep15_ckpt_var = tk.StringVar(value=initial_settings.get("ACESTEP15_CKPT_DIR", DEFAULT_ACESTEP15_CKPT_DIR))
        self.acestep15_python_var = tk.StringVar(value=initial_settings.get("ACESTEP15_PYTHON", sys.executable))
        sample_audio_files = discover_sample_audio_files(REPO_ROOT)
        default_sample_audio = sample_audio_files[0] if sample_audio_files else ""
        self.sample_audio_files = sample_audio_files
        self.sample_audio_var = tk.StringVar(value=initial_settings.get("TRANSCRIPTION_AUDIO_FILE", default_sample_audio))
        self.transcription_audio_var = tk.StringVar(value=initial_settings.get("TRANSCRIPTION_AUDIO_FILE", default_sample_audio))
        self.transcription_output_dir_var = tk.StringVar(value=initial_settings.get("TRANSCRIPTION_OUTPUT_DIR", DEFAULT_TRANSCRIPTION_OUTPUT_DIR))
        self.separator_model_var = tk.StringVar(value=initial_settings.get("SEPARATOR_MODEL", DEFAULT_SEPARATOR_MODEL))
        self.separator_python_var = tk.StringVar(value=initial_settings.get("SEPARATOR_PYTHON", default_separator_python_path(REPO_ROOT, sys.executable)))
        self.transcription_language_var = tk.StringVar(value=initial_settings.get("TRANSCRIPTION_LANGUAGE", ""))
        self.transcription_skip_separation_var = tk.BooleanVar(value=initial_settings.get("TRANSCRIPTION_SKIP_SEPARATION", "false").strip().lower() in {"1", "true", "yes", "on"})

        self.model_vars = {
            name: tk.BooleanVar(value=name != "ace_step_v15") for name, _ in MODEL_OPTIONS
        }
        self.status_var = tk.StringVar(value="Ready")
        self.global_speed_var = tk.StringVar(value="Runtime: idle")
        self.global_vram_var = tk.StringVar(value="VRAM: N/A")
        self.current_run_var = tk.StringVar(value="Current model: none")
        self.transcription_status_var = tk.StringVar(value="Transcript: idle")
        self.transcription_runtime_var = tk.StringVar(value="Runtime: --")
        self.transcription_summary_var = tk.StringVar(value="Summary: --")
        self.transcription_vocal_var = tk.StringVar(value="Vocal stem: --")
        self.resize_state_var = tk.StringVar(value="Resize: ready")
        self.theme_var = tk.StringVar(value=(initial_settings.get("APP_THEME", "dark").strip().lower() or "dark"))
        self.initial_prompt_text = initial_settings.get("PROMPT_TEXT", "")
        self.initial_lyrics_text = initial_settings.get("LYRICS_TEXT", "")
        self.tags_var.set(initial_settings.get("TAGS_TEXT", ""))
        self.current_transcription_summary_path: Path | None = None
        self.current_transcription_vocal_path: str | None = None
        self._resize_edge_active = False
        self._card_columns = 2
        self.initial_config_tab = initial_settings.get("CONFIG_PANEL_TAB", "transcription").strip().lower() or "transcription"

        self.model_cards: dict[str, ModelCardState] = {}
        self.config_scroll_areas: list[ScrollAreaState] = []
        self.hover_tooltips: list[HoverTooltip] = []

        self._build_ui()
        self._apply_theme()
        self._start_gpu_monitor()
        self.root.bind("<Motion>", self._handle_root_motion)
        self.root.bind("<Leave>", self._handle_root_leave)
        self.root.bind_all("<MouseWheel>", self._handle_global_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._handle_global_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._handle_global_mousewheel, add="+")
        self._mark_resize_ready()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.shell_frame = tk.Frame(
            self.root,
            bg=DEFAULT_BORDER_COLOR,
            highlightthickness=4,
            highlightbackground=DEFAULT_BORDER_COLOR,
            highlightcolor=DEFAULT_BORDER_COLOR,
            bd=0,
        )
        self.shell_frame.grid(row=0, column=0, sticky="nsew")
        self.shell_frame.columnconfigure(0, weight=1)
        self.shell_frame.rowconfigure(3, weight=1)

        top = ttk.Frame(self.shell_frame, padding=12)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)
        top.columnconfigure(5, weight=1)

        ttk.Label(top, text="Prompt / Chat").grid(row=0, column=0, sticky="nw", padx=(0, 8))
        self.prompt_text = tk.Text(top, height=6, wrap="word")
        self.prompt_text.grid(row=0, column=1, columnspan=5, sticky="nsew")
        if self.initial_prompt_text:
            self.prompt_text.insert("1.0", self.initial_prompt_text)

        ttk.Label(top, text="Lyrics (optional)").grid(row=1, column=0, sticky="nw", pady=(10, 0), padx=(0, 8))
        self.lyrics_text = tk.Text(top, height=4, wrap="word")
        self.lyrics_text.grid(row=1, column=1, columnspan=5, sticky="nsew", pady=(10, 0))
        if self.initial_lyrics_text:
            self.lyrics_text.insert("1.0", self.initial_lyrics_text)

        ttk.Label(top, text="Duration (s)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.duration_var, width=10).grid(row=2, column=1, sticky="w", pady=(10, 0))

        ttk.Label(top, text="Seed").grid(row=2, column=2, sticky="e", pady=(10, 0))
        ttk.Entry(top, textvariable=self.seed_var, width=12).grid(row=2, column=3, sticky="w", pady=(10, 0))

        ttk.Label(top, text="Tags").grid(row=2, column=4, sticky="e", pady=(10, 0))
        ttk.Entry(top, textvariable=self.tags_var).grid(row=2, column=5, sticky="ew", pady=(10, 0))

        ttk.Label(top, text="Output Dir").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.output_dir_var).grid(row=3, column=1, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(top, text="Browse", command=self._browse_output_dir).grid(row=3, column=5, sticky="e", pady=(10, 0))

        ttk.Label(top, text="Reference Audio").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.reference_audio_var).grid(row=4, column=1, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(top, text="Browse", command=self._browse_reference_audio).grid(row=4, column=5, sticky="e", pady=(10, 0))

        controls = ttk.LabelFrame(self.shell_frame, text="Shared Controls", padding=12)
        controls.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        controls.columnconfigure(5, weight=1)

        column = 0
        for name, label in MODEL_OPTIONS:
            ttk.Checkbutton(controls, text=label, variable=self.model_vars[name], command=self._refresh_card_visibility).grid(row=0, column=column, sticky="w", padx=(0, 16))
            column += 1

        ttk.Button(controls, text="Check Setup", command=self._show_preflight_report).grid(row=0, column=column, sticky="ew", padx=(8, 0))
        column += 1
        self.run_comparison_button = ttk.Button(controls, text="Run Comparison", command=self._start_run)
        self.run_comparison_button.grid(row=0, column=column, sticky="ew", padx=(8, 0))
        column += 1
        ttk.Button(controls, text="Stop Audio", command=stop_audio_preview).grid(row=0, column=column, sticky="ew", padx=(8, 0))
        column += 1
        ttk.Button(controls, text="Clear Text", command=self._clear_text_inputs).grid(row=0, column=column, sticky="ew", padx=(8, 0))

        status_bar = ttk.Frame(self.shell_frame, padding=(12, 0, 12, 8))
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.columnconfigure(1, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_bar, textvariable=self.current_run_var).grid(row=0, column=1, sticky="w", padx=(16, 16))
        ttk.Label(status_bar, textvariable=self.global_speed_var).grid(row=0, column=2, sticky="e", padx=(0, 16))
        ttk.Label(status_bar, textvariable=self.global_vram_var).grid(row=0, column=3, sticky="e")
        ttk.Label(status_bar, textvariable=self.resize_state_var).grid(row=0, column=4, sticky="e", padx=(16, 0))
        ttk.Label(status_bar, text="Theme").grid(row=0, column=5, sticky="e", padx=(16, 6))
        theme_combo = ttk.Combobox(status_bar, textvariable=self.theme_var, values=THEME_OPTIONS, state="readonly", width=8)
        theme_combo.grid(row=0, column=6, sticky="e")
        theme_combo.bind("<<ComboboxSelected>>", self._on_theme_selected)
        self.theme_combo = theme_combo

        main = ttk.Panedwindow(self.shell_frame, orient=tk.HORIZONTAL)
        main.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.main_pane = main

        config_panel = ttk.Frame(main)
        config_panel.columnconfigure(0, weight=1)
        config_panel.rowconfigure(0, weight=1)
        main.add(config_panel, weight=1)
        config_tabs = ttk.Notebook(config_panel)
        config_tabs.grid(row=0, column=0, sticky="nsew")
        self.config_tabs = config_tabs

        backend_tab = ttk.Frame(config_tabs)
        backend_tab.columnconfigure(0, weight=1)
        backend_tab.rowconfigure(0, weight=1)
        transcription_tab = ttk.Frame(config_tabs)
        transcription_tab.columnconfigure(0, weight=1)
        transcription_tab.rowconfigure(0, weight=1)
        config_tabs.add(backend_tab, text="Backends")
        config_tabs.add(transcription_tab, text="Transcription")
        self.backend_tab = backend_tab
        self.transcription_tab = transcription_tab

        backend_scroll_area = self._create_scrollable_panel(backend_tab)
        transcription_scroll_area = self._create_scrollable_panel(transcription_tab)
        self._build_backend_config(backend_scroll_area.container)
        self._build_transcription_panel(transcription_scroll_area.container)
        config_tabs.select(transcription_tab if self.initial_config_tab == "transcription" else backend_tab)

        cards_panel = ttk.Frame(main)
        cards_panel.columnconfigure(0, weight=1)
        cards_panel.rowconfigure(0, weight=1)
        main.add(cards_panel, weight=3)

        canvas = tk.Canvas(cards_panel, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(cards_panel, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_container = ttk.Frame(canvas)
        for column in range(len(MODEL_OPTIONS)):
            self.cards_container.columnconfigure(column, weight=1 if column < 2 else 0)
        self.cards_window = canvas.create_window((0, 0), window=self.cards_container, anchor="nw")
        self.cards_canvas = canvas
        self.cards_scrollbar = scrollbar

        self.cards_container.bind("<Configure>", self._handle_cards_container_configure)
        canvas.bind("<Configure>", self._handle_cards_canvas_configure)

        self._build_model_cards()
        self._refresh_card_visibility()

        footer = ttk.Frame(self.shell_frame, padding=(12, 0, 12, 12))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, text="Amber border means the pointer is on a resize edge. Hover a model card to see which inputs it uses. Both panes support mouse-wheel scrolling when content exceeds the visible area.").grid(row=0, column=0, sticky="w")
        ttk.Sizegrip(footer).grid(row=0, column=1, sticky="se")

    def _create_scrollable_panel(self, parent: ttk.Frame) -> ScrollAreaState:
        canvas = tk.Canvas(parent, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        container = ttk.Frame(canvas, padding=4)
        window_id = canvas.create_window((0, 0), window=container, anchor="nw")
        area = ScrollAreaState(canvas=canvas, container=container, scrollbar=scrollbar, window_id=window_id)
        self.config_scroll_areas.append(area)

        container.bind("<Configure>", lambda _event, scroll_area=area: self._handle_scrollable_container_configure(scroll_area))
        canvas.bind("<Configure>", lambda event, scroll_area=area: self._handle_scrollable_canvas_configure(scroll_area, event))
        return area

    def _build_backend_config(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Backend Config", padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        self._add_labeled_entry(frame, 0, "HeartMuLa Root", self.heart_root_var, self._browse_heart_root)
        self._add_labeled_entry(frame, 1, "HeartMuLa Happy New Year", self.heart_hny_ckpt_var, self._browse_heart_hny_ckpt)
        self._add_labeled_entry(frame, 2, "HeartMuLa Base 3B", self.heart_base_ckpt_var, self._browse_heart_base_ckpt)
        self._add_labeled_entry(frame, 3, "HeartMuLa Python", self.heart_python_var, self._browse_heart_python)
        self._add_labeled_entry(frame, 4, "HeartMuLa CFG Scale", self.heart_cfg_scale_var)
        self._add_labeled_combobox(frame, 5, "HeartCodec Dtype", self.heart_codec_dtype_var, HEARTMULA_CODEC_DTYPE_OPTIONS)
        self._add_labeled_entry(frame, 6, "HeartMuLa VRAM Limit (GB)", self.heart_max_vram_gb_var)
        ttk.Checkbutton(frame, text="HeartMuLa Resident Mode (disable lazy load)", variable=self.heart_lazy_load_var, onvalue=False, offvalue=True).grid(row=7, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Checkbutton(frame, text="HeartMuLa Staged Decode (CPU frames, unload HeartMuLa, then load codec)", variable=self.heart_stage_codec_var).grid(row=8, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(frame, text="Apply HeartMuLa Fast Preset", command=self._apply_heartmula_fast_preset).grid(row=9, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Button(frame, text="Apply HeartMuLa Low-Memory Preset", command=self._apply_heartmula_low_memory_preset).grid(row=10, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self._add_labeled_entry(frame, 11, "MelodyFlow Model Dir", self.melodyflow_model_var, self._browse_melodyflow_model)
        self._add_labeled_entry(frame, 12, "MelodyFlow Python", self.melodyflow_python_var, self._browse_melodyflow_python)
        self._add_labeled_entry(frame, 13, "ACE-Step v1-3.5B Checkpoints", self.acestep_ckpt_var, self._browse_acestep_ckpt)
        self._add_labeled_entry(frame, 14, "ACE-Step v1-3.5B Python", self.acestep_python_var, self._browse_acestep_python)
        self._add_labeled_entry(frame, 15, "ACE-Step 1.5 Root", self.acestep15_root_var, self._browse_acestep15_root)
        self._add_labeled_entry(frame, 16, "ACE-Step 1.5 Checkpoints", self.acestep15_ckpt_var, self._browse_acestep15_ckpt)
        self._add_labeled_entry(frame, 17, "ACE-Step 1.5 Python", self.acestep15_python_var, self._browse_acestep15_python)

        help_text = tk.Text(frame, height=11, wrap="word")
        help_text.grid(row=18, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        help_text.insert(
            "1.0",
            "Each model card shows:\n"
            "- live runtime while the backend is rendering\n"
            "- peak VRAM observed during that model run\n"
            "- measured generated audio seconds against the requested target\n"
            "- output seconds versus runtime after completion\n"
            "- waveform and spectrogram views for quick shape and frequency comparison\n"
            "- rating and notes persisted to comparison_summary.json\n\n"
            "HeartMuLa now has two comparison slots in the GUI: the original Base 3B release and the Happy New Year variant. Both reuse the same HeartMuLa python environment and heartlib clone, but each points at its own checkpoint root. CFG scale and resident mode are shared experiment knobs for latency testing; resident mode keeps HeartMuLa and HeartCodec loaded instead of lazy-loading each request.\n\n"
            "HeartCodec dtype can be switched between float32 and bfloat16 here. Float32 stays the conservative default; bfloat16 is the lower-VRAM option we just A/B tested and can help memory-constrained GPUs.\n\n"
            "HeartMuLa VRAM Limit (GB) applies a per-process CUDA allocator cap for HeartMuLa runs. This is useful for testing whether a bf16 run can survive inside an approximate 12 GB budget, but it is not a full hardware partition of the GPU.\n\n"
            "HeartMuLa Staged Decode keeps only one major model on the GPU at a time: it generates frames with HeartMuLa, moves the frames to CPU memory, unloads HeartMuLa, then loads HeartCodec and decodes. This is the current best low-memory mode for getting under a 12 GB cap while still using GPU decode.\n\n"
            "The HeartMuLa fast preset applies the quickest safe settings validated so far in this repo: CFG 1.5 with resident mode enabled.\n\n"
            "The HeartMuLa low-memory preset applies the memory-saving settings validated so far in this repo: CFG 1.0, bfloat16 codec, staged decode enabled, and lazy loading enabled.\n\n"
            "MelodyFlow and both ACE-Step lines also need local runtime assets. The legacy ACE-Step slot points at the older v1-3.5B checkpoint layout. ACE-Step 1.5 follows the upstream fork in third_party/ACE-Step-1.5 and expects a checkpoints root containing folders such as acestep-v15-turbo, vae, and the LM models. MelodyFlow is currently wired for text-only generation in this comparison UI; reference-audio editing is not exposed here yet.\n\n"
            "Rendering progress is shown as live runtime plus measured audio currently written to the backend output file. Some models only flush that file near the end, so their progress can stay near zero until the final write."
        )
        help_text.configure(state="disabled")
        self.backend_help_text = help_text

    def _build_transcription_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Lyrics Transcription", padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(12, weight=1)

        ttk.Label(frame, text="Sample Library").grid(row=0, column=0, sticky="w", pady=(0, 8), padx=(0, 8))
        self.sample_audio_combo = ttk.Combobox(frame, textvariable=self.sample_audio_var, values=self.sample_audio_files, state="readonly")
        self.sample_audio_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        self.sample_audio_combo.bind("<<ComboboxSelected>>", self._apply_selected_sample_audio)
        ttk.Button(frame, text="Refresh", command=self._refresh_sample_audio_files).grid(row=0, column=2, sticky="e", pady=(0, 8), padx=(8, 0))

        self._add_labeled_entry(frame, 1, "Song / Stem", self.transcription_audio_var, self._browse_transcription_audio)
        self._add_labeled_entry(frame, 2, "Transcription Out", self.transcription_output_dir_var, self._browse_transcription_output_dir)
        self._add_labeled_entry(frame, 3, "Separator Model", self.separator_model_var)
        self._add_labeled_entry(frame, 4, "Separator Python", self.separator_python_var, self._browse_separator_python)
        self._add_labeled_entry(frame, 5, "Language Override", self.transcription_language_var)

        ttk.Checkbutton(frame, text="Skip separation (for isolated vocal stems)", variable=self.transcription_skip_separation_var).grid(
            row=6,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 8),
        )

        controls = ttk.Frame(frame)
        controls.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(4, weight=1)
        self.transcribe_button = ttk.Button(controls, text="Transcribe", command=self._start_transcription)
        self.transcribe_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.play_vocals_button = ttk.Button(controls, text="Play Vocals", command=self._play_transcription_vocals, state="disabled")
        self.play_vocals_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.open_transcription_summary_button = ttk.Button(controls, text="Open Summary", command=self._open_transcription_summary, state="disabled")
        self.open_transcription_summary_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        self.open_transcription_dir_button = ttk.Button(controls, text="Open Folder", command=self._open_transcription_output_dir, state="disabled")
        self.open_transcription_dir_button.grid(row=0, column=3, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Copy Text", command=self._copy_transcription_text).grid(row=0, column=4, sticky="ew")

        ttk.Label(frame, textvariable=self.transcription_status_var, wraplength=360).grid(row=8, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, textvariable=self.transcription_runtime_var).grid(row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(frame, textvariable=self.transcription_vocal_var, wraplength=360).grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(frame, textvariable=self.transcription_summary_var, wraplength=360).grid(row=11, column=0, columnspan=3, sticky="w", pady=(4, 0))

        self.transcription_text = tk.Text(frame, height=12, wrap="word")
        self.transcription_text.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.transcription_text.insert(
            "1.0",
            "Pick a song from assets/music samples or browse to any local track, then run the existing separator plus HeartTranscriptor pipeline.\n\n"
            "The result summary JSON and vocal-only stem stay on disk so you can compare multiple songs quickly.",
        )

    def _set_shell_border_color(self, color: str) -> None:
        self.shell_frame.configure(bg=color, highlightbackground=color, highlightcolor=color)

    def _update_shell_border(self) -> None:
        palette = active_theme_palette(self.theme_var.get())
        color = RESIZING_BORDER_COLOR if self._resize_edge_active else str(palette["border_ready"])
        self._set_shell_border_color(color)

    def _pointer_near_resize_edge(self, x_root: int, y_root: int) -> bool:
        try:
            left = self.root.winfo_rootx()
            top = self.root.winfo_rooty()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
        except tk.TclError:
            return False
        if width <= 0 or height <= 0:
            return False
        right = left + width
        bottom = top + height
        if x_root < left or x_root > right or y_root < top or y_root > bottom:
            return False
        return (
            x_root - left <= RESIZE_HIT_ZONE
            or right - x_root <= RESIZE_HIT_ZONE
            or y_root - top <= RESIZE_HIT_ZONE
            or bottom - y_root <= RESIZE_HIT_ZONE
        )

    def _set_resize_edge_active(self, active: bool) -> None:
        if self._resize_edge_active == active:
            return
        self._resize_edge_active = active
        self.resize_state_var.set("Resize: edge hover" if active else "Resize: ready")
        self._update_shell_border()

    def _handle_root_motion(self, event=None) -> None:
        if event is None:
            return
        self._set_resize_edge_active(self._pointer_near_resize_edge(event.x_root, event.y_root))

    def _handle_root_leave(self, _event=None) -> None:
        self._set_resize_edge_active(False)

    def _mark_resize_ready(self) -> None:
        self._set_resize_edge_active(False)

    def _on_theme_selected(self, _event=None) -> None:
        self._apply_theme()
        self.status_var.set(f"Theme set to {self.theme_var.get()}")

    def _configure_text_widget(self, widget: tk.Text) -> None:
        palette = active_theme_palette(self.theme_var.get())
        widget.configure(
            bg=palette["entry_bg"],
            fg=palette["entry_fg"],
            insertbackground=palette["entry_fg"],
            selectbackground=palette["accent"],
            selectforeground=palette["entry_fg"],
            highlightbackground=palette["border"],
            highlightcolor=palette["accent"],
            relief="flat",
            bd=1,
        )

    def _apply_theme(self) -> None:
        palette = active_theme_palette(self.theme_var.get())
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.configure(bg=palette["root_bg"])
        self.style.configure("TFrame", background=palette["surface"])
        self.style.configure("TLabelframe", background=palette["surface"], foreground=palette["text"], bordercolor=palette["border"])
        self.style.configure("TLabelframe.Label", background=palette["surface"], foreground=palette["text"])
        self.style.configure("TLabel", background=palette["surface"], foreground=palette["text"])
        self.style.configure("TButton", background=palette["button_bg"], foreground=palette["text"], bordercolor=palette["border"], focusthickness=1, focuscolor=palette["accent"])
        self.style.map("TButton", background=[("active", palette["button_active"]), ("pressed", palette["button_active"])] )
        self.style.configure("TCheckbutton", background=palette["surface"], foreground=palette["text"])
        self.style.map("TCheckbutton", background=[("active", palette["surface_alt"])])
        self.style.configure("TEntry", fieldbackground=palette["entry_bg"], foreground=palette["entry_fg"], bordercolor=palette["border"], insertcolor=palette["entry_fg"])
        self.style.configure("TCombobox", fieldbackground=palette["entry_bg"], background=palette["entry_bg"], foreground=palette["entry_fg"], arrowcolor=palette["text"], bordercolor=palette["border"])
        self.style.map("TCombobox", fieldbackground=[("readonly", palette["entry_bg"])], foreground=[("readonly", palette["entry_fg"])])
        self.style.configure("TNotebook", background=palette["surface"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=palette["surface_alt"], foreground=palette["muted"], padding=(10, 6))
        self.style.map("TNotebook.Tab", background=[("selected", palette["surface"]), ("active", palette["surface_alt"])], foreground=[("selected", palette["text"]), ("active", palette["text"])])
        self.style.configure("TPanedwindow", background=palette["surface"])
        self.style.configure("TProgressbar", troughcolor=palette["surface_alt"], background=palette["accent"], bordercolor=palette["border"], lightcolor=palette["accent_active"], darkcolor=palette["accent"])
        self.style.configure("Vertical.TScrollbar", background=palette["button_bg"], troughcolor=palette["surface_alt"], bordercolor=palette["border"], arrowcolor=palette["text"])

        self._update_shell_border()

        for widget_name in ("prompt_text", "lyrics_text", "backend_help_text", "transcription_text"):
            widget = getattr(self, widget_name, None)
            if isinstance(widget, tk.Text):
                self._configure_text_widget(widget)
        for card in self.model_cards.values():
            self._configure_text_widget(card.notes_text)
        if hasattr(self, "log_text") and isinstance(self.log_text, tk.Text):
            self._configure_text_widget(self.log_text)
        if hasattr(self, "cards_canvas"):
            self.cards_canvas.configure(bg=palette["canvas_bg"])
        for area in self.config_scroll_areas:
            area.canvas.configure(bg=palette["canvas_bg"])

    def _refresh_sample_audio_files(self) -> None:
        self.sample_audio_files = discover_sample_audio_files(REPO_ROOT)
        self.sample_audio_combo.configure(values=self.sample_audio_files)
        if self.sample_audio_files and not self.transcription_audio_var.get().strip():
            selected = self.sample_audio_files[0]
            self.sample_audio_var.set(selected)
            self.transcription_audio_var.set(selected)

    def _apply_selected_sample_audio(self, _event=None) -> None:
        selected = self.sample_audio_var.get().strip()
        if selected:
            self.transcription_audio_var.set(selected)

    def _apply_heartmula_fast_preset(self) -> None:
        self.heart_cfg_scale_var.set("1.5")
        self.heart_stage_codec_var.set(False)
        self.heart_lazy_load_var.set(False)
        self.status_var.set("Applied HeartMuLa fast preset")

    def _apply_heartmula_low_memory_preset(self) -> None:
        self.heart_cfg_scale_var.set("1.0")
        self.heart_codec_dtype_var.set("bfloat16")
        self.heart_stage_codec_var.set(True)
        self.heart_lazy_load_var.set(True)
        self.status_var.set("Applied HeartMuLa low-memory preset")

    def _build_model_cards(self) -> None:
        for idx, (backend, label) in enumerate(MODEL_OPTIONS):
            row = idx // 2
            column = idx % 2

            frame = ttk.LabelFrame(self.cards_container, text=label, padding=12)
            frame.grid(row=row, column=column, sticky="nsew", padx=(0, 12 if column == 0 else 0), pady=(0, 12))
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)

            status_var = tk.StringVar(value="Idle")
            runtime_var = tk.StringVar(value="Runtime: --")
            output_var = tk.StringVar(value="Output: --")
            progress_var = tk.StringVar(value="Generated audio: --")
            vram_var = tk.StringVar(value="VRAM peak: --")
            rating_var = tk.StringVar(value="")

            ttk.Label(frame, textvariable=status_var).grid(row=0, column=0, sticky="w")
            ttk.Label(frame, textvariable=runtime_var).grid(row=0, column=1, sticky="e")
            output_label = ttk.Label(frame, textvariable=output_var, wraplength=CARD_MAX_WRAP)
            output_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
            ttk.Label(frame, textvariable=vram_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

            progressbar = ttk.Progressbar(frame, mode="determinate", maximum=1.0, value=0.0)
            progressbar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
            ttk.Label(frame, textvariable=progress_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

            controls = ttk.Frame(frame)
            controls.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
            controls.columnconfigure(0, weight=1)
            controls.columnconfigure(1, weight=1)
            controls.columnconfigure(2, weight=1)
            controls.columnconfigure(3, weight=1)
            generate_button = ttk.Button(controls, text="Generate", command=lambda b=backend: self._start_single_backend_run(b))
            generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            preview_button = ttk.Button(controls, text="Play", command=lambda b=backend: self._play_backend_output(b), state="disabled")
            preview_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
            ttk.Button(controls, text="Stop", command=stop_audio_preview).grid(row=0, column=2, sticky="ew", padx=(0, 6))
            open_button = ttk.Button(controls, text="Open File", command=lambda b=backend: self._open_backend_output(b), state="disabled")
            open_button.grid(row=0, column=3, sticky="ew")

            visual_frame = ttk.Frame(frame)
            visual_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
            visual_frame.columnconfigure(0, weight=1)

            waveform_label = ttk.Label(visual_frame, text="No waveform yet", anchor="center")
            waveform_label.grid(row=0, column=0, sticky="ew")

            spectrogram_label = ttk.Label(visual_frame, text="No spectrogram yet", anchor="center")
            spectrogram_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

            rating_row = ttk.Frame(frame)
            rating_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
            ttk.Label(rating_row, text="Rating").grid(row=0, column=0, sticky="w")
            rating_combo = ttk.Combobox(rating_row, values=RATING_OPTIONS, textvariable=rating_var, width=6, state="readonly")
            rating_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
            rating_combo.bind("<<ComboboxSelected>>", lambda _event, b=backend: self._save_ratings_to_summary(b))

            notes_text = tk.Text(frame, height=4, wrap="word")
            notes_text.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
            notes_text.bind("<FocusOut>", lambda _event, b=backend: self._save_ratings_to_summary(b))

            self.model_cards[backend] = ModelCardState(
                backend=backend,
                label=label,
                frame=frame,
                status_var=status_var,
                runtime_var=runtime_var,
                output_var=output_var,
                output_label=output_label,
                progress_var=progress_var,
                vram_var=vram_var,
                rating_var=rating_var,
                notes_text=notes_text,
                progressbar=progressbar,
                waveform_label=waveform_label,
                spectrogram_label=spectrogram_label,
                generate_button=generate_button,
                preview_button=preview_button,
                open_button=open_button,
            )
            self._configure_text_widget(notes_text)
            self._attach_backend_hint(frame, BACKEND_FIELD_HINTS.get(backend, "Valid inputs vary by backend."))

    def _attach_backend_hint(self, widget: tk.Misc, text: str) -> None:
        tooltip = HoverTooltip(self.root, text)
        self.hover_tooltips.append(tooltip)
        self._bind_tooltip_recursive(widget, tooltip)

    def _bind_tooltip_recursive(self, widget: tk.Misc, tooltip: HoverTooltip) -> None:
        widget.bind("<Enter>", tooltip.schedule, add="+")
        widget.bind("<Leave>", tooltip.cancel, add="+")
        widget.bind("<ButtonPress>", tooltip.cancel, add="+")
        for child in widget.winfo_children():
            self._bind_tooltip_recursive(child, tooltip)

    def _add_labeled_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, browse_command=None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 8), padx=(0, 8))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=(0, 8))
        if browse_command:
            ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="e", pady=(0, 8), padx=(8, 0))

    def _add_labeled_combobox(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 8), padx=(0, 8))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").grid(row=row, column=1, sticky="ew", pady=(0, 8))

    def _refresh_card_visibility(self) -> None:
        visible = [backend for backend, _ in MODEL_OPTIONS if self.model_vars[backend].get()]
        canvas_width = self.cards_canvas.winfo_width() if hasattr(self, "cards_canvas") else 0
        canvas_height = self.cards_canvas.winfo_height() if hasattr(self, "cards_canvas") else 0
        self._card_columns = self._desired_card_columns(canvas_width, canvas_height, len(visible))
        for column in range(len(MODEL_OPTIONS)):
            self.cards_container.columnconfigure(column, weight=1 if column < self._card_columns else 0)
        wraplength = self._card_wraplength(canvas_width, self._card_columns)
        for idx, backend in enumerate(visible):
            row = idx // self._card_columns
            column = idx % self._card_columns
            padx = (0, 12) if self._card_columns > 1 and column < self._card_columns - 1 else (0, 0)
            self.model_cards[backend].frame.grid(row=row, column=column, sticky="nsew", padx=padx, pady=(0, 12))
            self.model_cards[backend].output_label.configure(wraplength=wraplength)
        for backend, _ in MODEL_OPTIONS:
            if backend not in visible:
                self.model_cards[backend].frame.grid_remove()
        self._update_cards_scrollregion()

    def _desired_card_columns(self, canvas_width: int, canvas_height: int, visible_count: int) -> int:
        if visible_count <= 1:
            return 1
        if canvas_width and canvas_height and canvas_height >= max(900, canvas_width):
            return 1
        if visible_count >= 4 and canvas_width >= CARD_FOUR_COLUMN_WIDTH:
            return 4
        if visible_count >= 3 and canvas_width >= CARD_THREE_COLUMN_WIDTH:
            return 3
        if canvas_width >= CARD_TWO_COLUMN_WIDTH:
            return min(2, visible_count)
        if canvas_width and canvas_width < CARD_SINGLE_COLUMN_WIDTH:
            return 1
        return min(2, visible_count)

    def _card_wraplength(self, canvas_width: int, columns: int) -> int:
        if columns <= 1:
            return CARD_MAX_WRAP
        usable_width = max(canvas_width - 24, 0)
        per_card = usable_width // columns if usable_width else CARD_MAX_WRAP
        return max(CARD_MIN_WRAP, min(CARD_MAX_WRAP, per_card - 60))

    def _update_cards_scrollregion(self) -> None:
        if not hasattr(self, "cards_canvas"):
            return
        self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all"))

    def _handle_cards_container_configure(self, _event=None) -> None:
        self._update_cards_scrollregion()

    def _handle_cards_canvas_configure(self, event) -> None:
        self.cards_canvas.itemconfigure(self.cards_window, width=event.width)
        self._refresh_card_visibility()

    def _widget_in_cards_panel(self, widget: tk.Misc | None) -> bool:
        while widget is not None:
            if widget is self.cards_canvas or widget is self.cards_container or widget is getattr(self, "cards_scrollbar", None):
                return True
            widget = widget.master
        return False

    def _handle_global_mousewheel(self, event) -> None:
        widget_under_pointer = self.root.winfo_containing(self.root.winfo_pointerx(), self.root.winfo_pointery())
        if widget_under_pointer is None:
            return

        if widget_under_pointer.winfo_class() == "Text":
            return

        for area in self.config_scroll_areas:
            if self._widget_in_scroll_area(widget_under_pointer, area):
                self._scroll_canvas(area.canvas, event)
                return

        if not hasattr(self, "cards_canvas") or not self._widget_in_cards_panel(widget_under_pointer):
            return
        self._scroll_canvas(self.cards_canvas, event)

    def _widget_in_scroll_area(self, widget: tk.Misc | None, area: ScrollAreaState) -> bool:
        while widget is not None:
            if widget is area.canvas or widget is area.container or widget is area.scrollbar:
                return True
            widget = widget.master
        return False

    def _scroll_canvas(self, canvas: tk.Canvas, event) -> None:
        scrollregion = canvas.bbox("all")
        if not scrollregion:
            return
        total_height = scrollregion[3] - scrollregion[1]
        if total_height <= canvas.winfo_height():
            return

        if getattr(event, "num", None) == 4:
            delta_units = -1
        elif getattr(event, "num", None) == 5:
            delta_units = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            delta_units = -max(1, int(abs(delta) / 120)) if delta > 0 else max(1, int(abs(delta) / 120))
        canvas.yview_scroll(delta_units, "units")

    def _handle_scrollable_container_configure(self, area: ScrollAreaState) -> None:
        area.canvas.configure(scrollregion=area.canvas.bbox("all"))

    def _handle_scrollable_canvas_configure(self, area: ScrollAreaState, event) -> None:
        area.canvas.itemconfigure(area.window_id, width=event.width)
        area.canvas.configure(scrollregion=area.canvas.bbox("all"))

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(Path.cwd()))
        if path:
            self.output_dir_var.set(path)

    def _browse_reference_audio(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")])
        if path:
            self.reference_audio_var.set(path)

    def _browse_transcription_audio(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All files", "*.*")])
        if path:
            self.transcription_audio_var.set(path)
            self.sample_audio_var.set(path)

    def _browse_transcription_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.transcription_output_dir_var.get() or str(Path.cwd()))
        if path:
            self.transcription_output_dir_var.set(path)

    def _browse_heart_root(self) -> None:
        path = filedialog.askdirectory(initialdir=self.heart_root_var.get() or str(Path.cwd()))
        if path:
            self.heart_root_var.set(path)

    def _browse_heart_hny_ckpt(self) -> None:
        path = filedialog.askdirectory(initialdir=self.heart_hny_ckpt_var.get() or str(Path.cwd()))
        if path:
            self.heart_hny_ckpt_var.set(path)

    def _browse_heart_base_ckpt(self) -> None:
        path = filedialog.askdirectory(initialdir=self.heart_base_ckpt_var.get() or str(Path.cwd()))
        if path:
            self.heart_base_ckpt_var.set(path)

    def _browse_heart_python(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python", "python.exe"), ("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.heart_python_var.set(path)

    def _browse_melodyflow_model(self) -> None:
        path = filedialog.askdirectory(initialdir=self.melodyflow_model_var.get() or str(Path.cwd()))
        if path:
            self.melodyflow_model_var.set(path)

    def _browse_melodyflow_python(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python", "python.exe"), ("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.melodyflow_python_var.set(path)

    def _browse_acestep_ckpt(self) -> None:
        path = filedialog.askdirectory(initialdir=self.acestep_ckpt_var.get() or str(Path.cwd()))
        if path:
            self.acestep_ckpt_var.set(path)

    def _browse_acestep_python(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python", "python.exe"), ("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.acestep_python_var.set(path)

    def _browse_acestep15_root(self) -> None:
        path = filedialog.askdirectory(initialdir=self.acestep15_root_var.get() or str(Path.cwd()))
        if path:
            self.acestep15_root_var.set(path)

    def _browse_acestep15_ckpt(self) -> None:
        path = filedialog.askdirectory(initialdir=self.acestep15_ckpt_var.get() or str(Path.cwd()))
        if path:
            self.acestep15_ckpt_var.set(path)

    def _browse_acestep15_python(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python", "python.exe"), ("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.acestep15_python_var.set(path)

    def _browse_separator_python(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python", "python.exe"), ("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.separator_python_var.set(path)

    def _clear_text_inputs(self) -> None:
        if hasattr(self, "prompt_text"):
            self.prompt_text.delete("1.0", "end")
        if hasattr(self, "lyrics_text"):
            self.lyrics_text.delete("1.0", "end")
        self.tags_var.set("")
        save_gui_settings(self._get_settings())
        self.status_var.set("Cleared prompt, lyrics, and tags")

    def _apply_env(self) -> None:
        updates = self._get_settings()
        save_gui_settings(updates)
        for key, value in updates.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def _get_settings(self) -> dict[str, str]:
        config_panel_tab = "transcription"
        if hasattr(self, "config_tabs"):
            selected = self.config_tabs.select()
            config_panel_tab = "transcription" if selected and selected == str(self.transcription_tab) else "backends"
        return {
            "PROMPT_TEXT": self.prompt_text.get("1.0", "end").strip() if hasattr(self, "prompt_text") else self.initial_prompt_text,
            "LYRICS_TEXT": self.lyrics_text.get("1.0", "end").strip() if hasattr(self, "lyrics_text") else self.initial_lyrics_text,
            "TAGS_TEXT": self.tags_var.get().strip(),
            "HEARTMULA_ROOT": self.heart_root_var.get().strip(),
            "HEARTMULA_CKPT_DIR": self.heart_hny_ckpt_var.get().strip(),
            "HEARTMULA_HNY_CKPT_DIR": self.heart_hny_ckpt_var.get().strip(),
            "HEARTMULA_BASE_CKPT_DIR": self.heart_base_ckpt_var.get().strip(),
            "HEARTMULA_PYTHON": self.heart_python_var.get().strip(),
            "HEARTMULA_CFG_SCALE": self.heart_cfg_scale_var.get().strip(),
            "HEARTMULA_LAZY_LOAD": "true" if self.heart_lazy_load_var.get() else "false",
            "HEARTMULA_CODEC_DTYPE": self.heart_codec_dtype_var.get().strip(),
            "HEARTMULA_MAX_VRAM_GB": self.heart_max_vram_gb_var.get().strip(),
            "HEARTMULA_STAGE_CODEC": "true" if self.heart_stage_codec_var.get() else "false",
            "MELODYFLOW_MODEL_DIR": self.melodyflow_model_var.get().strip(),
            "MELODYFLOW_PYTHON": self.melodyflow_python_var.get().strip(),
            "ACESTEP_CKPT_DIR": self.acestep_ckpt_var.get().strip(),
            "ACESTEP_PYTHON": self.acestep_python_var.get().strip(),
            "ACESTEP15_ROOT": self.acestep15_root_var.get().strip(),
            "ACESTEP15_CKPT_DIR": self.acestep15_ckpt_var.get().strip(),
            "ACESTEP15_PYTHON": self.acestep15_python_var.get().strip(),
            "TRANSCRIPTION_AUDIO_FILE": self.transcription_audio_var.get().strip(),
            "TRANSCRIPTION_OUTPUT_DIR": self.transcription_output_dir_var.get().strip(),
            "TRANSCRIPTION_LANGUAGE": self.transcription_language_var.get().strip(),
            "TRANSCRIPTION_SKIP_SEPARATION": "true" if self.transcription_skip_separation_var.get() else "false",
            "SEPARATOR_MODEL": self.separator_model_var.get().strip(),
            "SEPARATOR_PYTHON": self.separator_python_var.get().strip(),
            "CONFIG_PANEL_TAB": config_panel_tab,
            "APP_THEME": self.theme_var.get().strip(),
        }

    def _collect_preflight_issues(self, models: list[str]) -> list[str]:
        return collect_preflight_issues(models, self._get_settings())

    def _show_preflight_report(self) -> None:
        self._apply_env()
        models = self._selected_models()
        if not models:
            messagebox.showinfo("Setup Check", "Select at least one model to check.")
            return
        issues = self._collect_preflight_issues(models)
        if issues:
            report = "Preflight found the following setup gaps:\n\n- " + "\n- ".join(issues)
            self._log(report.replace("\n", " | "))
            messagebox.showwarning("Setup Check", report)
            return
        self._log(f"Preflight passed for: {', '.join(models)}")
        messagebox.showinfo("Setup Check", f"Selected backends look configured: {', '.join(models)}")

    def _selected_models(self) -> list[str]:
        return [backend for backend, _ in MODEL_OPTIONS if self.model_vars[backend].get()]

    def _reset_card_for_run(self, backend: str) -> None:
        card = self.model_cards[backend]
        card.status_var.set("Idle")
        card.runtime_var.set("Runtime: --")
        card.output_var.set("Output: --")
        card.progress_var.set("Generated audio: --")
        card.vram_var.set("VRAM peak: --")
        card.result = None
        card.started_at = None
        card.started_epoch = None
        card.target_audio_seconds = 0.0
        card.output_path_hint = None
        card.completed = False
        card.peak_vram_mb = 0
        card.progressbar.configure(mode="determinate", maximum=1.0, value=0.0)
        card.preview_button.configure(state="disabled")
        card.open_button.configure(state="disabled")
        card.waveform_label.configure(text="No waveform yet", image="")
        card.spectrogram_label.configure(text="No spectrogram yet", image="")
        card.waveform_photo = None
        card.spectrogram_photo = None

    def _reset_cards_for_run(self) -> None:
        for backend, _ in MODEL_OPTIONS:
            self._reset_card_for_run(backend)

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        shared_state = "normal" if enabled else "disabled"
        self.run_comparison_button.configure(state=shared_state)
        self.transcribe_button.configure(state=shared_state)
        for card in self.model_cards.values():
            card.generate_button.configure(state=shared_state)

    def _set_transcription_buttons_enabled(self, has_result: bool) -> None:
        result_state = "normal" if has_result else "disabled"
        self.play_vocals_button.configure(state=result_state)
        self.open_transcription_summary_button.configure(state=result_state)
        self.open_transcription_dir_button.configure(state=result_state)

    def _reset_transcription_for_run(self) -> None:
        self.current_transcription_summary_path = None
        self.current_transcription_vocal_path = None
        self.transcription_status_var.set("Transcript: running...")
        self.transcription_runtime_var.set("Runtime: --")
        self.transcription_vocal_var.set("Vocal stem: waiting")
        self.transcription_summary_var.set("Summary: waiting")
        self.transcription_text.delete("1.0", "end")
        self.transcription_text.insert("1.0", "Running separator + HeartTranscriptor...")
        self._set_transcription_buttons_enabled(False)

    def _log(self, message: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, lambda m=message: self._log(m))
            return
        if not hasattr(self, "log_window"):
            self.log_window = tk.Toplevel(self.root)
            self.log_window.title("Run Log")
            self.log_window.geometry("820x240")
            self.log_text = tk.Text(self.log_window, wrap="word")
            self.log_text.pack(fill="both", expand=True)
            self._configure_text_widget(self.log_text)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _log_backend_launch_details(self, backend: str, label: str) -> None:
        if backend == "heartmula_hny":
            checkpoint_dir = self.heart_hny_ckpt_var.get().strip()
            self._log(
                f"{label} config: checkpoint={checkpoint_dir or '<missing>'} | "
                f"cfg_scale={self.heart_cfg_scale_var.get().strip() or '<default>'} | "
                f"lazy_load={'true' if self.heart_lazy_load_var.get() else 'false'} | "
                f"codec_dtype={self.heart_codec_dtype_var.get().strip() or 'float32'} | "
                f"max_vram_gb={self.heart_max_vram_gb_var.get().strip() or '<none>'} | "
                f"stage_codec={'true' if self.heart_stage_codec_var.get() else 'false'} | "
                f"python={self.heart_python_var.get().strip() or sys.executable}"
            )
            return
        if backend == "heartmula_base":
            checkpoint_dir = self.heart_base_ckpt_var.get().strip()
            self._log(
                f"{label} config: checkpoint={checkpoint_dir or '<missing>'} | "
                f"cfg_scale={self.heart_cfg_scale_var.get().strip() or '<default>'} | "
                f"lazy_load={'true' if self.heart_lazy_load_var.get() else 'false'} | "
                f"codec_dtype={self.heart_codec_dtype_var.get().strip() or 'float32'} | "
                f"max_vram_gb={self.heart_max_vram_gb_var.get().strip() or '<none>'} | "
                f"stage_codec={'true' if self.heart_stage_codec_var.get() else 'false'} | "
                f"python={self.heart_python_var.get().strip() or sys.executable}"
            )
            return
        if backend == "ace_step":
            self._log(
                f"{label} config: checkpoint={self.acestep_ckpt_var.get().strip() or '<missing>'} | "
                f"python={self.acestep_python_var.get().strip() or sys.executable}"
            )
            return
        if backend == "ace_step_v15":
            self._log(
                f"{label} config: root={self.acestep15_root_var.get().strip() or '<missing>'} | "
                f"checkpoints={self.acestep15_ckpt_var.get().strip() or '<missing>'} | "
                f"python={self.acestep15_python_var.get().strip() or sys.executable}"
            )

    def _log_backend_result_details(self, label: str, result: dict) -> None:
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            return

        checkpoint_dir = metadata.get("checkpoint_dir")
        cfg_scale = metadata.get("cfg_scale")
        lazy_load = metadata.get("lazy_load")
        codec_dtype = metadata.get("codec_dtype")
        max_vram_gb = metadata.get("max_vram_gb")
        stage_codec = metadata.get("stage_codec")
        python_executable = metadata.get("python_executable")
        if any(value is not None for value in (checkpoint_dir, cfg_scale, lazy_load, codec_dtype, max_vram_gb, stage_codec, python_executable)):
            self._log(
                f"{label} resolved: checkpoint={checkpoint_dir or '<unknown>'} | "
                f"cfg_scale={cfg_scale if cfg_scale is not None else '<unknown>'} | "
                f"lazy_load={lazy_load if lazy_load is not None else '<unknown>'} | "
                f"codec_dtype={codec_dtype or '<unknown>'} | "
                f"max_vram_gb={max_vram_gb if max_vram_gb is not None else '<none>'} | "
                f"stage_codec={stage_codec if stage_codec is not None else '<unknown>'} | "
                f"python={python_executable or '<unknown>'}"
            )

        stdout = metadata.get("stdout")
        if isinstance(stdout, str) and stdout.strip():
            backend_lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
            important_lines = [
                line
                for line in backend_lines
                if (
                    "Applied CUDA per-process limit:" in line
                    or
                    "Loaded model dtypes:" in line
                    or "Loaded model dtypes after HeartMuLa stage:" in line
                    or "Loaded model dtypes after generation:" in line
                    or "CUDA after load:" in line
                    or "CUDA after HeartMuLa stage:" in line
                    or "CUDA after generation:" in line
                    or "Staged codec mode requires lazy_load=true; overriding lazy_load to true." in line
                    or "HeartMuLa settings:" in line
                    or "Generated music saved to" in line
                )
            ]
            for line in (important_lines or backend_lines[-5:]):
                self._log(f"{label} backend: {line}")

        command = result.get("command")
        if isinstance(command, list) and command:
            self._log(f"{label} command: {shlex.join([str(part) for part in command])}")

    def _start_gpu_monitor(self) -> None:
        def poll() -> None:
            while not self.monitor_stop.is_set():
                stats = query_gpu_stats()
                used = stats["used_mb"]
                total = stats["total_mb"]
                util = stats["util"]
                if used.isdigit():
                    used_int = int(used)
                    self.global_peak_vram_mb = max(self.global_peak_vram_mb, used_int)
                    if self.current_backend_name and self.current_backend_name in self.model_cards:
                        current_card = self.model_cards[self.current_backend_name]
                        current_card.peak_vram_mb = max(current_card.peak_vram_mb, used_int)
                global_text = f"VRAM: {used}/{total} MB | GPU util: {util}% | Global peak: {self.global_peak_vram_mb if self.global_peak_vram_mb else 'N/A'} MB"
                self.root.after(0, lambda value=global_text: self.global_vram_var.set(value))
                if self.current_backend_name and self.current_backend_name in self.model_cards:
                    card = self.model_cards[self.current_backend_name]
                    peak_text = f"VRAM peak: {card.peak_vram_mb if card.peak_vram_mb else 'N/A'} MB"
                    self.root.after(0, lambda c=card, value=peak_text: c.vram_var.set(value))
                time.sleep(0.8)

        threading.Thread(target=poll, daemon=True).start()

    def _parse_run_inputs(self) -> tuple[str, float, int | None] | None:
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showerror("Missing prompt", "Enter a prompt before running comparison.")
            return None

        try:
            duration = float(self.duration_var.get().strip())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid duration", "Duration must be a number greater than 0.")
            return None

        seed = None
        if self.seed_var.get().strip():
            try:
                seed = int(self.seed_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid seed", "Seed must be an integer.")
                return None

        return prompt, duration, seed

    def _start_models_run(self, models: list[str], *, reset_all_cards: bool, run_label: str) -> None:
        if self.running:
            return
        if not models:
            messagebox.showerror("No models selected", "Select at least one model.")
            return

        parsed = self._parse_run_inputs()
        if not parsed:
            return
        prompt, duration, seed = parsed

        self._apply_env()
        issues = self._collect_preflight_issues(models)
        if issues:
            report = "Preflight found the following setup gaps:\n\n- " + "\n- ".join(issues)
            self._log(report.replace("\n", " | "))
            continue_anyway = messagebox.askyesno("Setup Incomplete", report + "\n\nRun anyway?")
            if not continue_anyway:
                self.status_var.set("Setup incomplete")
                return
        self.running = True
        self.global_peak_vram_mb = 0
        self.current_summary_path = None
        self._set_run_buttons_enabled(False)
        self.status_var.set(f"Running {run_label}...")
        self.current_run_var.set(f"Queued models: {', '.join(models)}")
        self.global_speed_var.set("Runtime: starting")
        if reset_all_cards:
            self._reset_cards_for_run()
        else:
            for model_name in models:
                self._reset_card_for_run(model_name)
        self._log(f"Starting {run_label} for: {', '.join(models)}")
        for model_name in models:
            label = next(display for name, display in MODEL_OPTIONS if name == model_name)
            self._log_backend_launch_details(model_name, label)

        thread = threading.Thread(
            target=self._run_models_thread,
            args=(prompt, models, duration, seed, run_label),
            daemon=True,
        )
        thread.start()

    def _current_transcription_checkpoint_root(self) -> str:
        return self.heart_hny_ckpt_var.get().strip() or self.heart_root_var.get().strip() or self.heart_python_var.get().strip()

    def _start_transcription(self) -> None:
        if self.running:
            return

        audio_file = self.transcription_audio_var.get().strip()
        if not audio_file:
            messagebox.showerror("Missing audio", "Choose a song or vocal stem to transcribe.")
            return
        audio_path = Path(audio_file)
        if not audio_path.exists():
            messagebox.showerror("Missing audio", f"Input audio does not exist:\n{audio_path}")
            return

        if self.transcription_skip_separation_var.get() and not self._looks_like_isolated_vocal_stem(audio_path):
            continue_direct = messagebox.askyesno(
                "Skip separation?",
                "Skip separation is usually only for isolated vocal stems.\n\n"
                "This file looks like a full song mix, so direct transcription may be less accurate than separating vocals first. Continue anyway?",
            )
            if not continue_direct:
                self.transcription_skip_separation_var.set(False)
                return

        separator_python = self.separator_python_var.get().strip()
        if not self.transcription_skip_separation_var.get() and (not separator_python or not Path(separator_python).exists()):
            messagebox.showerror("Missing separator Python", "Configure the separator environment Python before transcribing full songs.")
            return

        heart_python = self.heart_python_var.get().strip()
        if not heart_python or not Path(heart_python).exists():
            messagebox.showerror("Missing HeartMuLa Python", "Configure the HeartMuLa Python environment before transcribing.")
            return

        checkpoint_root = self.heart_hny_ckpt_var.get().strip() or self.heart_base_ckpt_var.get().strip()
        if not checkpoint_root or not Path(checkpoint_root).exists():
            messagebox.showerror("Missing checkpoint root", "Configure a HeartMuLa checkpoint root that contains or can download HeartTranscriptor.")
            return

        output_dir = self.transcription_output_dir_var.get().strip() or DEFAULT_TRANSCRIPTION_OUTPUT_DIR
        self._apply_env()
        save_gui_settings(self._get_settings())
        self.running = True
        self._set_run_buttons_enabled(False)
        self.status_var.set("Running transcription...")
        self.current_run_var.set(f"Transcribing: {audio_path.name}")
        self.global_speed_var.set("Runtime: transcribing")
        self._reset_transcription_for_run()
        self._log(f"Starting transcription for {audio_path.name}")
        thread = threading.Thread(
            target=self._run_transcription_thread,
            args=(audio_path, output_dir, separator_python, heart_python, checkpoint_root),
            daemon=True,
        )
        thread.start()

    def _looks_like_isolated_vocal_stem(self, audio_path: Path) -> bool:
        name = audio_path.name.lower()
        return any(token in name for token in ("vocal", "vocals", "stem", "acapella", "karaoke"))

    def _start_run(self) -> None:
        self._start_models_run(self._selected_models(), reset_all_cards=True, run_label="comparison")

    def _start_single_backend_run(self, backend: str) -> None:
        label = next(display for name, display in MODEL_OPTIONS if name == backend)
        self._start_models_run([backend], reset_all_cards=False, run_label=label)

    def _begin_card_run(self, backend: str, label: str, target_audio_seconds: float, output_dir: str) -> None:
        card = self.model_cards[backend]
        card.started_at = time.perf_counter()
        card.started_epoch = time.time()
        card.target_audio_seconds = max(float(target_audio_seconds), 0.1)
        expected_output = expected_backend_output_path(output_dir, backend)
        card.output_path_hint = str(expected_output) if expected_output else None
        card.completed = False
        card.result = None
        card.peak_vram_mb = 0
        card.status_var.set(f"Generating with {label}...")
        card.runtime_var.set("Runtime: 0.00s")
        card.output_var.set("Output file: waiting for measurable audio")
        card.progressbar.configure(mode="determinate", maximum=card.target_audio_seconds, value=0.0)
        card.progress_var.set(f"Generated audio: 0.00 / {card.target_audio_seconds:.2f}s")
        card.vram_var.set("VRAM peak: measuring...")
        self._tick_card_runtime(backend)

    def _tick_card_runtime(self, backend: str) -> None:
        card = self.model_cards[backend]
        if not self.running or card.completed or card.started_at is None:
            return
        elapsed = time.perf_counter() - card.started_at
        card.runtime_var.set(f"Runtime: {elapsed:.2f}s")
        self.global_speed_var.set(f"Current runtime: {elapsed:.2f}s on {card.label}")
        live_audio_seconds = live_generated_audio_seconds(card.output_path_hint, started_epoch=card.started_epoch)
        if isinstance(live_audio_seconds, (int, float)):
            progress_value = min(float(live_audio_seconds), card.target_audio_seconds)
            card.progressbar.configure(value=progress_value)
            card.progress_var.set(f"Generated audio: {float(live_audio_seconds):.2f} / {card.target_audio_seconds:.2f}s")
        self.root.after(250, lambda b=backend: self._tick_card_runtime(b))

    def _finish_card_run(self, backend: str, result: dict) -> None:
        card = self.model_cards[backend]
        card.completed = True
        card.result = result

        elapsed = result.get("elapsed_seconds")
        elapsed_text = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "--"
        card.runtime_var.set(f"Runtime: {elapsed_text}")

        output_path = result.get("output_path")
        output_seconds = safe_audio_duration_seconds(output_path) if output_path else None
        if result.get("success"):
            if isinstance(output_seconds, (int, float)):
                progress_max = max(card.target_audio_seconds, float(output_seconds), 0.1)
                card.progressbar.configure(maximum=progress_max, value=float(output_seconds))
                card.progress_var.set(f"Generated audio: {float(output_seconds):.2f} / {card.target_audio_seconds:.2f}s")
            else:
                card.progressbar.configure(value=card.progressbar.cget("maximum"))
                card.progress_var.set("Generated audio: duration unavailable")
            if isinstance(output_seconds, (int, float)) and isinstance(elapsed, (int, float)) and output_seconds > 0:
                realtime_factor = elapsed / output_seconds
                card.output_var.set(f"Output {output_seconds:.2f}s | Runtime {elapsed:.2f}s | RTF {realtime_factor:.2f}")
            elif isinstance(output_seconds, (int, float)):
                card.output_var.set(f"Output {output_seconds:.2f}s | Runtime {elapsed_text}")
            else:
                card.output_var.set(f"Output duration unavailable | Runtime {elapsed_text}")
            card.status_var.set("Completed")
            card.vram_var.set(f"VRAM peak: {card.peak_vram_mb if card.peak_vram_mb else 'N/A'} MB")
            card.preview_button.configure(state="normal")
            card.open_button.configure(state="normal")
            self._update_card_visuals(backend)
        else:
            partial_output_seconds = live_generated_audio_seconds(card.output_path_hint, started_epoch=card.started_epoch)
            if isinstance(partial_output_seconds, (int, float)):
                progress_value = min(float(partial_output_seconds), card.target_audio_seconds)
                card.progressbar.configure(value=progress_value)
                card.progress_var.set(f"Generated audio: {float(partial_output_seconds):.2f} / {card.target_audio_seconds:.2f}s")
            else:
                card.progressbar.configure(value=0.0)
                card.progress_var.set("Generated audio: 0.00s")
            card.status_var.set("Failed")
            card.output_var.set((result.get("error") or "Generation failed")[:180])
            card.vram_var.set(f"VRAM peak: {card.peak_vram_mb if card.peak_vram_mb else 'N/A'} MB")
            card.waveform_label.configure(text="No output", image="")
            card.spectrogram_label.configure(text="No output", image="")

    def _update_card_visuals(self, backend: str) -> None:
        card = self.model_cards[backend]
        output_path = card.result.get("output_path") if card.result else None
        if not output_path:
            return

        waveform_image = generate_waveform_image(output_path)
        if not waveform_image:
            card.waveform_label.configure(text="Waveform unavailable", image="")
            card.waveform_photo = None
        else:
            waveform_photo = ImageTk.PhotoImage(waveform_image)
            card.waveform_photo = waveform_photo
            card.waveform_label.configure(image=waveform_photo, text="")

        image = generate_spectrogram_image(output_path, self.theme_var.get())
        if not image:
            card.spectrogram_label.configure(text="Spectrogram unavailable", image="")
            card.spectrogram_photo = None
            return
        photo = ImageTk.PhotoImage(image)
        card.spectrogram_photo = photo
        card.spectrogram_label.configure(image=photo, text="")

    def _run_models_thread(self, prompt: str, models: list[str], duration: float, seed: int | None, run_label: str) -> None:
        output_dir = self.output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR
        reference_audio = self.reference_audio_var.get().strip() or None
        lyrics = self.lyrics_text.get("1.0", "end").strip() or None
        tags = self.tags_var.get().strip() or None

        request = MusicGenRequest(
            prompt=prompt,
            output_dir=output_dir,
            duration_seconds=duration,
            reference_audio=reference_audio,
            lyrics=lyrics,
            tags=tags,
            seed=seed,
        )

        registry = get_backend_registry()
        results = []
        run_started = time.perf_counter()
        summary_path: Path | None = None
        try:
            for model_name in models:
                label = next(display for name, display in MODEL_OPTIONS if name == model_name)
                self.current_backend_name = model_name
                self.root.after(0, lambda name=label: self.status_var.set(f"Running {name}..."))
                self.root.after(0, lambda name=label: self.current_run_var.set(f"Current model: {name}"))
                self.root.after(0, lambda b=model_name, l=label, d=duration, out=output_dir: self._begin_card_run(b, l, d, out))
                self._log(f"Running {label}")

                result = registry[model_name].run(request)
                result_dict = result.to_dict()
                results.append(result_dict)
                self.root.after(0, lambda b=model_name, res=result_dict: self._finish_card_run(b, res))
                self._log_backend_result_details(label, result_dict)

                if result.success:
                    self._log(f"{label} wrote {result.output_path}")
                else:
                    self._log(f"{label} failed: {result.error}")

            total_elapsed = round(time.perf_counter() - run_started, 3)
            summary = {
                "success": any(item["success"] for item in results),
                "prompt": prompt,
                "lyrics": lyrics,
                "tags": tags,
                "models": models,
                "duration_seconds": duration,
                "reference_audio": reference_audio,
                "peak_vram_mb": self.global_peak_vram_mb,
                "total_elapsed_seconds": total_elapsed,
                "results": results,
                "ratings": self._collect_ratings(),
            }

            summary_dir = Path(output_dir)
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / "comparison_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self.current_summary_path = summary_path

            self.root.after(0, lambda: self.status_var.set(f"Done. Summary: {summary_path}"))
            self.root.after(0, lambda: self.current_run_var.set("Current model: none"))
            self.root.after(0, lambda: self.global_speed_var.set(f"Total runtime: {total_elapsed:.2f}s"))
            self._log(f"Saved {run_label} summary to {summary_path}")
            self.root.after(0, self._save_all_ratings_to_summary)
        except Exception as exc:
            self._log(f"{run_label} aborted: {exc}")
            self.root.after(0, lambda: self.status_var.set(f"Run failed: {exc}"))
            self.root.after(0, lambda: self.current_run_var.set("Current model: none"))
        finally:
            self.current_backend_name = None
            self.running = False
            self.root.after(0, lambda: self._set_run_buttons_enabled(True))

    def _run_transcription_thread(
        self,
        audio_path: Path,
        output_dir: str,
        separator_python: str,
        heart_python: str,
        checkpoint_root: str,
    ) -> None:
        started = time.perf_counter()
        try:
            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / "voice" / "transcribe_lyrics.py"),
                str(audio_path.resolve()),
                "--output-dir",
                str(Path(output_dir).resolve()),
                "--separator-python",
                separator_python,
                "--separator-model",
                self.separator_model_var.get().strip() or DEFAULT_SEPARATOR_MODEL,
                "--heart-python",
                heart_python,
                "--checkpoint-root",
                checkpoint_root,
                "--download-missing-transcriptor",
            ]
            language = self.transcription_language_var.get().strip()
            if language:
                command.extend(["--language", language])
            if self.transcription_skip_separation_var.get():
                command.append("--skip-separation")

            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"Transcription failed with exit code {completed.returncode}.")
            payload = extract_json_payload(completed.stdout, completed.stderr)
            transcript_text = extract_transcription_text(payload)
            summary_path = payload.get("summary_path")
            vocal_file = payload.get("vocal_file")
            elapsed = round(time.perf_counter() - started, 3)
            self.current_transcription_summary_path = Path(summary_path) if isinstance(summary_path, str) else None
            self.current_transcription_vocal_path = vocal_file if isinstance(vocal_file, str) else None
            self._log(f"Transcription finished for {audio_path.name}")
            self.root.after(
                0,
                lambda p=payload, text=transcript_text, run_seconds=elapsed: self._finish_transcription_run(p, text, run_seconds),
            )
        except Exception as exc:
            self._log(f"Transcription failed for {audio_path.name}: {exc}")
            self.root.after(0, lambda e=str(exc): self._fail_transcription_run(e))
        finally:
            self.running = False
            self.root.after(0, lambda: self._set_run_buttons_enabled(True))

    def _finish_transcription_run(self, payload: Mapping[str, object], transcript_text: str, elapsed_seconds: float) -> None:
        summary_path = payload.get("summary_path")
        vocal_file = payload.get("vocal_file")
        self.transcription_status_var.set("Transcript: completed")
        self.transcription_runtime_var.set(f"Runtime: {elapsed_seconds:.2f}s")
        self.transcription_vocal_var.set(f"Vocal stem: {vocal_file}" if isinstance(vocal_file, str) else "Vocal stem: unavailable")
        self.transcription_summary_var.set(f"Summary: {summary_path}" if isinstance(summary_path, str) else "Summary: unavailable")
        self.transcription_text.delete("1.0", "end")
        self.transcription_text.insert("1.0", transcript_text or "No transcript text returned.")
        self.status_var.set("Transcription completed")
        self.current_run_var.set("Current model: none")
        self.global_speed_var.set(f"Transcription runtime: {elapsed_seconds:.2f}s")
        self._set_transcription_buttons_enabled(True)

    def _fail_transcription_run(self, error_message: str) -> None:
        self.transcription_status_var.set("Transcript: failed")
        self.transcription_runtime_var.set("Runtime: --")
        self.transcription_vocal_var.set("Vocal stem: unavailable")
        self.transcription_summary_var.set("Summary: unavailable")
        self.transcription_text.delete("1.0", "end")
        self.transcription_text.insert("1.0", error_message)
        self.status_var.set(f"Transcription failed: {error_message}")
        self.current_run_var.set("Current model: none")
        self.global_speed_var.set("Runtime: idle")
        self._set_transcription_buttons_enabled(False)

    def _play_transcription_vocals(self) -> None:
        if not self.current_transcription_vocal_path:
            return
        try:
            play_audio_preview(self.current_transcription_vocal_path)
        except Exception as exc:
            messagebox.showerror("Playback error", str(exc))

    def _open_transcription_summary(self) -> None:
        if self.current_transcription_summary_path and self.current_transcription_summary_path.exists() and os.name == "nt":
            os.startfile(str(self.current_transcription_summary_path))  # type: ignore[attr-defined]

    def _open_transcription_output_dir(self) -> None:
        target_dir = None
        if self.current_transcription_summary_path:
            target_dir = self.current_transcription_summary_path.parent
        else:
            configured = self.transcription_output_dir_var.get().strip()
            if configured:
                target_dir = Path(configured)
        if target_dir and target_dir.exists() and os.name == "nt":
            os.startfile(str(target_dir))  # type: ignore[attr-defined]

    def _copy_transcription_text(self) -> None:
        text = self.transcription_text.get("1.0", "end").strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Transcript copied to clipboard")

    def _play_backend_output(self, backend: str) -> None:
        card = self.model_cards[backend]
        output_path = card.result.get("output_path") if card.result else None
        if not output_path:
            return
        try:
            play_audio_preview(output_path)
        except Exception as exc:
            messagebox.showerror("Playback error", str(exc))

    def _open_backend_output(self, backend: str) -> None:
        card = self.model_cards[backend]
        output_path = card.result.get("output_path") if card.result else None
        if output_path and os.path.exists(output_path) and os.name == "nt":
            os.startfile(output_path)  # type: ignore[attr-defined]

    def _collect_ratings(self) -> dict[str, dict[str, str]]:
        data: dict[str, dict[str, str]] = {}
        for backend, card in self.model_cards.items():
            data[backend] = {
                "rating": card.rating_var.get().strip(),
                "notes": card.notes_text.get("1.0", "end").strip(),
            }
        return data

    def _save_ratings_to_summary(self, _backend: str | None = None) -> None:
        if not self.current_summary_path or not self.current_summary_path.exists():
            return
        self._save_all_ratings_to_summary()

    def _save_all_ratings_to_summary(self) -> None:
        if not self.current_summary_path or not self.current_summary_path.exists():
            return
        try:
            payload = json.loads(self.current_summary_path.read_text(encoding="utf-8"))
        except Exception:
            return
        payload["ratings"] = self._collect_ratings()
        self.current_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _on_close(self) -> None:
        save_gui_settings(self._get_settings())
        self.monitor_stop.set()
        stop_audio_preview()
        if hasattr(self, "log_window") and self.log_window.winfo_exists():
            self.log_window.destroy()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MusicCompareGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()