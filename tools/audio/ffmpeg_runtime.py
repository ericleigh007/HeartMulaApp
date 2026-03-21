from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from shutil import which


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

    ffmpeg_binary = _ensure_ffmpeg_command_name(ffmpeg_binary)

    os.environ["FFMPEG_BINARY"] = ffmpeg_binary
    ffmpeg_dir = str(Path(ffmpeg_binary).parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir
    return ffmpeg_binary


def _ensure_ffmpeg_command_name(ffmpeg_binary: str) -> str:
    binary_path = Path(ffmpeg_binary)
    if binary_path.name.lower() == "ffmpeg.exe":
        return str(binary_path)

    shim_dir = Path(__file__).resolve().parents[2] / ".tmp" / "ffmpeg-shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "ffmpeg.exe"

    source_stat = binary_path.stat()
    if shim_path.exists():
        shim_stat = shim_path.stat()
        if shim_stat.st_size == source_stat.st_size and int(shim_stat.st_mtime) >= int(source_stat.st_mtime):
            return str(shim_path)

    shutil.copy2(binary_path, shim_path)
    return str(shim_path)


def ensure_pcm_wav_input(audio_path: str | Path, *, working_dir: str | Path) -> str:
    input_path = Path(audio_path).resolve()
    if input_path.suffix.lower() == ".wav":
        return str(input_path)

    ffmpeg_binary = configure_ffmpeg()
    if not ffmpeg_binary:
        raise RuntimeError("Could not find ffmpeg for input normalization.")

    prepared_dir = Path(working_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_path = prepared_dir / f"{input_path.stem}_prepared.wav"
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"Failed to normalize audio: {input_path}")
    return str(output_path)