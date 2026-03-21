"""
Orchestrate vocal separation and HeartTranscriptor lyrics transcription.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from tools.ai.download_hf_repo import download_hf_repo
from tools.audio.ffmpeg_runtime import configure_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"
HEARTTRANSCRIPTOR_REPO = "HeartMuLa/HeartTranscriptor-oss"


def _default_separator_python() -> str:
    candidate = REPO_ROOT / ".venv-separator310" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def _default_heart_python() -> str:
    configured = os.environ.get("HEARTMULA_PYTHON", "").strip()
    if configured:
        return configured
    candidate = REPO_ROOT / ".venv-heartmula" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def _default_checkpoint_root() -> str:
    return (
        os.environ.get("HEARTMULA_HNY_CKPT_DIR", "").strip()
        or os.environ.get("HEARTMULA_CKPT_DIR", "").strip()
        or str(REPO_ROOT / "models" / "heartmula" / "ckpt")
    )


def _run_json_command(command: list[str], *, cwd: Path) -> dict:
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"Command failed: {' '.join(command)}")
    stdout = completed.stdout.strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        json_start = stdout.find("{")
        if json_start != -1:
            try:
                return json.loads(stdout[json_start:])
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Command did not return JSON: {' '.join(command)}\n{completed.stdout}") from exc


def _ensure_hearttranscriptor_checkpoint(checkpoint_root: Path, *, download_missing: bool) -> None:
    transcriptor_dir = checkpoint_root / "HeartTranscriptor-oss"
    if transcriptor_dir.exists():
        return
    if not download_missing:
        raise FileNotFoundError(
            f"Missing HeartTranscriptor checkpoint at {transcriptor_dir}. "
            f"Run setup_heartmula_checkpoints with --include-transcriptor or rerun with --download-missing-transcriptor."
        )
    result = download_hf_repo(HEARTTRANSCRIPTOR_REPO, transcriptor_dir)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or f"Failed to download {HEARTTRANSCRIPTOR_REPO}")


def _resolve_vocal_output(output_files: list[str], separated_dir: Path) -> Path:
    candidates = [Path(name) for name in output_files]
    vocal_candidates = [path for path in candidates if "vocal" in path.name.lower()]
    chosen = vocal_candidates[-1] if vocal_candidates else candidates[0]
    return (separated_dir / chosen).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate vocals and transcribe lyrics with HeartTranscriptor")
    parser.add_argument("audio_file")
    parser.add_argument("--output-dir", default=".tmp/lyrics_transcription")
    parser.add_argument("--separator-python", default=_default_separator_python())
    parser.add_argument("--separator-model", default=DEFAULT_SEPARATOR_MODEL)
    parser.add_argument("--separator-model-dir", default=None)
    parser.add_argument("--heart-python", default=_default_heart_python())
    parser.add_argument("--checkpoint-root", default=_default_checkpoint_root())
    parser.add_argument("--language", default=None)
    parser.add_argument("--skip-separation", action="store_true")
    parser.add_argument("--download-missing-transcriptor", action="store_true")
    args = parser.parse_args()

    configure_ffmpeg()

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root).resolve()
    _ensure_hearttranscriptor_checkpoint(checkpoint_root, download_missing=args.download_missing_transcriptor)

    if args.skip_separation:
        vocal_path = Path(args.audio_file).resolve()
        separation_payload = None
    else:
        separated_dir = output_root / "separated"
        separation_payload = _run_json_command(
            [
                args.separator_python,
                str(REPO_ROOT / "tools" / "audio" / "separate_vocals.py"),
                args.audio_file,
                "--output-dir",
                str(separated_dir),
                "--model",
                args.separator_model,
                *( ["--model-file-dir", args.separator_model_dir] if args.separator_model_dir else [] ),
            ],
            cwd=REPO_ROOT,
        )
        output_files = separation_payload.get("output_files") or []
        if not output_files:
            raise RuntimeError("Vocal separation completed without returning any output files.")
        vocal_path = _resolve_vocal_output(output_files, separated_dir)

    transcription_payload = _run_json_command(
        [
            args.heart_python,
            str(REPO_ROOT / "tools" / "voice" / "run_hearttranscriptor.py"),
            str(vocal_path),
            "--checkpoint-root",
            str(checkpoint_root),
            *( ["--language", args.language] if args.language else [] ),
        ],
        cwd=REPO_ROOT,
    )

    payload = {
        "audio_file": str(Path(args.audio_file).resolve()),
        "separator_model": None if args.skip_separation else args.separator_model,
        "separation": separation_payload,
        "vocal_file": str(vocal_path),
        "transcription": transcription_payload,
    }
    summary_path = output_root / f"{Path(args.audio_file).stem}_lyrics_transcription.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()