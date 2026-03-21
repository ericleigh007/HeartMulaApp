"""
Run a single vocal-separation model against one audio file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from tools.audio.ffmpeg_runtime import configure_ffmpeg, ensure_pcm_wav_input


def _load_separator(model_file_dir: str | None, output_dir: str):
    try:
        from audio_separator.separator import Separator
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "python-audio-separator is not installed in this environment. "
            "Install it with `pip install \"audio-separator[gpu]\"`."
        ) from exc

    separator_kwargs = {
        "output_dir": output_dir,
        "output_format": "WAV",
        "use_soundfile": True,
    }
    if model_file_dir:
        separator_kwargs["model_file_dir"] = model_file_dir
    return Separator(**separator_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate vocals from one audio file")
    parser.add_argument("audio_file")
    parser.add_argument("--output-dir", default=".tmp/vocal_separation")
    parser.add_argument("--model-file-dir", default=None)
    parser.add_argument("--model", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--single-stem", default="Vocals")
    args = parser.parse_args()

    ffmpeg_binary = configure_ffmpeg()
    if not ffmpeg_binary:
        raise SystemExit("Could not find ffmpeg. Install ffmpeg or `imageio-ffmpeg`, or set FFMPEG_BINARY.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.audio_file).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input audio does not exist: {input_path}")

    prepared_input = ensure_pcm_wav_input(input_path, working_dir=output_root / "prepared")

    separator = _load_separator(args.model_file_dir, str(output_root))
    started = time.perf_counter()
    separator.load_model(model_filename=args.model)
    output_files = separator.separate(
        prepared_input,
        {args.single_stem: f"{input_path.stem}_{Path(args.model).stem}_{args.single_stem.lower()}"},
    )
    payload = {
        "audio_file": str(input_path),
        "prepared_audio_file": prepared_input,
        "model": args.model,
        "single_stem": args.single_stem,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "ffmpeg_binary": ffmpeg_binary,
        "output_files": output_files,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()