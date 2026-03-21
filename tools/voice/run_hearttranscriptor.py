"""
Run HeartTranscriptor on a local audio file and emit JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path

import soundfile as sf
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from heartlib import HeartTranscriptorPipeline
from tools.audio.ffmpeg_runtime import configure_ffmpeg, ensure_pcm_wav_input


def _build_generation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 2,
        "task": "transcribe",
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.8,
        "temperature": (0.0, 0.1, 0.2, 0.4),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.4,
        "return_timestamps": args.return_timestamps,
    }
    if args.language:
        generation_kwargs["language"] = args.language
    return generation_kwargs


def _normalize_joined_text(text: str) -> str:
    return " ".join(text.split())


def _transcribe_manual_chunks(
    pipe: HeartTranscriptorPipeline,
    input_path: Path,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    generation_kwargs: dict[str, object],
) -> dict[str, object]:
    audio, sample_rate = sf.read(str(input_path), always_2d=True)
    total_frames = audio.shape[0]
    chunk_frames = max(1, int(chunk_seconds * sample_rate))
    overlap_frames = max(0, int(overlap_seconds * sample_rate))
    if overlap_frames >= chunk_frames:
        raise ValueError("chunk overlap must be smaller than chunk length")
    step_frames = chunk_frames - overlap_frames

    chunks: list[dict[str, object]] = []
    chunk_texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="hearttranscriptor_chunks_") as temp_dir:
        temp_root = Path(temp_dir)
        start_frame = 0
        chunk_index = 0
        while start_frame < total_frames:
            end_frame = min(total_frames, start_frame + chunk_frames)
            chunk_audio = audio[start_frame:end_frame]
            chunk_path = temp_root / f"chunk_{chunk_index:03d}.wav"
            sf.write(str(chunk_path), chunk_audio, sample_rate, subtype="PCM_16")
            chunk_result = pipe(str(chunk_path), **generation_kwargs)
            chunk_text = str(chunk_result.get("text", "")).strip()
            chunk_texts.append(chunk_text)
            chunks.append(
                {
                    "index": chunk_index,
                    "start_seconds": round(start_frame / sample_rate, 3),
                    "end_seconds": round(end_frame / sample_rate, 3),
                    "text": chunk_text,
                }
            )
            if end_frame >= total_frames:
                break
            start_frame += step_frames
            chunk_index += 1

    return {
        "text": _normalize_joined_text(" ".join(part for part in chunk_texts if part)),
        "chunks": chunks,
        "chunking": {
            "mode": "manual-overlap",
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "count": len(chunks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HeartTranscriptor on one local audio file")
    parser.add_argument("audio_file")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--return-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manual-chunk-seconds", type=float, default=0.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=1.0)
    args = parser.parse_args()

    ffmpeg_binary = configure_ffmpeg()
    if not ffmpeg_binary:
        raise SystemExit("Could not find ffmpeg. Install ffmpeg, install imageio-ffmpeg, or set FFMPEG_BINARY.")

    input_path = Path(args.audio_file).resolve()
    ckpt_root = Path(args.checkpoint_root).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input audio does not exist: {input_path}")
    if not ckpt_root.exists():
        raise SystemExit(f"Checkpoint root does not exist: {ckpt_root}")

    prepared_input_path = Path(
        ensure_pcm_wav_input(
            input_path,
            working_dir=Path(__file__).resolve().parents[2] / ".tmp" / "hearttranscriptor_prepared",
        )
    ).resolve()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        if dtype == torch.float16:
            dtype = torch.float32

    started = time.perf_counter()
    pipe = HeartTranscriptorPipeline.from_pretrained(
        str(ckpt_root),
        device=torch.device(args.device),
        dtype=dtype,
    )
    generation_kwargs = _build_generation_kwargs(args)
    with torch.no_grad():
        if args.manual_chunk_seconds > 0:
            result = _transcribe_manual_chunks(
                pipe,
                prepared_input_path,
                chunk_seconds=args.manual_chunk_seconds,
                overlap_seconds=args.chunk_overlap_seconds,
                generation_kwargs=generation_kwargs,
            )
        else:
            result = pipe(str(prepared_input_path), **generation_kwargs)
    payload = {
        "audio_file": str(input_path),
        "prepared_audio_file": str(prepared_input_path),
        "checkpoint_root": str(ckpt_root),
        "device": args.device,
        "dtype": args.dtype if args.device == "cuda" else "float32",
        "language": args.language,
        "return_timestamps": args.return_timestamps,
        "manual_chunk_seconds": args.manual_chunk_seconds,
        "chunk_overlap_seconds": args.chunk_overlap_seconds,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "result": result,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()