"""
tools/ai/run_acestep_backend.py
Run ACE-Step locally with a direct prompt-driven interface for the comparison harness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from acestep.pipeline_ace_step import ACEStepPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE-Step local inference")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--infer-step", type=int, default=60)
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--omega-scale", type=float, default=10.0)
    parser.add_argument("--scheduler-type", default="euler")
    parser.add_argument("--cfg-type", default="apg")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--overlapped-decode", action="store_true")
    return parser.parse_args()


def _merge_prompt_and_tags(prompt: str, tags: str) -> str:
    prompt_text = prompt.strip()
    tags_text = tags.strip()
    if not tags_text:
        return prompt_text
    if not prompt_text:
        return tags_text
    return f"{prompt_text}. {tags_text}"


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    audio = waveform.detach().to(torch.float32).cpu().transpose(0, 1).numpy()
    sf.write(path, audio, sample_rate)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_prompt = _merge_prompt_and_tags(args.prompt, args.tags)

    original_save = torchaudio.save

    def patched_save(uri, src, sample_rate, *patched_args, **patched_kwargs):
        del patched_args, patched_kwargs
        save_audio(Path(uri), src, sample_rate)

    torchaudio.save = patched_save
    try:
        pipeline = ACEStepPipeline(
            checkpoint_dir=str(Path(args.checkpoint_dir).resolve()),
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            torch_compile=args.torch_compile,
            cpu_offload=args.cpu_offload,
            overlapped_decode=args.overlapped_decode,
        )
        result = pipeline(
            audio_duration=max(1.0, float(args.duration)),
            prompt=merged_prompt,
            lyrics=args.lyrics,
            infer_step=args.infer_step,
            guidance_scale=args.guidance_scale,
            scheduler_type=args.scheduler_type,
            cfg_type=args.cfg_type,
            omega_scale=args.omega_scale,
            save_path=str(output_path),
            batch_size=1,
        )
    finally:
        torchaudio.save = original_save

    if not output_path.exists():
        raise SystemExit(f"ACE-Step finished without creating output: {output_path}")

    print(result)


if __name__ == "__main__":
    main()