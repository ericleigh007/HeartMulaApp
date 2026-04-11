"""
tools/ai/compare_music_models.py
Run the same music prompt across multiple model backends and save a structured
comparison report.

CLI:
  python tools/ai/compare_music_models.py \
      --prompt "dark ambient techno with metallic percussion" \
                --models heartmula_hny heartmula_base melodyflow ace_step ace_step_v15_turbo ace_step_v15_sft \
      --output-dir .tmp/music_compare/demo

Output JSON to stdout:
  {
    "success": true,
    "summary_path": ".tmp/music_compare/demo/comparison_summary.json",
    "results": [...]
  }
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

from tools.ai.music_model_backends import (
    MusicGenRequest,
    get_backend_registry,
    write_comparison_summary,
)


def _read_optional_text(file_path: str | None, inline_text: str | None) -> str | None:
    if inline_text is not None:
        return inline_text
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return None


def run_comparison(
    *,
    prompt: str,
    models: list[str],
    output_dir: str,
    duration: float = 15.0,
    reference_audio: str | None = None,
    lyrics: str | None = None,
    tags: str | None = None,
    seed: int | None = None,
) -> dict:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    request = MusicGenRequest(
        prompt=prompt,
        output_dir=str(output_dir_path),
        duration_seconds=duration,
        reference_audio=reference_audio,
        lyrics=lyrics,
        tags=tags,
        seed=seed,
    )

    registry = get_backend_registry()
    started = time.perf_counter()
    results = []
    for model_name in models:
        result = registry[model_name].run(request)
        results.append(result.to_dict())

    total_elapsed = round(time.perf_counter() - started, 3)
    success = any(result["success"] for result in results)
    summary = {
        "success": success,
        "prompt": prompt,
        "lyrics": lyrics,
        "tags": tags,
        "seed": seed,
        "models": models,
        "duration_seconds": duration,
        "reference_audio": reference_audio,
        "total_elapsed_seconds": total_elapsed,
        "results": results,
    }
    summary_path = write_comparison_summary(output_dir_path, summary)
    return {"success": success, "summary_path": str(summary_path), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple local music model backends")
    parser.add_argument("--prompt", required=True, help="Shared prompt used for every backend")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "heartmula",
            "heartmula_hny",
            "heartmula_base",
            "audiox",
            "melodyflow",
            "ace_step",
            "ace_step_v15",
            "ace_step_v15_turbo",
            "ace_step_v15_sft",
        ],
        default=["heartmula_hny", "heartmula_base", "melodyflow", "ace_step", "ace_step_v15_turbo", "ace_step_v15_sft"],
    )
    parser.add_argument("--output-dir", default=".tmp/music_compare/latest")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--reference-audio", default=None)
    parser.add_argument("--lyrics-file", default=None)
    parser.add_argument("--lyrics-text", default=None)
    parser.add_argument("--tags-file", default=None)
    parser.add_argument("--tags-text", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_comparison(
        prompt=args.prompt,
        models=args.models,
        output_dir=args.output_dir,
        duration=args.duration,
        reference_audio=args.reference_audio,
        lyrics=_read_optional_text(args.lyrics_file, args.lyrics_text),
        tags=_read_optional_text(args.tags_file, args.tags_text),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()