"""
tools/ai/run_acestep15_backend.py
Run ACE-Step 1.5 locally through the upstream handler/inference API for the
comparison harness.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    raw = value.strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE-Step 1.5 local inference")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config-path", default="acestep-v15-turbo")
    parser.add_argument("--lm-model-path", default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--lm-backend", default="vllm")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--init-llm", default="false")
    parser.add_argument("--offload-to-cpu", default="false")
    return parser.parse_args()


def _merge_prompt_and_tags(prompt: str, tags: str) -> str:
    prompt_text = prompt.strip()
    tags_text = tags.strip()
    if not tags_text:
        return prompt_text
    if not prompt_text:
        return tags_text
    return f"{prompt_text}. {tags_text}"


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise SystemExit(f"ACE-Step 1.5 source root does not exist: {source_root}")
    if not checkpoints_dir.exists():
        raise SystemExit(f"ACE-Step 1.5 checkpoints directory does not exist: {checkpoints_dir}")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    init_llm = _parse_bool(args.init_llm, False)
    offload_to_cpu = _parse_bool(args.offload_to_cpu, False)
    normalized_lyrics = args.lyrics.strip()
    instrumental = normalized_lyrics.lower() in {"[instrumental]", "[inst]"}
    merged_caption = _merge_prompt_and_tags(args.prompt, args.tags)
    project_root = checkpoints_dir.parent

    dit_handler = AceStepHandler()
    init_status, init_ok = dit_handler.initialize_service(
        project_root=str(project_root),
        config_path=args.config_path,
        device=args.device,
        offload_to_cpu=offload_to_cpu,
    )
    if not init_ok:
        raise SystemExit(f"ACE-Step 1.5 DiT init failed: {init_status}")

    llm_handler = LLMHandler()
    if init_llm:
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=str(checkpoints_dir),
            lm_model_path=args.lm_model_path,
            backend=args.lm_backend,
            device=args.device,
            offload_to_cpu=offload_to_cpu,
            dtype=None,
        )
        if not lm_ok:
            raise SystemExit(f"ACE-Step 1.5 LM init failed: {lm_status}")

    params = GenerationParams(
        task_type="text2music",
        caption=merged_caption,
        lyrics="[Instrumental]" if instrumental else normalized_lyrics,
        instrumental=instrumental,
        duration=max(1.0, float(args.duration)),
        inference_steps=max(1, int(args.inference_steps)),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
        thinking=init_llm,
        use_cot_metas=init_llm,
        use_cot_caption=init_llm,
        use_cot_language=init_llm,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=args.seed < 0,
        seeds=None if args.seed < 0 else [int(args.seed)],
        audio_format="wav",
    )

    temp_output_dir = output_path.parent / f"{output_path.stem}_acestep15_tmp"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(temp_output_dir),
    )
    if not result.success or not result.audios:
        raise SystemExit(result.error or "ACE-Step 1.5 failed without producing audio.")

    produced_path = Path(result.audios[0].get("path") or "")
    if not produced_path.exists():
        raise SystemExit(f"ACE-Step 1.5 finished without creating output: {produced_path}")

    shutil.copy2(produced_path, output_path)
    print(
        json.dumps(
            {
                "status_message": result.status_message,
                "produced_path": str(produced_path),
                "output_path": str(output_path),
                "caption": merged_caption,
                "config_path": args.config_path,
                "init_llm": init_llm,
                "lm_model_path": args.lm_model_path if init_llm else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()