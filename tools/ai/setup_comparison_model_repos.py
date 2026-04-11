"""
tools/ai/setup_comparison_model_repos.py
Download official Hugging Face model repos for the comparison backends.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from tools.ai.download_hf_repo import download_hf_repo


DEFAULT_ACESTEP15_SHARED_ALLOW_PATTERNS = [
    "acestep-5Hz-lm-1.7B/*",
    "acestep-5Hz-lm-1.7B/**",
    "vae/*",
    "vae/**",
    "Qwen3-Embedding-0.6B/*",
    "Qwen3-Embedding-0.6B/**",
]


MODEL_REPO_MAP = {
    "audiox": {
        "downloads": [
            {
                "repo_id": "HKUSTAudio/AudioX-MAF",
                "local_subdir": "audiox/AudioX-MAF",
            }
        ],
    },
    "melodyflow": {
        "downloads": [
            {
                "repo_id": "facebook/melodyflow-t24-30secs",
                "local_subdir": "melodyflow/melodyflow-t24-30secs",
            }
        ],
    },
    "ace_step": {
        "downloads": [
            {
                "repo_id": "ACE-Step/ACE-Step-v1-3.5B",
                "local_subdir": "ace-step/ACE-Step-v1-3.5B",
            }
        ],
    },
    "ace_step_v15": {
        "downloads": [
            {
                "repo_id": "ACE-Step/Ace-Step1.5",
                "local_subdir": "ace-step-1.5/checkpoints",
                "allow_patterns": [
                    "acestep-v15-turbo/*",
                    "acestep-v15-turbo/**",
                    *DEFAULT_ACESTEP15_SHARED_ALLOW_PATTERNS,
                ],
            }
        ],
    },
    "ace_step_v15_turbo": {
        "downloads": [
            {
                "repo_id": "ACE-Step/Ace-Step1.5",
                "local_subdir": "ace-step-1.5/checkpoints",
                "allow_patterns": [
                    "acestep-v15-turbo/*",
                    "acestep-v15-turbo/**",
                    *DEFAULT_ACESTEP15_SHARED_ALLOW_PATTERNS,
                ],
            }
        ],
    },
    "ace_step_v15_sft": {
        "downloads": [
            {
                "repo_id": "ACE-Step/Ace-Step1.5",
                "local_subdir": "ace-step-1.5/checkpoints",
                "allow_patterns": DEFAULT_ACESTEP15_SHARED_ALLOW_PATTERNS,
            },
            {
                "repo_id": "ACE-Step/acestep-v15-sft",
                "local_subdir": "ace-step-1.5/checkpoints/acestep-v15-sft",
            },
        ],
    },
}


def download_comparison_model_repos(destination: str | Path, *, models: list[str], token: str | None = None, dry_run: bool = False) -> dict:
    destination_path = Path(destination)
    plan = []
    for model_name in models:
        spec = MODEL_REPO_MAP[model_name]
        for download_spec in spec["downloads"]:
            plan.append(
                {
                    "model": model_name,
                    "repo_id": download_spec["repo_id"],
                    "local_dir": str(destination_path / download_spec["local_subdir"]),
                    "allow_patterns": list(download_spec.get("allow_patterns") or []),
                }
            )

    deduped_plan = []
    seen = set()
    for item in plan:
        key = (item["repo_id"], item["local_dir"], tuple(item.get("allow_patterns") or []))
        if key in seen:
            continue
        seen.add(key)
        deduped_plan.append(item)

    if dry_run:
        return {"ok": True, "dry_run": True, "download_plan": deduped_plan}

    completed = []
    for item in deduped_plan:
        result = download_hf_repo(
            item["repo_id"],
            item["local_dir"],
            token=token,
            dry_run=False,
            allow_patterns=item.get("allow_patterns") or None,
        )
        if not result.get("ok"):
            result["download_plan"] = deduped_plan
            return result
        completed.append(item)
    return {"ok": True, "dry_run": False, "download_plan": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Hugging Face repos for comparison backends")
    parser.add_argument("--destination", default="./models/comparison")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REPO_MAP), default=sorted(MODEL_REPO_MAP))
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = download_comparison_model_repos(args.destination, models=args.models, token=args.token, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()