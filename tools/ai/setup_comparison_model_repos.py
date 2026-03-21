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


MODEL_REPO_MAP = {
    "audiox": {
        "repo_id": "HKUSTAudio/AudioX-MAF",
        "local_subdir": "audiox/AudioX-MAF",
    },
    "melodyflow": {
        "repo_id": "facebook/melodyflow-t24-30secs",
        "local_subdir": "melodyflow/melodyflow-t24-30secs",
    },
    "ace_step": {
        "repo_id": "ACE-Step/ACE-Step-v1-3.5B",
        "local_subdir": "ace-step/ACE-Step-v1-3.5B",
    },
    "ace_step_v15": {
        "repo_id": "ACE-Step/Ace-Step1.5",
        "local_subdir": "ace-step-1.5/checkpoints",
    },
}


def download_comparison_model_repos(destination: str | Path, *, models: list[str], token: str | None = None, dry_run: bool = False) -> dict:
    destination_path = Path(destination)
    plan = []
    for model_name in models:
        spec = MODEL_REPO_MAP[model_name]
        plan.append(
            {
                "model": model_name,
                "repo_id": spec["repo_id"],
                "local_dir": str(destination_path / spec["local_subdir"]),
            }
        )

    if dry_run:
        return {"ok": True, "dry_run": True, "download_plan": plan}

    completed = []
    for item in plan:
        result = download_hf_repo(item["repo_id"], item["local_dir"], token=token, dry_run=False)
        if not result.get("ok"):
            result["download_plan"] = plan
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