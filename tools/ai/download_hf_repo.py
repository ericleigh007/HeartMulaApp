"""
tools/ai/download_hf_repo.py
Download a Hugging Face model or dataset repo to a local directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def download_hf_repo(
    repo_id: str,
    local_dir: str | Path,
    *,
    repo_type: str = "model",
    token: str | None = None,
    dry_run: bool = False,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> dict:
    destination = Path(local_dir)
    effective_token = (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or None
    payload = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_dir": str(destination),
        "uses_token": bool(effective_token),
        "allow_patterns": allow_patterns or [],
        "ignore_patterns": ignore_patterns or [],
    }

    if dry_run:
        return {"ok": True, "dry_run": True, "download": payload}

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        snapshot_download = huggingface_hub.snapshot_download
    except Exception as exc:
        return {
            "ok": False,
            "error": "huggingface_hub is required. Install it with `pip install huggingface_hub`.",
            "details": str(exc),
            "download": payload,
        }

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        token=effective_token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return {"ok": True, "dry_run": False, "download": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face repo to a local directory")
    parser.add_argument("repo_id")
    parser.add_argument("local_dir")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow", nargs="*", default=None)
    parser.add_argument("--ignore", nargs="*", default=None)
    args = parser.parse_args()

    result = download_hf_repo(
        args.repo_id,
        args.local_dir,
        repo_type=args.repo_type,
        token=args.token,
        dry_run=args.dry_run,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()