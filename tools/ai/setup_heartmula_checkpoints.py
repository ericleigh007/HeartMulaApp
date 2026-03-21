"""
tools/ai/setup_heartmula_checkpoints.py
Download the recommended HeartMuLa Hugging Face checkpoints into a heartlib-style ckpt directory.
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

from tools.ai.music_backend_checks import (
    HEARTMULA_DEFAULT_CODEC_REPO,
    HEARTMULA_DEFAULT_GEN_REPO,
    HEARTMULA_DEFAULT_MODEL_REPO,
)
from tools.ai.download_hf_repo import download_hf_repo


def write_checkpoint_manifest(
    destination: str | Path,
    *,
    model_repo: str,
    codec_repo: str,
    gen_repo: str,
    transcriptor_repo: str | None = None,
) -> Path:
    destination_path = Path(destination)
    manifest_path = destination_path / "checkpoint_manifest.json"
    payload = {
        "model_repo": model_repo,
        "codec_repo": codec_repo,
        "gen_repo": gen_repo,
        "local_layout": {
            "root": str(destination_path),
            "model_dir": str(destination_path / "HeartMuLa-oss-3B"),
            "codec_dir": str(destination_path / "HeartCodec-oss"),
        },
    }
    if transcriptor_repo:
        payload["transcriptor_repo"] = transcriptor_repo
        payload["local_layout"]["transcriptor_dir"] = str(destination_path / "HeartTranscriptor-oss")
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def download_heartmula_checkpoints(
    destination: str | Path,
    *,
    model_repo: str = HEARTMULA_DEFAULT_MODEL_REPO,
    codec_repo: str = HEARTMULA_DEFAULT_CODEC_REPO,
    gen_repo: str = HEARTMULA_DEFAULT_GEN_REPO,
    transcriptor_repo: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> dict:
    destination_path = Path(destination)
    plan = [
        {"repo_id": gen_repo, "local_dir": str(destination_path)},
        {"repo_id": model_repo, "local_dir": str(destination_path / "HeartMuLa-oss-3B")},
        {"repo_id": codec_repo, "local_dir": str(destination_path / "HeartCodec-oss")},
    ]
    source_repos = {
        "model_repo": model_repo,
        "codec_repo": codec_repo,
        "gen_repo": gen_repo,
    }
    if transcriptor_repo:
        plan.append({"repo_id": transcriptor_repo, "local_dir": str(destination_path / "HeartTranscriptor-oss")})
        source_repos["transcriptor_repo"] = transcriptor_repo

    if dry_run:
        return {"ok": True, "download_plan": plan, "dry_run": True, "source_repos": source_repos}

    completed = []
    for item in plan:
        result = download_hf_repo(
            item["repo_id"],
            item["local_dir"],
            token=token,
            dry_run=False,
        )
        if not result.get("ok"):
            result["download_plan"] = plan
            return result
        completed.append(item)

    manifest_path = write_checkpoint_manifest(
        destination_path,
        model_repo=model_repo,
        codec_repo=codec_repo,
        gen_repo=gen_repo,
        transcriptor_repo=transcriptor_repo,
    )

    return {
        "ok": True,
        "dry_run": False,
        "download_plan": completed,
        "source_repos": source_repos,
        "expected_layout": {
            "root": str(destination_path),
            "entries": [
                "gen_config.json",
                "tokenizer.json",
                "HeartMuLa-oss-3B/",
                "HeartCodec-oss/",
                *( ["HeartTranscriptor-oss/"] if transcriptor_repo else [] ),
                manifest_path.name,
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the recommended HeartMuLa checkpoints from Hugging Face")
    parser.add_argument("--destination", default="./ckpt")
    parser.add_argument("--model-repo", default=HEARTMULA_DEFAULT_MODEL_REPO)
    parser.add_argument("--codec-repo", default=HEARTMULA_DEFAULT_CODEC_REPO)
    parser.add_argument("--gen-repo", default=HEARTMULA_DEFAULT_GEN_REPO)
    parser.add_argument("--transcriptor-repo", default=None)
    parser.add_argument("--include-transcriptor", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = download_heartmula_checkpoints(
        args.destination,
        model_repo=args.model_repo,
        codec_repo=args.codec_repo,
        gen_repo=args.gen_repo,
        transcriptor_repo=args.transcriptor_repo or ("HeartMuLa/HeartTranscriptor-oss" if args.include_transcriptor else None),
        token=args.token,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
