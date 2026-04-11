"""
tools/ai/check_music_backends.py
Validate local music backend setup without launching the GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from tools.ai.music_backend_checks import collect_preflight_issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Check music backend setup without launching the GUI")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["heartmula", "heartmula_hny", "heartmula_base", "audiox", "melodyflow", "ace_step", "ace_step_v15", "ace_step_v15_turbo", "ace_step_v15_sft"],
        default=["heartmula_hny", "heartmula_base", "melodyflow", "ace_step", "ace_step_v15_turbo", "ace_step_v15_sft"],
    )
    args = parser.parse_args()

    settings = {
        "HEARTMULA_ROOT": os.environ.get("HEARTMULA_ROOT", ""),
        "HEARTMULA_CKPT_DIR": os.environ.get("HEARTMULA_CKPT_DIR", ""),
        "HEARTMULA_HNY_CKPT_DIR": os.environ.get("HEARTMULA_HNY_CKPT_DIR", ""),
        "HEARTMULA_BASE_CKPT_DIR": os.environ.get("HEARTMULA_BASE_CKPT_DIR", ""),
        "HEARTMULA_PYTHON": os.environ.get("HEARTMULA_PYTHON", sys.executable),
        "AUDIOX_MODEL_ID": os.environ.get("AUDIOX_MODEL_ID", "HKUSTAudio/AudioX-MAF"),
        "AUDIOX_PYTHON": os.environ.get("AUDIOX_PYTHON", sys.executable),
        "MELODYFLOW_PYTHON": os.environ.get("MELODYFLOW_PYTHON", sys.executable),
        "MELODYFLOW_MODEL_DIR": os.environ.get("MELODYFLOW_MODEL_DIR", ""),
        "MELODYFLOW_SPACE_DIR": os.environ.get("MELODYFLOW_SPACE_DIR", ""),
        "ACESTEP_COMMAND_TEMPLATE": os.environ.get("ACESTEP_COMMAND_TEMPLATE", ""),
        "ACESTEP_CWD": os.environ.get("ACESTEP_CWD", ""),
        "ACESTEP_PYTHON": os.environ.get("ACESTEP_PYTHON", sys.executable),
        "ACESTEP_CKPT_DIR": os.environ.get("ACESTEP_CKPT_DIR", ""),
        "ACESTEP15_COMMAND_TEMPLATE": os.environ.get("ACESTEP15_COMMAND_TEMPLATE", ""),
        "ACESTEP15_CWD": os.environ.get("ACESTEP15_CWD", ""),
        "ACESTEP15_ROOT": os.environ.get("ACESTEP15_ROOT", ""),
        "ACESTEP15_PYTHON": os.environ.get("ACESTEP15_PYTHON", sys.executable),
        "ACESTEP15_CKPT_DIR": os.environ.get("ACESTEP15_CKPT_DIR", ""),
        "ACESTEP15_SFT_CONFIG_PATH": os.environ.get("ACESTEP15_SFT_CONFIG_PATH", ""),
    }
    issues = collect_preflight_issues(args.models, settings)
    payload = {
        "ok": not issues,
        "models": args.models,
        "issues": issues,
    }
    print(json.dumps(payload, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()