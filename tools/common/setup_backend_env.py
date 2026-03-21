from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"

PROFILE_CONFIG = {
    "app": {
        "venv": ".venv",
        "editable_repo": True,
        "requirements": [],
        "checkout": None,
        "post_message": "App environment ready. Use launch_AIMusicApp.bat or aimusicapp-gui.",
    },
    "heartmula": {
        "venv": ".venv-heartmula",
        "editable_repo": False,
        "requirements": ["soundfile"],
        "checkout": THIRD_PARTY_ROOT / "heartlib",
        "post_message": "Set HEARTMULA_PYTHON to this venv's python.exe if you want to override defaults.",
    },
    "melodyflow": {
        "venv": ".venv-melodyflow",
        "editable_repo": False,
        "requirements": ["soundfile", "omegaconf", "transformers", "sentencepiece", "torchdiffeq", "xformers"],
        "checkout": None,
        "post_message": "Also clone the MelodyFlow Space checkout via bootstrap_AIMusicApp.bat --download-models or set MELODYFLOW_SPACE_DIR.",
    },
    "acestep": {
        "venv": ".venv-acestep",
        "editable_repo": False,
        "requirements": ["soundfile"],
        "checkout": THIRD_PARTY_ROOT / "ACE-Step",
        "post_message": "Set ACESTEP_PYTHON to this venv's python.exe if you want to override defaults.",
    },
    "acestep15": {
        "venv": ".venv-acestep15",
        "editable_repo": False,
        "requirements": ["soundfile"],
        "checkout": THIRD_PARTY_ROOT / "ACE-Step-1.5",
        "post_message": "Set ACESTEP15_PYTHON to this venv's python.exe if you want to override defaults.",
    },
}


def _venv_python_path(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _run(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> dict[str, object]:
    payload = {
        "command": command,
        "cwd": str((cwd or REPO_ROOT).resolve()),
        "dry_run": dry_run,
    }
    if dry_run:
        return {"ok": True, "result": payload}

    completed = subprocess.run(
        command,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    payload["returncode"] = completed.returncode
    payload["stdout"] = completed.stdout.strip()
    payload["stderr"] = completed.stderr.strip()
    if completed.returncode != 0:
        return {"ok": False, "result": payload}
    return {"ok": True, "result": payload}


def _create_venv(venv_root: Path, *, dry_run: bool) -> dict[str, object]:
    python_path = _venv_python_path(venv_root)
    if python_path.exists():
        return {"ok": True, "status": "present", "python": str(python_path)}
    result = _run([sys.executable, "-m", "venv", str(venv_root)], dry_run=dry_run)
    if not result["ok"]:
        return {"ok": False, "status": "failed", "details": result["result"]}
    return {"ok": True, "status": "created", "python": str(python_path)}


def _install_profile(profile: str, *, dry_run: bool) -> dict[str, object]:
    config = PROFILE_CONFIG[profile]
    venv_root = REPO_ROOT / config["venv"]
    steps: list[dict[str, object]] = []

    venv_result = _create_venv(venv_root, dry_run=dry_run)
    if not venv_result["ok"]:
        return {"profile": profile, "ok": False, "steps": steps, "venv": venv_result}

    python_path = _venv_python_path(venv_root)
    steps.append(_run([str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=dry_run))

    if config["editable_repo"]:
        steps.append(_run([str(python_path), "-m", "pip", "install", "-e", str(REPO_ROOT)], dry_run=dry_run))

    checkout = config["checkout"]
    if checkout is not None:
        if not checkout.exists():
            return {
                "profile": profile,
                "ok": False,
                "venv": venv_result,
                "steps": steps,
                "error": f"Required checkout is missing: {checkout}. Run bootstrap_AIMusicApp.bat first.",
            }
        steps.append(_run([str(python_path), "-m", "pip", "install", "-e", str(checkout)], dry_run=dry_run))

    if config["requirements"]:
        steps.append(_run([str(python_path), "-m", "pip", "install", *config["requirements"]], dry_run=dry_run))

    ok = all(step.get("ok", False) for step in steps)
    return {
        "profile": profile,
        "ok": ok,
        "venv": venv_result,
        "steps": steps,
        "python": str(python_path),
        "post_message": config["post_message"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a repo-local backend environment for AIMusicApp")
    parser.add_argument("profile", choices=sorted(PROFILE_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = _install_profile(args.profile, dry_run=args.dry_run)
    print(result)
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()