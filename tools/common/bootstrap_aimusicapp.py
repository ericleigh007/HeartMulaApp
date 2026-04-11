from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
MODELS_ROOT = REPO_ROOT / "models"
TMP_ROOT = REPO_ROOT / ".tmp"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

HEARTLIB_REPO_URL = "https://github.com/ericleigh007/heartlib.git"
ACESTEP_REPO_URL = "https://github.com/ericleigh007/ACE-Step.git"
ACESTEP15_REPO_URL = "https://github.com/ericleigh007/ACE-Step-1.5.git"
MELODYFLOW_SPACE_REPO_ID = "facebook/MelodyFlow"
HEARTMULA_BASE_MODEL_REPO = "HeartMuLa/HeartMuLa-oss-3B"
DEFAULT_BOOTSTRAP_MODELS = ["heartmula_hny", "heartmula_base", "melodyflow", "ace_step", "ace_step_v15_turbo", "ace_step_v15_sft"]
BOOTSTRAP_MODEL_CHOICES = [*DEFAULT_BOOTSTRAP_MODELS, "ace_step_v15", "audiox"]


def _normalize_bootstrap_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if normalized == "ace_step_v15":
        return "ace_step_v15_turbo"
    return normalized


def _load_model_selection(model_config_path: str | None, cli_models: list[str] | None, *, include_audiox: bool) -> list[str]:
    selected_models = [_normalize_bootstrap_model_name(name) for name in (cli_models or DEFAULT_BOOTSTRAP_MODELS)]
    if model_config_path:
        payload = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            models_payload = payload.get("models", payload)
            if isinstance(models_payload, dict):
                selected_models = [_normalize_bootstrap_model_name(name) for name, enabled in models_payload.items() if enabled]
            elif isinstance(models_payload, list):
                selected_models = [_normalize_bootstrap_model_name(name) for name in models_payload]
            else:
                raise ValueError("Model config must contain a 'models' list or object.")
        elif isinstance(payload, list):
            selected_models = [_normalize_bootstrap_model_name(name) for name in payload]
        else:
            raise ValueError("Model config must be a list or object.")

    if include_audiox and "audiox" not in selected_models:
        selected_models.append("audiox")

    deduped = []
    seen = set()
    for model_name in selected_models:
        if model_name not in BOOTSTRAP_MODEL_CHOICES:
            raise ValueError(f"Unsupported bootstrap model selection: {model_name}")
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


def _venv_python_path() -> Path:
    if os.name == "nt":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


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


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_git_checkout(name: str, repo_url: str, destination: Path, *, dry_run: bool) -> dict[str, object]:
    if (destination / ".git").exists():
        fetch_result = _run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin"], dry_run=dry_run)
        if not fetch_result["ok"]:
            return {"name": name, "status": "failed", "details": fetch_result["result"]}
        pull_result = _run(["git", "-C", str(destination), "pull", "--ff-only"], dry_run=dry_run)
        if not pull_result["ok"]:
            return {"name": name, "status": "failed", "details": pull_result["result"]}
        return {"name": name, "status": "updated", "path": str(destination)}

    if destination.exists():
        return {
            "name": name,
            "status": "skipped-existing-nonrepo",
            "path": str(destination),
            "message": "Directory already exists but is not a git checkout.",
        }

    clone_result = _run(["git", "clone", "--depth", "1", repo_url, str(destination)], dry_run=dry_run)
    if not clone_result["ok"]:
        return {"name": name, "status": "failed", "details": clone_result["result"]}
    return {"name": name, "status": "cloned", "path": str(destination)}


def _ensure_venv(*, dry_run: bool) -> dict[str, object]:
    venv_python = _venv_python_path()
    if venv_python.exists():
        return {"status": "present", "python": str(venv_python)}

    result = _run([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv")], dry_run=dry_run)
    if not result["ok"]:
        return {"status": "failed", "details": result["result"]}
    return {"status": "created", "python": str(venv_python)}


def _install_requirements(venv_python: Path, *, dry_run: bool) -> list[dict[str, object]]:
    steps = []
    steps.append(_run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=dry_run))
    if PYPROJECT_PATH.exists():
        steps.append(_run([str(venv_python), "-m", "pip", "install", "-e", str(REPO_ROOT)], dry_run=dry_run))
    else:
        steps.append(_run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)], dry_run=dry_run))
    return steps


def _download_repo_local(
    venv_python: Path,
    repo_id: str,
    destination: Path,
    *,
    repo_type: str = "model",
    dry_run: bool,
) -> dict[str, object]:
    command = [
        str(venv_python),
        str(REPO_ROOT / "tools" / "ai" / "download_hf_repo.py"),
        repo_id,
        str(destination),
        "--repo-type",
        repo_type,
    ]
    if dry_run:
        command.append("--dry-run")
    result = _run(command, dry_run=dry_run)
    if not result["ok"]:
        return {"repo_id": repo_id, "status": "failed", "details": result["result"]}
    return {"repo_id": repo_id, "status": "downloaded" if not dry_run else "planned", "path": str(destination)}


def _download_models(venv_python: Path, *, selected_models: list[str], dry_run: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if "melodyflow" in selected_models:
        results.append(
            _download_repo_local(
                venv_python,
                MELODYFLOW_SPACE_REPO_ID,
                TMP_ROOT / "MelodyFlowSpace",
                repo_type="space",
                dry_run=dry_run,
            )
        )

    if "heartmula_hny" in selected_models:
        heart_hny_command = [
            str(venv_python),
            str(REPO_ROOT / "tools" / "ai" / "setup_heartmula_checkpoints.py"),
            "--destination",
            str(MODELS_ROOT / "heartmula" / "happy-new-year"),
            "--include-transcriptor",
        ]
        if dry_run:
            heart_hny_command.append("--dry-run")
        results.append({"repo_id": "HeartMuLa HNY bundle", "status": "planned" if dry_run else "invoked", "command": heart_hny_command})
        hny_result = _run(heart_hny_command, dry_run=dry_run)
        if not hny_result["ok"]:
            results.append({"repo_id": "HeartMuLa HNY bundle", "status": "failed", "details": hny_result["result"]})

    if "heartmula_base" in selected_models:
        heart_base_command = [
            str(venv_python),
            str(REPO_ROOT / "tools" / "ai" / "setup_heartmula_checkpoints.py"),
            "--destination",
            str(MODELS_ROOT / "heartmula" / "base"),
            "--model-repo",
            HEARTMULA_BASE_MODEL_REPO,
        ]
        if dry_run:
            heart_base_command.append("--dry-run")
        results.append({"repo_id": "HeartMuLa base bundle", "status": "planned" if dry_run else "invoked", "command": heart_base_command})
        base_result = _run(heart_base_command, dry_run=dry_run)
        if not base_result["ok"]:
            results.append({"repo_id": "HeartMuLa base bundle", "status": "failed", "details": base_result["result"]})

    comparison_models = [
        name
        for name in selected_models
        if name in {"audiox", "melodyflow", "ace_step", "ace_step_v15", "ace_step_v15_turbo", "ace_step_v15_sft"}
    ]
    if comparison_models:
        comparison_command = [
            str(venv_python),
            str(REPO_ROOT / "tools" / "ai" / "setup_comparison_model_repos.py"),
            "--destination",
            str(MODELS_ROOT / "comparison"),
            "--models",
            *comparison_models,
        ]
        if dry_run:
            comparison_command.append("--dry-run")
        results.append({"repo_id": "Comparison model repos", "status": "planned" if dry_run else "invoked", "command": comparison_command})
        comparison_result = _run(comparison_command, dry_run=dry_run)
        if not comparison_result["ok"]:
            results.append({"repo_id": "Comparison model repos", "status": "failed", "details": comparison_result["result"]})

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap AIMusicApp after a fresh clone")
    parser.add_argument("--skip-venv", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-loaders", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--models", nargs="+", choices=BOOTSTRAP_MODEL_CHOICES, default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--include-audiox", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_directory(THIRD_PARTY_ROOT)
    _ensure_directory(MODELS_ROOT)
    _ensure_directory(TMP_ROOT)

    summary: dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "dry_run": args.dry_run,
        "selected_models": None,
        "model_config": args.model_config,
        "venv": None,
        "requirements": [],
        "loader_checkouts": [],
        "model_downloads": [],
    }

    selected_models = _load_model_selection(args.model_config, args.models, include_audiox=args.include_audiox)
    summary["selected_models"] = selected_models

    if not args.skip_venv:
        venv_summary = _ensure_venv(dry_run=args.dry_run)
        summary["venv"] = venv_summary
        venv_python = _venv_python_path()
        if not args.skip_install:
            summary["requirements"] = _install_requirements(venv_python, dry_run=args.dry_run)
    else:
        venv_python = Path(sys.executable)
        summary["venv"] = {"status": "skipped", "python": str(venv_python)}

    if not args.skip_loaders:
        summary["loader_checkouts"] = [
            _ensure_git_checkout("heartlib", HEARTLIB_REPO_URL, THIRD_PARTY_ROOT / "heartlib", dry_run=args.dry_run),
            _ensure_git_checkout("ACE-Step", ACESTEP_REPO_URL, THIRD_PARTY_ROOT / "ACE-Step", dry_run=args.dry_run),
            _ensure_git_checkout("ACE-Step-1.5", ACESTEP15_REPO_URL, THIRD_PARTY_ROOT / "ACE-Step-1.5", dry_run=args.dry_run),
        ]

    if args.download_models:
        summary["model_downloads"] = _download_models(venv_python, selected_models=selected_models, dry_run=args.dry_run)

    summary_path = TMP_ROOT / "bootstrap_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    failure_detected = False
    for item in summary.get("requirements", []):
        if isinstance(item, dict) and not item.get("ok", True):
            failure_detected = True
    for section_name in ["loader_checkouts", "model_downloads"]:
        for item in summary.get(section_name, []):
            if isinstance(item, dict) and item.get("status") == "failed":
                failure_detected = True
    if failure_detected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()