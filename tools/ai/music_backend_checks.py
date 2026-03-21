"""
tools/ai/music_backend_checks.py
Shared preflight validation helpers for local music model backends.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HEARTMULA_BASE_MODEL_REPO = "HeartMuLa/HeartMuLa-oss-3B"
HEARTMULA_HNY_MODEL_REPO = "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"
HEARTMULA_DEFAULT_MODEL_REPO = HEARTMULA_HNY_MODEL_REPO
HEARTMULA_DEFAULT_CODEC_REPO = "HeartMuLa/HeartCodec-oss-20260123"
HEARTMULA_DEFAULT_GEN_REPO = "HeartMuLa/HeartMuLaGen"
MELODYFLOW_DEFAULT_MODEL_REPO = "facebook/melodyflow-t24-30secs"
ACESTEP_DEFAULT_MODEL_REPO = "ACE-Step/ACE-Step-v1-3.5B"
ACESTEP15_DEFAULT_MODEL_REPO = "ACE-Step/Ace-Step1.5"


def _melodyflow_space_repo_candidates(configured: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    configured_value = (configured or "").strip()
    if configured_value:
        candidates.append(Path(configured_value).expanduser())
    candidates.extend(
        [
            REPO_ROOT / ".tmp" / "MelodyFlowSpace",
            REPO_ROOT / "third_party" / "MelodyFlowSpace",
        ]
    )
    return candidates


def find_melodyflow_space_repo(configured: str | None = None) -> Path | None:
    for candidate in _melodyflow_space_repo_candidates(configured):
        if candidate.joinpath("audiocraft", "models", "melodyflow.py").exists():
            return candidate.resolve()
    return None


def _collect_heartmula_variant_issues(
    issues: list[str],
    *,
    label: str,
    model_repo: str,
    heart_root: str,
    heart_ckpt: str,
    heart_python: str,
) -> None:
    if not heart_root or not Path(heart_root).exists():
        issues.append(f"{label}: set HeartMuLa Root to a local clone of HeartMuLa/heartlib.")
    if not heart_ckpt or not Path(heart_ckpt).exists():
        issues.append(
            f"{label}: set the checkpoint directory to a heartlib-style ckpt folder containing "
            f"{model_repo}, {HEARTMULA_DEFAULT_CODEC_REPO}, and {HEARTMULA_DEFAULT_GEN_REPO}."
        )
    if not python_exists(heart_python):
        issues.append(f"{label}: HeartMuLa Python must point to a valid python.exe.")
        return
    missing = find_missing_python_modules(heart_python, ["torch", "torchaudio", "soundfile", "heartlib"])
    if missing:
        issues.append(f"{label}: {heart_python} is missing modules: {', '.join(missing)}.")


def python_exists(python_executable: str) -> bool:
    candidate = (python_executable or "").strip()
    if not candidate:
        return False
    return Path(candidate).exists()


def resolve_python_executable(configured: str | None, *, fallback: str | None = None) -> str:
    candidate = (configured or "").strip()
    if candidate:
        return candidate
    return fallback or sys.executable


def find_missing_python_modules(python_executable: str, modules: list[str]) -> list[str]:
    command = [
        python_executable,
        "-c",
        (
            "import importlib.util as u; "
            f"mods={modules!r}; "
            "missing=[name for name in mods if u.find_spec(name) is None]; "
            "print('\\n'.join(missing))"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except Exception:
        return modules
    if completed.returncode != 0:
        return modules
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def get_python_runtime_info(python_executable: str) -> dict[str, object] | None:
    command = [
        python_executable,
        "-c",
        (
            "import json, platform, sys; "
            "print(json.dumps({"
            "'major': sys.version_info.major, "
            "'minor': sys.version_info.minor, "
            "'micro': sys.version_info.micro, "
            "'platform': sys.platform, "
            "'system': platform.system()"
            "}))"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return None


def collect_preflight_issues(models: list[str], settings: dict[str, str]) -> list[str]:
    issues: list[str] = []

    requested_heartmula_models = {name for name in models if name in {"heartmula", "heartmula_hny", "heartmula_base"}}
    if requested_heartmula_models:
        heart_root = settings.get("HEARTMULA_ROOT", "").strip()
        heart_python = resolve_python_executable(settings.get("HEARTMULA_PYTHON"))
        if "heartmula" in requested_heartmula_models or "heartmula_hny" in requested_heartmula_models:
            heart_hny_ckpt = settings.get("HEARTMULA_HNY_CKPT_DIR", "").strip() or settings.get("HEARTMULA_CKPT_DIR", "").strip()
            _collect_heartmula_variant_issues(
                issues,
                label="HeartMuLa Happy New Year",
                model_repo=HEARTMULA_HNY_MODEL_REPO,
                heart_root=heart_root,
                heart_ckpt=heart_hny_ckpt,
                heart_python=heart_python,
            )
        if "heartmula_base" in requested_heartmula_models:
            heart_base_ckpt = settings.get("HEARTMULA_BASE_CKPT_DIR", "").strip()
            _collect_heartmula_variant_issues(
                issues,
                label="HeartMuLa Base 3B",
                model_repo=HEARTMULA_BASE_MODEL_REPO,
                heart_root=heart_root,
                heart_ckpt=heart_base_ckpt,
                heart_python=heart_python,
            )

    if "audiox" in models:
        audiox_python = resolve_python_executable(settings.get("AUDIOX_PYTHON"))
        if not python_exists(audiox_python):
            issues.append("AudioX: AudioX Python must point to a valid python.exe.")
        else:
            runtime_info = get_python_runtime_info(audiox_python)
            if runtime_info:
                version_tuple = (int(runtime_info["major"]), int(runtime_info["minor"]))
                if version_tuple >= (3, 12):
                    issues.append(
                        "AudioX: Python 3.12+ is currently a poor fit for the upstream AudioX package on this machine; "
                        "use a Python 3.10 or 3.11 environment for backend setup."
                    )
            missing = find_missing_python_modules(audiox_python, ["torch", "torchaudio", "einops", "soundfile", "audiox"])
            if missing:
                issues.append(f"AudioX: {audiox_python} is missing modules: {', '.join(missing)}.")

    if "melodyflow" in models:
        melodyflow_python = resolve_python_executable(settings.get("MELODYFLOW_PYTHON"))
        melodyflow_model_dir = settings.get("MELODYFLOW_MODEL_DIR", "").strip()
        melodyflow_space_dir = settings.get("MELODYFLOW_SPACE_DIR", "").strip()
        if not python_exists(melodyflow_python):
            issues.append("MelodyFlow: MELODYFLOW_PYTHON must point to a valid python.exe.")
        else:
            runtime_info = get_python_runtime_info(melodyflow_python)
            if runtime_info:
                version_tuple = (int(runtime_info["major"]), int(runtime_info["minor"]))
                if version_tuple < (3, 10):
                    issues.append("MelodyFlow: use Python 3.10 or newer for backend setup.")
            missing = find_missing_python_modules(
                melodyflow_python,
                ["torch", "soundfile", "omegaconf", "transformers", "sentencepiece", "torchdiffeq", "audiocraft", "xformers"],
            )
            if missing:
                issues.append(f"MelodyFlow: {melodyflow_python} is missing modules: {', '.join(missing)}.")
        if melodyflow_model_dir and not Path(melodyflow_model_dir).exists():
            issues.append(f"MelodyFlow: MELODYFLOW_MODEL_DIR does not exist: {melodyflow_model_dir}.")
        elif not melodyflow_model_dir:
            issues.append(
                "MelodyFlow: set MELODYFLOW_MODEL_DIR to a local MelodyFlow checkpoint directory. "
                f"Recommended repo: {MELODYFLOW_DEFAULT_MODEL_REPO}."
            )
        melodyflow_space_repo = find_melodyflow_space_repo(melodyflow_space_dir)
        if melodyflow_space_repo is None:
            if melodyflow_space_dir:
                issues.append(
                    "MelodyFlow: MELODYFLOW_SPACE_DIR does not point to a checkout containing "
                    "audiocraft/models/melodyflow.py."
                )
            else:
                issues.append(
                    "MelodyFlow: clone the official MelodyFlow Space or your maintained fork into .tmp/MelodyFlowSpace, "
                    "third_party/MelodyFlowSpace, or set MELODYFLOW_SPACE_DIR to that checkout."
                )

    if "ace_step" in models and not settings.get("ACESTEP_COMMAND_TEMPLATE", "").strip():
        acestep_python = resolve_python_executable(settings.get("ACESTEP_PYTHON"))
        acestep_ckpt = settings.get("ACESTEP_CKPT_DIR", "").strip()
        if not python_exists(acestep_python):
            issues.append("ACE-Step: ACESTEP_PYTHON must point to a valid python.exe.")
        else:
            runtime_info = get_python_runtime_info(acestep_python)
            if runtime_info:
                version_tuple = (int(runtime_info["major"]), int(runtime_info["minor"]))
                if version_tuple < (3, 10):
                    issues.append("ACE-Step: use Python 3.10 or newer for backend setup.")
            missing = find_missing_python_modules(acestep_python, ["torch", "torchaudio", "soundfile", "acestep"])
            if missing:
                issues.append(f"ACE-Step: {acestep_python} is missing modules: {', '.join(missing)}.")
        if acestep_ckpt and not Path(acestep_ckpt).exists():
            issues.append(f"ACE-Step: ACESTEP_CKPT_DIR does not exist: {acestep_ckpt}.")
        elif not acestep_ckpt and not settings.get("ACESTEP_COMMAND_TEMPLATE", "").strip():
            issues.append(
                "ACE-Step: set ACESTEP_CKPT_DIR to a local ACE-Step checkpoint directory "
                f"or configure ACE-Step Template. Recommended repo: {ACESTEP_DEFAULT_MODEL_REPO}."
            )

    if "ace_step_v15" in models and not settings.get("ACESTEP15_COMMAND_TEMPLATE", "").strip():
        acestep15_root = settings.get("ACESTEP15_ROOT", "").strip()
        acestep15_python = resolve_python_executable(settings.get("ACESTEP15_PYTHON"))
        acestep15_ckpt = settings.get("ACESTEP15_CKPT_DIR", "").strip()
        if not acestep15_root or not Path(acestep15_root).exists():
            issues.append("ACE-Step 1.5: set ACESTEP15_ROOT to a local clone of ACE-Step-1.5.")
        if not python_exists(acestep15_python):
            issues.append("ACE-Step 1.5: ACESTEP15_PYTHON must point to a valid python.exe.")
        else:
            runtime_info = get_python_runtime_info(acestep15_python)
            if runtime_info:
                version_tuple = (int(runtime_info["major"]), int(runtime_info["minor"]))
                if version_tuple < (3, 11):
                    issues.append("ACE-Step 1.5: use Python 3.11 or newer for backend setup.")
            missing = find_missing_python_modules(acestep15_python, ["torch", "torchaudio", "soundfile", "acestep"])
            if missing:
                issues.append(f"ACE-Step 1.5: {acestep15_python} is missing modules: {', '.join(missing)}.")
        if acestep15_ckpt and not Path(acestep15_ckpt).exists():
            issues.append(f"ACE-Step 1.5: ACESTEP15_CKPT_DIR does not exist: {acestep15_ckpt}.")
        elif not acestep15_ckpt:
            issues.append(
                "ACE-Step 1.5: set ACESTEP15_CKPT_DIR to the local checkpoints directory "
                f"or configure ACE-Step 1.5 Template. Recommended repo: {ACESTEP15_DEFAULT_MODEL_REPO}."
            )

    return issues
