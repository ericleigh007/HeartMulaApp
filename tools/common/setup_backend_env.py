from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"

PROFILE_CONFIG = {
    "app": {
        "venv": ".venv",
        "editable_repo": True,
        "requirements": ["pytest"],
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
        "install_strategy": "uv-sync",
        "base_python_env": "ACESTEP15_BASE_PYTHON",
        "preferred_python_spec": "3.12",
        "min_python": (3, 11),
        "max_python_exclusive": (3, 13),
        "post_message": "ACE-Step 1.5 environment ready in third_party/ACE-Step-1.5/.venv. Set ACESTEP15_PYTHON to that python.exe if you want to override defaults.",
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


def _read_python_version(python_executable: str, *, dry_run: bool = False) -> tuple[int, int, int] | None:
    result = _run(
        [
            python_executable,
            "-c",
            "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))",
        ]
    )
    if not result["ok"]:
        return None
    stdout = str(result["result"].get("stdout", "")).strip()
    parts = stdout.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
        micro = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return (major, minor, micro)


def _python_version_is_compatible(
    version: tuple[int, int, int] | None,
    *,
    min_python: tuple[int, int] | None,
    max_python_exclusive: tuple[int, int] | None,
) -> bool:
    if version is None:
        return False
    major_minor = version[:2]
    if min_python is not None and major_minor < min_python:
        return False
    if max_python_exclusive is not None and major_minor >= max_python_exclusive:
        return False
    return True


def _format_python_requirement(config: dict[str, object]) -> str | None:
    min_python = config.get("min_python")
    max_python_exclusive = config.get("max_python_exclusive")
    if min_python is None and max_python_exclusive is None:
        return None
    fragments: list[str] = []
    if min_python is not None:
        fragments.append(f">={min_python[0]}.{min_python[1]}")
    if max_python_exclusive is not None:
        fragments.append(f"<{max_python_exclusive[0]}.{max_python_exclusive[1]}")
    return ", ".join(fragments)


def _resolve_base_python(profile: str, *, base_python: str | None, dry_run: bool) -> dict[str, object]:
    config = PROFILE_CONFIG[profile]
    min_python = config.get("min_python")
    max_python_exclusive = config.get("max_python_exclusive")
    requirement = _format_python_requirement(config)
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_candidate(source: str, value: str | None) -> None:
        if not value:
            return
        normalized = str(Path(value))
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append((source, normalized))

    add_candidate("--base-python", base_python)
    add_candidate(str(config.get("base_python_env") or ""), os.environ.get(str(config.get("base_python_env") or "")))
    add_candidate("AIMUSICAPP_BASE_PYTHON", os.environ.get("AIMUSICAPP_BASE_PYTHON"))
    add_candidate("sys.executable", sys.executable)

    rejections: list[str] = []
    for source, candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            rejections.append(f"{source}: not found at {candidate_path}")
            continue
        version = _read_python_version(candidate, dry_run=dry_run)
        if min_python is None and max_python_exclusive is None:
            return {"ok": True, "python": candidate, "source": source, "version": version}
        if _python_version_is_compatible(
            version,
            min_python=min_python,
            max_python_exclusive=max_python_exclusive,
        ):
            return {"ok": True, "python": candidate, "source": source, "version": version}
        version_text = "unknown" if version is None else ".".join(str(part) for part in version)
        rejections.append(f"{source}: Python {version_text} does not satisfy {requirement}")

    env_name = config.get("base_python_env")
    preferred_python_spec = config.get("preferred_python_spec")
    if config.get("install_strategy") == "uv-sync" and preferred_python_spec:
        return {
            "ok": True,
            "python": str(preferred_python_spec),
            "source": "uv-managed",
            "version": None,
        }
    if requirement is None:
        error = f"Unable to determine a base Python interpreter for profile '{profile}'."
    else:
        override_hint = f" Set {env_name} or pass --base-python <path-to-python.exe>." if env_name else " Pass --base-python <path-to-python.exe>."
        error = f"No compatible base Python interpreter found for profile '{profile}'. Required Python {requirement}.{override_hint}"
    return {"ok": False, "error": error, "rejections": rejections}


def _create_venv(venv_root: Path, *, base_python: str, recreate: bool, dry_run: bool) -> dict[str, object]:
    python_path = _venv_python_path(venv_root)
    if python_path.exists():
        if not recreate:
            return {"ok": True, "status": "present", "python": str(python_path), "base_python": base_python}
        if not dry_run:
            shutil.rmtree(venv_root)
    result = _run([base_python, "-m", "venv", str(venv_root)], dry_run=dry_run)
    if not result["ok"]:
        return {"ok": False, "status": "failed", "details": result["result"]}
    status = "recreated" if recreate else "created"
    return {"ok": True, "status": status, "python": str(python_path), "base_python": base_python}


def _resolve_uv_executable() -> str | None:
    candidates = [
        os.environ.get("UV_EXE"),
        shutil.which("uv"),
        str(Path.home() / ".local" / "bin" / "uv.exe"),
        str(Path.home() / ".cargo" / "bin" / "uv.exe"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "uv.exe") if os.environ.get("LOCALAPPDATA") else None,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = str(Path(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if Path(normalized).exists():
            return normalized
    return None


def _install_profile_with_uv_sync(
    profile: str,
    *,
    config: dict[str, object],
    base_python_result: dict[str, object],
    recreate: bool,
    dry_run: bool,
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    checkout = config["checkout"]
    if checkout is None or not Path(checkout).exists():
        return {
            "profile": profile,
            "ok": False,
            "steps": steps,
            "base_python": base_python_result,
            "error": f"Required checkout is missing: {checkout}. Run bootstrap_AIMusicApp.bat first.",
        }

    uv_executable = _resolve_uv_executable()
    if not uv_executable:
        return {
            "profile": profile,
            "ok": False,
            "steps": steps,
            "base_python": base_python_result,
            "error": "uv is required for ACE-Step 1.5 setup but was not found. Run third_party\\ACE-Step-1.5\\install_uv.bat first or set UV_EXE.",
        }

    project_venv_root = Path(checkout) / ".venv"
    python_path = _venv_python_path(project_venv_root)
    if recreate and project_venv_root.exists() and not dry_run:
        shutil.rmtree(project_venv_root)

    steps.append(
        _run(
            [uv_executable, "sync", "--python", str(base_python_result["python"])],
            cwd=Path(checkout),
            dry_run=dry_run,
        )
    )
    ok = all(step.get("ok", False) for step in steps)
    venv_status = "recreated" if recreate else ("present" if project_venv_root.exists() else "created")
    return {
        "profile": profile,
        "ok": ok,
        "base_python": base_python_result,
        "venv": {
            "ok": ok,
            "status": venv_status,
            "python": str(python_path),
            "base_python": str(base_python_result["python"]),
            "uv": uv_executable,
        },
        "steps": steps,
        "python": str(python_path),
        "post_message": config["post_message"],
    }


def _install_profile(profile: str, *, base_python: str | None, recreate: bool, dry_run: bool) -> dict[str, object]:
    config = PROFILE_CONFIG[profile]
    venv_root = REPO_ROOT / config["venv"]
    steps: list[dict[str, object]] = []

    base_python_result = _resolve_base_python(profile, base_python=base_python, dry_run=dry_run)
    if not base_python_result["ok"]:
        return {"profile": profile, "ok": False, "steps": steps, "error": base_python_result["error"], "base_python": base_python_result}

    if config.get("install_strategy") == "uv-sync":
        return _install_profile_with_uv_sync(
            profile,
            config=config,
            base_python_result=base_python_result,
            recreate=recreate,
            dry_run=dry_run,
        )

    venv_result = _create_venv(
        venv_root,
        base_python=str(base_python_result["python"]),
        recreate=recreate,
        dry_run=dry_run,
    )
    if not venv_result["ok"]:
        return {"profile": profile, "ok": False, "steps": steps, "venv": venv_result, "base_python": base_python_result}

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
        "base_python": base_python_result,
        "venv": venv_result,
        "steps": steps,
        "python": str(python_path),
        "post_message": config["post_message"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a repo-local backend environment for AIMusicApp")
    parser.add_argument("profile", choices=sorted(PROFILE_CONFIG))
    parser.add_argument("--base-python", help="Base Python interpreter to use when creating the virtual environment")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the target virtual environment before installing")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = _install_profile(args.profile, base_python=args.base_python, recreate=args.recreate, dry_run=args.dry_run)
    print(result)
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()