from __future__ import annotations

from pathlib import Path

from tools.common import setup_backend_env


def test_python_version_is_compatible_for_acestep15_range() -> None:
    assert setup_backend_env._python_version_is_compatible(
        (3, 12, 11),
        min_python=(3, 11),
        max_python_exclusive=(3, 13),
    )
    assert not setup_backend_env._python_version_is_compatible(
        (3, 10, 14),
        min_python=(3, 11),
        max_python_exclusive=(3, 13),
    )
    assert not setup_backend_env._python_version_is_compatible(
        (3, 14, 3),
        min_python=(3, 11),
        max_python_exclusive=(3, 13),
    )


def test_resolve_base_python_accepts_explicit_compatible_path(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "python312.exe"
    candidate.write_text("", encoding="utf-8")

    monkeypatch.delenv("ACESTEP15_BASE_PYTHON", raising=False)
    monkeypatch.delenv("AIMUSICAPP_BASE_PYTHON", raising=False)
    monkeypatch.setattr(setup_backend_env, "_read_python_version", lambda python_executable, dry_run=False: (3, 12, 11))

    result = setup_backend_env._resolve_base_python("acestep15", base_python=str(candidate), dry_run=False)

    assert result["ok"]
    assert result["python"] == str(candidate)
    assert result["source"] == "--base-python"


def test_resolve_base_python_reports_incompatible_current_python(monkeypatch, tmp_path: Path) -> None:
    current = tmp_path / "python314.exe"
    current.write_text("", encoding="utf-8")

    original_config = setup_backend_env.PROFILE_CONFIG["acestep15"]
    monkeypatch.setitem(
        setup_backend_env.PROFILE_CONFIG,
        "acestep15",
        {
            **original_config,
            "install_strategy": "pip",
            "preferred_python_spec": None,
        },
    )
    monkeypatch.delenv("ACESTEP15_BASE_PYTHON", raising=False)
    monkeypatch.delenv("AIMUSICAPP_BASE_PYTHON", raising=False)
    monkeypatch.setattr(setup_backend_env.sys, "executable", str(current))
    monkeypatch.setattr(setup_backend_env, "_read_python_version", lambda python_executable, dry_run=False: (3, 14, 3))

    result = setup_backend_env._resolve_base_python("acestep15", base_python=None, dry_run=False)

    assert not result["ok"]
    assert "--base-python" in result["error"]
    assert any("3.14.3" in rejection for rejection in result["rejections"])


def test_install_profile_with_uv_sync_uses_checkout_venv(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "ACE-Step-1.5"
    checkout.mkdir()
    profile_config = dict(setup_backend_env.PROFILE_CONFIG["acestep15"])
    profile_config["checkout"] = checkout

    monkeypatch.setattr(setup_backend_env, "_resolve_uv_executable", lambda: "C:/tools/uv.exe")
    monkeypatch.setattr(
        setup_backend_env,
        "_run",
        lambda command, cwd=None, dry_run=False: {"ok": True, "result": {"command": command, "cwd": str(cwd), "dry_run": dry_run}},
    )

    result = setup_backend_env._install_profile_with_uv_sync(
        "acestep15",
        config=profile_config,
        base_python_result={"ok": True, "python": "C:/Python312/python.exe"},
        recreate=False,
        dry_run=False,
    )

    assert result["ok"]
    assert result["python"].endswith("ACE-Step-1.5\\.venv\\Scripts\\python.exe")
    assert result["steps"][0]["result"]["command"] == ["C:/tools/uv.exe", "sync", "--python", "C:/Python312/python.exe"]


def test_resolve_base_python_falls_back_to_uv_managed_spec(monkeypatch) -> None:
    monkeypatch.delenv("ACESTEP15_BASE_PYTHON", raising=False)
    monkeypatch.delenv("AIMUSICAPP_BASE_PYTHON", raising=False)
    monkeypatch.setattr(setup_backend_env.sys, "executable", "C:/Python314/python.exe")
    monkeypatch.setattr(setup_backend_env, "_read_python_version", lambda python_executable, dry_run=False: (3, 14, 3))

    result = setup_backend_env._resolve_base_python("acestep15", base_python=None, dry_run=False)

    assert result["ok"]
    assert result["python"] == "3.12"
    assert result["source"] == "uv-managed"


def test_app_profile_includes_pytest_for_regression_gate() -> None:
    assert "pytest" in setup_backend_env.PROFILE_CONFIG["app"]["requirements"]