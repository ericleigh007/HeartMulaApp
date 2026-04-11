"""
tools/ai/music_model_backends.py
Shared backend registry for local music model experiments.

Backends currently supported:
  - heartmula   : native adapter via the HeartMuLa heartlib example script
  - audiox      : native adapter via a local helper runner in this repo
    - melodyflow  : native adapter via a local helper runner in this repo
    - ace_step    : native adapter via a local ACE-Step runner in this repo
    - ace_step_v15: native adapter via a local ACE-Step 1.5 runner in this repo

The goal is to keep a stable comparison interface even when each model family has
different runtime requirements.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tools.ai.music_backend_checks import find_missing_python_modules, python_exists, resolve_python_executable


REPO_ROOT = Path(__file__).resolve().parents[2]
MELODYFLOW_DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "comparison" / "melodyflow" / "melodyflow-t24-30secs"
ACESTEP15_DEFAULT_SOURCE_ROOT = REPO_ROOT / "third_party" / "ACE-Step-1.5"
ACESTEP15_DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "models" / "comparison" / "ace-step-1.5" / "checkpoints"
ACESTEP15_TURBO_DEFAULT_CONFIG = "acestep-v15-turbo"
ACESTEP15_SFT_DEFAULT_CONFIG = "acestep-v15-sft"
ACESTEP15_SUPPORTED_TASKS = {"text2music", "cover", "repaint"}
ACESTEP15_TURBO_DEFAULT_STEPS = 8
ACESTEP15_SFT_DEFAULT_STEPS = 50
ACESTEP15_DEFAULT_LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"
ACESTEP15_DEFAULT_LM_BACKEND = "vllm"


@dataclass
class MusicGenRequest:
    prompt: str
    output_dir: str
    duration_seconds: float = 15.0
    reference_audio: str | None = None
    lyrics: str | None = None
    tags: str | None = None
    seed: int | None = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class MusicBackendResult:
    backend: str
    success: bool
    output_path: str | None = None
    elapsed_seconds: float | None = None
    command: list[str] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class MusicBackend:
    name = "backend"

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        raise NotImplementedError


def _run_subprocess(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 0,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=None if timeout <= 0 else timeout,
        check=False,
    )


def _ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_first_env_value(keys: list[str], default: str = "") -> str:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _parse_env_float_from_keys(keys: list[str], default: float) -> float:
    raw = _get_first_env_value(keys)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_env_optional_bool_from_keys(keys: list[str]) -> bool | None:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and raw.strip():
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return None


def _parse_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _parse_env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _classify_acestep15_config(config_path: str) -> str:
    normalized = (config_path or "").strip().lower()
    if "turbo" in normalized:
        return "turbo"
    if "sft" in normalized:
        return "sft"
    return "other"


def _compose_descriptor_text(prompt: str | None, tags: str | None) -> str:
    prompt_text = (prompt or "").strip()
    tags_text = (tags or "").strip()
    if prompt_text and tags_text:
        return f"{prompt_text}. {tags_text}"
    return prompt_text or tags_text


def _format_template(template: str, request: MusicGenRequest, output_path: Path) -> list[str]:
    substitutions = {
        "prompt": request.prompt,
        "output": str(output_path),
        "duration": str(request.duration_seconds),
        "reference_audio": request.reference_audio or "",
        "lyrics": request.lyrics or "",
        "tags": request.tags or request.prompt,
        "work_dir": request.output_dir,
        "python": sys.executable,
    }
    substitutions.update(request.extra)
    tokens = shlex.split(template.format(**substitutions), posix=False)
    normalized = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
            normalized.append(token[1:-1])
        else:
            normalized.append(token)
    return normalized


class HeartMuLaBackend(MusicBackend):
    def __init__(
        self,
        *,
        name: str = "heartmula",
        checkpoint_env: str = "HEARTMULA_CKPT_DIR",
        fallback_checkpoint_env: str | None = None,
        output_stem: str | None = None,
    ):
        self.name = name
        self.checkpoint_env = checkpoint_env
        self.fallback_checkpoint_env = fallback_checkpoint_env
        self.output_stem = output_stem or name

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        heart_root = os.environ.get("HEARTMULA_ROOT")
        ckpt_dir = os.environ.get(self.checkpoint_env)
        if (not ckpt_dir or not ckpt_dir.strip()) and self.fallback_checkpoint_env:
            ckpt_dir = os.environ.get(self.fallback_checkpoint_env)
        python_executable = resolve_python_executable(os.environ.get("HEARTMULA_PYTHON"), fallback=sys.executable)

        if not heart_root or not Path(heart_root).exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error="Set HEARTMULA_ROOT to a local clone of HeartMuLa/heartlib.",
            )

        if not ckpt_dir or not Path(ckpt_dir).exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error="Set HEARTMULA_CKPT_DIR to the local HeartMuLa checkpoint directory.",
            )

        if not python_exists(python_executable):
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"HEARTMULA_PYTHON points to a missing interpreter: {python_executable}",
            )

        output_dir = _ensure_dir(request.output_dir).resolve()
        output_path = (output_dir / f"{self.output_stem}_output.wav").resolve()
        descriptor_text = _compose_descriptor_text(request.prompt, request.tags)
        tags_path = _write_text(output_dir / "heartmula_tags.txt", (descriptor_text + "\n") if descriptor_text else "").resolve()
        lyrics_text = request.lyrics.strip() if request.lyrics is not None else ""
        lyrics_path = _write_text(output_dir / "heartmula_lyrics.txt", lyrics_text).resolve()
        cfg_scale = _parse_env_float("HEARTMULA_CFG_SCALE", 1.5)
        lazy_load = _parse_env_bool("HEARTMULA_LAZY_LOAD", True)
        codec_dtype = _parse_env_str("HEARTMULA_CODEC_DTYPE", "float32")
        max_vram_gb = _parse_env_float("HEARTMULA_MAX_VRAM_GB", 0.0)
        stage_codec = _parse_env_bool("HEARTMULA_STAGE_CODEC", False)

        command = [
            python_executable,
            str(REPO_ROOT / "tools" / "ai" / "run_heartmula_backend.py"),
            "--model-path",
            str(ckpt_dir),
            "--version",
            "3B",
            "--tags",
            str(tags_path),
            "--lyrics",
            str(lyrics_path),
            "--save-path",
            str(output_path),
            "--max-audio-length-ms",
            str(int(request.duration_seconds * 1000)),
            "--cfg-scale",
            str(cfg_scale),
            "--codec-dtype",
            codec_dtype,
            "--lazy-load",
            "true" if lazy_load else "false",
        ]
        if max_vram_gb > 0:
            command.extend(["--max-vram-gb", str(max_vram_gb)])
        if stage_codec:
            command.extend(["--stage-codec", "true"])

        started = time.perf_counter()
        try:
            completed = _run_subprocess(command, cwd=heart_root)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - started, 3)
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                elapsed_seconds=elapsed,
                error=stderr[:4000] or "HeartMuLa failed without producing output.",
            )

        return MusicBackendResult(
            backend=self.name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=elapsed,
            command=command,
            metadata={
                "stdout": completed.stdout[-2000:] if completed.stdout else "",
                "checkpoint_dir": str(Path(ckpt_dir).resolve()),
                "heart_root": str(Path(heart_root).resolve()),
                "python_executable": python_executable,
                "effective_descriptor_text": descriptor_text,
                "lyrics": request.lyrics or "",
                "tags": request.tags or "",
                "cfg_scale": cfg_scale,
                "lazy_load": lazy_load,
                "codec_dtype": codec_dtype,
                "max_vram_gb": max_vram_gb if max_vram_gb > 0 else None,
                "stage_codec": stage_codec,
            },
        )


class AudioXBackend(MusicBackend):
    name = "audiox"

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        output_dir = _ensure_dir(request.output_dir)
        output_path = output_dir / "audiox_output.wav"
        model_id = os.environ.get("AUDIOX_MODEL_ID", "HKUSTAudio/AudioX-MAF")
        python_executable = resolve_python_executable(os.environ.get("AUDIOX_PYTHON"), fallback=sys.executable)

        if not python_exists(python_executable):
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"AUDIOX_PYTHON points to a missing interpreter: {python_executable}",
            )

        required_modules = ["torch", "torchaudio", "einops", "soundfile", "audiox"]
        missing_modules = find_missing_python_modules(python_executable, required_modules)
        if missing_modules:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"AudioX Python environment is missing required modules: {', '.join(missing_modules)}. "
                    f"Set AUDIOX_PYTHON to an environment with those packages or install them into {python_executable}."
                ),
                metadata={"python_executable": python_executable, "missing_modules": missing_modules},
            )

        command = [
            python_executable,
            str(REPO_ROOT / "tools" / "ai" / "run_audiox_backend.py"),
            "--prompt",
            request.prompt,
            "--output",
            str(output_path),
            "--model-id",
            model_id,
            "--duration",
            str(request.duration_seconds),
        ]
        if request.reference_audio:
            command.extend(["--reference-audio", request.reference_audio])

        started = time.perf_counter()
        try:
            completed = _run_subprocess(command, cwd=REPO_ROOT)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - started, 3)
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                elapsed_seconds=elapsed,
                error=stderr[:4000] or "AudioX failed without producing output.",
            )

        return MusicBackendResult(
            backend=self.name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=elapsed,
            command=command,
            metadata={"stdout": completed.stdout[-2000:] if completed.stdout else ""},
        )


class ACEStepBackend(MusicBackend):
    name = "ace_step"

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        output_dir = _ensure_dir(request.output_dir).resolve()
        output_path = (output_dir / "ace_step_output.wav").resolve()
        checkpoint_dir = os.environ.get("ACESTEP_CKPT_DIR", str(REPO_ROOT / "models" / "comparison" / "ace-step" / "ACE-Step-v1-3.5B"))
        python_executable = resolve_python_executable(os.environ.get("ACESTEP_PYTHON"), fallback=sys.executable)

        if not python_exists(python_executable):
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"ACESTEP_PYTHON points to a missing interpreter: {python_executable}",
            )

        if not Path(checkpoint_dir).exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"Set ACESTEP_CKPT_DIR to a valid ACE-Step checkpoint directory: {checkpoint_dir}",
            )

        required_modules = ["torch", "torchaudio", "soundfile", "acestep"]
        missing_modules = find_missing_python_modules(python_executable, required_modules)
        if missing_modules:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"ACE-Step Python environment is missing required modules: {', '.join(missing_modules)}. "
                    f"Set ACESTEP_PYTHON to an environment with those packages or install them into {python_executable}."
                ),
                metadata={"python_executable": python_executable, "missing_modules": missing_modules},
            )

        command = [
            python_executable,
            str(REPO_ROOT / "tools" / "ai" / "run_acestep_backend.py"),
            "--prompt",
            request.prompt,
            "--tags",
            request.tags or "",
            "--output",
            str(output_path),
            "--checkpoint-dir",
            str(Path(checkpoint_dir).resolve()),
            "--duration",
            str(request.duration_seconds),
            "--lyrics",
            request.lyrics or "",
        ]

        started = time.perf_counter()
        try:
            completed = _run_subprocess(command, cwd=REPO_ROOT)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - started, 3)
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                elapsed_seconds=elapsed,
                error=stderr[:4000] or "ACE-Step failed without producing output.",
            )

        return MusicBackendResult(
            backend=self.name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=elapsed,
            command=command,
            metadata={
                "stdout": completed.stdout[-2000:] if completed.stdout else "",
                "prompt": request.prompt,
                "tags": request.tags or "",
                "lyrics": request.lyrics or "",
                "effective_descriptor_text": _compose_descriptor_text(request.prompt, request.tags),
            },
        )


class ACEStep15Backend(MusicBackend):
    def __init__(
        self,
        *,
        name: str = "ace_step_v15",
        output_stem: str | None = None,
        config_env: str = "ACESTEP15_CONFIG_PATH",
        default_config_path: str = ACESTEP15_TURBO_DEFAULT_CONFIG,
        env_prefix: str | None = None,
    ):
        self.name = name
        self.output_stem = output_stem or name
        self.config_env = config_env
        self.default_config_path = default_config_path
        self.env_prefix = env_prefix

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        output_dir = _ensure_dir(request.output_dir).resolve()
        output_path = (output_dir / f"{self.output_stem}_output.wav").resolve()
        source_root = Path(os.environ.get("ACESTEP15_ROOT", str(ACESTEP15_DEFAULT_SOURCE_ROOT))).resolve()
        checkpoints_dir = Path(os.environ.get("ACESTEP15_CKPT_DIR", str(ACESTEP15_DEFAULT_CHECKPOINTS_DIR))).resolve()
        python_executable = resolve_python_executable(os.environ.get("ACESTEP15_PYTHON"), fallback=sys.executable)
        config_path = os.environ.get(self.config_env, "").strip() or os.environ.get("ACESTEP15_CONFIG_PATH", "").strip() or self.default_config_path
        model_kind = _classify_acestep15_config(config_path)
        override_keys = lambda suffix: ([f"{self.env_prefix}_{suffix}"] if self.env_prefix else []) + [f"ACESTEP15_{suffix}"]
        lm_model_path = _get_first_env_value(override_keys("LM_MODEL_PATH"), ACESTEP15_DEFAULT_LM_MODEL_PATH)
        lm_backend = _get_first_env_value(override_keys("LM_BACKEND"), ACESTEP15_DEFAULT_LM_BACKEND)
        raw_task_type = os.environ.get("ACESTEP15_TASK_TYPE", "auto").strip().lower() or "auto"
        init_llm_override = _parse_env_optional_bool_from_keys(override_keys("INIT_LLM"))
        offload_to_cpu = _parse_env_bool("ACESTEP15_OFFLOAD_TO_CPU", False)
        default_inference_steps = ACESTEP15_TURBO_DEFAULT_STEPS if model_kind == "turbo" else ACESTEP15_SFT_DEFAULT_STEPS if model_kind == "sft" else ACESTEP15_TURBO_DEFAULT_STEPS
        default_guidance_scale = 1.0 if model_kind == "turbo" else 7.0
        inference_steps = max(1, int(_parse_env_float_from_keys(override_keys("INFERENCE_STEPS"), float(default_inference_steps))))
        guidance_scale = _parse_env_float_from_keys(override_keys("GUIDANCE_SCALE"), default_guidance_scale)
        repainting_start = _parse_env_float("ACESTEP15_REPAINT_START", 0.0)
        repainting_end = _parse_env_float("ACESTEP15_REPAINT_END", -1.0)
        audio_cover_strength = _parse_env_float("ACESTEP15_COVER_STRENGTH", 1.0)
        lm_model_dir = checkpoints_dir / lm_model_path
        init_llm = init_llm_override if init_llm_override is not None else model_kind in {"turbo", "sft"} and lm_model_dir.exists()
        task_type = "cover" if raw_task_type == "auto" and request.reference_audio else "text2music" if raw_task_type == "auto" else raw_task_type

        if task_type not in ACESTEP15_SUPPORTED_TASKS:
            supported = ", ".join(sorted(ACESTEP15_SUPPORTED_TASKS))
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"Unsupported ACE-Step 1.5 task type: {task_type}. "
                    f"This app currently exposes: {supported}."
                ),
            )

        if task_type in {"cover", "repaint"} and not request.reference_audio:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error="ACE-Step 1.5 cover and repaint require Reference Audio to point to a source audio file.",
            )

        if not python_exists(python_executable):
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"ACESTEP15_PYTHON points to a missing interpreter: {python_executable}",
            )

        if not source_root.exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"Set ACESTEP15_ROOT to a valid ACE-Step 1.5 source root: {source_root}",
            )

        if not checkpoints_dir.exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"Set ACESTEP15_CKPT_DIR to a valid ACE-Step 1.5 checkpoints directory: {checkpoints_dir}",
            )

        if init_llm and not lm_model_dir.exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"ACE-Step 1.5 LM is enabled but the LM checkpoint path is missing: {lm_model_dir}. "
                    "Download the shared ACE-Step 1.5 LM assets or set ACESTEP15_INIT_LLM=false."
                ),
                metadata={"lm_model_path": lm_model_path, "lm_model_dir": str(lm_model_dir)},
            )

        required_modules = ["torch", "torchaudio", "soundfile", "acestep"]
        missing_modules = find_missing_python_modules(python_executable, required_modules)
        if missing_modules:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"ACE-Step 1.5 Python environment is missing required modules: {', '.join(missing_modules)}. "
                    f"Set ACESTEP15_PYTHON to an environment with those packages or install them into {python_executable}."
                ),
                metadata={"python_executable": python_executable, "missing_modules": missing_modules},
            )

        command = [
            python_executable,
            str(REPO_ROOT / "tools" / "ai" / "run_acestep15_backend.py"),
            "--prompt",
            request.prompt,
            "--tags",
            request.tags or "",
            "--task-type",
            task_type,
            "--output",
            str(output_path),
            "--source-root",
            str(source_root),
            "--checkpoints-dir",
            str(checkpoints_dir),
            "--duration",
            str(request.duration_seconds),
            "--lyrics",
            request.lyrics or "",
            "--seed",
            str(request.seed if request.seed is not None else -1),
            "--config-path",
            config_path,
            "--lm-model-path",
            lm_model_path,
            "--lm-backend",
            lm_backend,
            "--inference-steps",
            str(inference_steps),
            "--guidance-scale",
            str(guidance_scale),
            "--init-llm",
            "true" if init_llm else "false",
            "--offload-to-cpu",
            "true" if offload_to_cpu else "false",
        ]
        if request.reference_audio:
            command.extend(["--source-audio", request.reference_audio])
        if task_type == "repaint":
            command.extend(
                [
                    "--repainting-start",
                    str(repainting_start),
                    "--repainting-end",
                    str(repainting_end),
                ]
            )
        if task_type in {"cover", "repaint"}:
            command.extend(["--audio-cover-strength", str(audio_cover_strength)])

        started = time.perf_counter()
        try:
            completed = _run_subprocess(command, cwd=source_root)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - started, 3)
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                elapsed_seconds=elapsed,
                error=stderr[:4000] or "ACE-Step 1.5 failed without producing output.",
            )

        return MusicBackendResult(
            backend=self.name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=elapsed,
            command=command,
            metadata={
                "stdout": completed.stdout[-2000:] if completed.stdout else "",
                "checkpoint_dir": str(checkpoints_dir),
                "python_executable": python_executable,
                "prompt": request.prompt,
                "tags": request.tags or "",
                "lyrics": request.lyrics or "",
                "effective_descriptor_text": _compose_descriptor_text(request.prompt, request.tags),
                "config_path": config_path,
                "model_kind": model_kind,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "lm_model_path": lm_model_path if init_llm else None,
                "init_llm": init_llm,
                "source_root": str(source_root),
                "task_type": task_type,
                "source_audio": request.reference_audio,
                "audio_cover_strength": audio_cover_strength if task_type in {"cover", "repaint"} else None,
                "repainting_start": repainting_start if task_type == "repaint" else None,
                "repainting_end": repainting_end if task_type == "repaint" else None,
            },
        )


class MelodyFlowBackend(MusicBackend):
    name = "melodyflow"

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        if request.reference_audio:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error="MelodyFlow reference-audio editing is not wired into this comparison runner yet; use text-only generation for now.",
            )

        output_dir = _ensure_dir(request.output_dir).resolve()
        output_path = (output_dir / "melodyflow_output.wav").resolve()
        model_dir = Path(os.environ.get("MELODYFLOW_MODEL_DIR", str(MELODYFLOW_DEFAULT_MODEL_DIR))).resolve()
        python_executable = resolve_python_executable(os.environ.get("MELODYFLOW_PYTHON"), fallback=sys.executable)
        cfg_scale = _parse_env_float("MELODYFLOW_CFG_SCALE", 3.0)
        cfg_text_scale = _parse_env_float("MELODYFLOW_CFG_TEXT_SCALE", 0.0)
        use_euler = _parse_env_bool("MELODYFLOW_USE_EULER", False)
        euler_steps = max(1, int(_parse_env_float("MELODYFLOW_EULER_STEPS", 100)))
        ode_rtol = _parse_env_float("MELODYFLOW_ODE_RTOL", 1e-5)
        ode_atol = _parse_env_float("MELODYFLOW_ODE_ATOL", 1e-5)

        if not python_exists(python_executable):
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"MELODYFLOW_PYTHON points to a missing interpreter: {python_executable}",
            )

        if not model_dir.exists():
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"Set MELODYFLOW_MODEL_DIR to a valid MelodyFlow checkpoint directory: {model_dir}",
            )

        required_modules = ["torch", "soundfile", "omegaconf", "transformers", "sentencepiece", "torchdiffeq", "audiocraft", "xformers"]
        missing_modules = find_missing_python_modules(python_executable, required_modules)
        if missing_modules:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"MelodyFlow Python environment is missing required modules: {', '.join(missing_modules)}. "
                    f"Set MELODYFLOW_PYTHON to an environment with those packages or install them into {python_executable}."
                ),
                metadata={"python_executable": python_executable, "missing_modules": missing_modules},
            )

        command = [
            python_executable,
            str(REPO_ROOT / "tools" / "ai" / "run_melodyflow_backend.py"),
            "--prompt",
            request.prompt,
            "--output",
            str(output_path),
            "--model-dir",
            str(model_dir),
            "--duration",
            str(request.duration_seconds),
            "--cfg-scale",
            str(cfg_scale),
            "--cfg-text-scale",
            str(cfg_text_scale),
            "--euler-steps",
            str(euler_steps),
            "--ode-rtol",
            str(ode_rtol),
            "--ode-atol",
            str(ode_atol),
        ]
        if use_euler:
            command.append("--use-euler")
        if request.seed is not None:
            command.extend(["--seed", str(request.seed)])

        started = time.perf_counter()
        try:
            completed = _run_subprocess(command, cwd=REPO_ROOT)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - started, 3)
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                elapsed_seconds=elapsed,
                error=stderr[:4000] or "MelodyFlow failed without producing output.",
            )

        return MusicBackendResult(
            backend=self.name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=elapsed,
            command=command,
            metadata={
                "stdout": completed.stdout[-2000:] if completed.stdout else "",
                "cfg_scale": cfg_scale,
                "cfg_text_scale": cfg_text_scale,
                "use_euler": use_euler,
                "euler_steps": euler_steps,
                "ode_rtol": ode_rtol,
                "ode_atol": ode_atol,
            },
        )


class CommandTemplateBackend(MusicBackend):
    def __init__(self, name: str, template_env: str, cwd_env: str | None = None):
        self.name = name
        self.template_env = template_env
        self.cwd_env = cwd_env

    def run(self, request: MusicGenRequest) -> MusicBackendResult:
        template = os.environ.get(self.template_env)
        if not template:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=(
                    f"Set {self.template_env} to a command template for this backend. "
                    "Available placeholders: {prompt}, {output}, {duration}, {reference_audio}, {lyrics}, {tags}, {work_dir}, {python}."
                ),
            )

        output_dir = _ensure_dir(request.output_dir)
        output_path = output_dir / f"{self.name}_output.wav"
        try:
            command = _format_template(template, request, output_path)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                error=f"Failed to format command template: {exc}",
            )

        cwd = os.environ.get(self.cwd_env) if self.cwd_env else None
        started = time.perf_counter()
        try:
            completed = _run_subprocess(command, cwd=cwd)
        except Exception as exc:
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - started, 3)
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            return MusicBackendResult(
                backend=self.name,
                success=False,
                command=command,
                elapsed_seconds=elapsed,
                error=stderr[:4000] or f"{self.name} failed without producing output.",
            )

        return MusicBackendResult(
            backend=self.name,
            success=True,
            output_path=str(output_path),
            elapsed_seconds=elapsed,
            command=command,
            metadata={"stdout": completed.stdout[-2000:] if completed.stdout else ""},
        )


def get_backend_registry() -> dict[str, MusicBackend]:
    return {
        "heartmula": HeartMuLaBackend(fallback_checkpoint_env="HEARTMULA_HNY_CKPT_DIR"),
        "heartmula_hny": HeartMuLaBackend(name="heartmula_hny", checkpoint_env="HEARTMULA_HNY_CKPT_DIR", fallback_checkpoint_env="HEARTMULA_CKPT_DIR"),
        "heartmula_base": HeartMuLaBackend(name="heartmula_base", checkpoint_env="HEARTMULA_BASE_CKPT_DIR"),
        "audiox": AudioXBackend(),
        "melodyflow": MelodyFlowBackend(),
        "ace_step": ACEStepBackend(),
        "ace_step_v15": ACEStep15Backend(),
        "ace_step_v15_turbo": ACEStep15Backend(
            name="ace_step_v15_turbo",
            output_stem="ace_step_v15_turbo",
            config_env="ACESTEP15_TURBO_CONFIG_PATH",
            default_config_path=ACESTEP15_TURBO_DEFAULT_CONFIG,
            env_prefix="ACESTEP15_TURBO",
        ),
        "ace_step_v15_sft": ACEStep15Backend(
            name="ace_step_v15_sft",
            output_stem="ace_step_v15_sft",
            config_env="ACESTEP15_SFT_CONFIG_PATH",
            default_config_path=ACESTEP15_SFT_DEFAULT_CONFIG,
            env_prefix="ACESTEP15_SFT",
        ),
    }


def write_comparison_summary(output_dir: str | Path, summary: dict) -> Path:
    output_dir = _ensure_dir(output_dir)
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path