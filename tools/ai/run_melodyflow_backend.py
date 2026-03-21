"""
tools/ai/run_melodyflow_backend.py
Run MelodyFlow locally for prompt-driven text-to-music generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MelodyFlow local inference")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=0.0)
    parser.add_argument("--solver", choices=["midpoint", "euler", "dopri5"], default="midpoint")
    parser.add_argument("--solver-steps", type=int, default=64)
    parser.add_argument("--euler-steps", type=int, default=100)
    parser.add_argument("--use-euler", action="store_true")
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    waveform = waveform.detach().to(torch.float32)
    peak = waveform.abs().max().item()
    if peak > 1.0:
        waveform = waveform / peak
    waveform = waveform.clamp(-1.0, 1.0)
    audio = waveform.cpu().transpose(0, 1).numpy()
    sf.write(path, audio, sample_rate)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _melodyflow_space_repo_candidates() -> list[Path]:
    configured = os.environ.get("MELODYFLOW_SPACE_DIR", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            REPO_ROOT / ".tmp" / "MelodyFlowSpace",
            REPO_ROOT / "third_party" / "MelodyFlowSpace",
        ]
    )
    return candidates


def discover_melodyflow_space_repo() -> Path:
    for candidate in _melodyflow_space_repo_candidates():
        if candidate.joinpath("audiocraft", "models", "melodyflow.py").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Official MelodyFlow code was not found. Set MELODYFLOW_SPACE_DIR to your maintained fork checkout "
        "or clone the upstream Space with: "
        "git clone --depth 1 https://huggingface.co/spaces/facebook/MelodyFlow .tmp/MelodyFlowSpace"
    )


def _trusted_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _trusted_torch_load.original(*args, **kwargs)


_trusted_torch_load.original = torch.load


def load_melodyflow_model(model_dir: Path, device: torch.device):
    space_repo = discover_melodyflow_space_repo()
    if str(space_repo) not in sys.path:
        sys.path.insert(0, str(space_repo))

    original_load = torch.load
    torch.load = _trusted_torch_load
    try:
        from audiocraft.models import MelodyFlow

        model = MelodyFlow.get_pretrained(str(model_dir), device=str(device))
    finally:
        torch.load = original_load
    return model


def _normalize_solver(args: argparse.Namespace) -> tuple[str, int]:
    if args.use_euler or args.solver == "euler":
        return "euler", max(1, int(args.euler_steps))

    steps = max(2, int(args.solver_steps))
    if steps % 2 == 1:
        steps += 1
    return "midpoint", steps


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f"MelodyFlow model directory does not exist: {model_dir}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_melodyflow_model(model_dir, device)
    solver, steps = _normalize_solver(args)
    duration = max(1.0, float(args.duration))

    del args.cfg_scale, args.cfg_text_scale, args.ode_rtol, args.ode_atol

    model.set_generation_params(solver=solver, steps=steps, duration=duration)
    with torch.no_grad():
        generated = model.generate([args.prompt], progress=False, return_tokens=False)

    waveform = generated[0]
    save_audio(output_path, waveform, int(model.sample_rate))
    print(
        json.dumps(
            {
                "success": True,
                "path": str(output_path),
                "sample_rate": int(model.sample_rate),
                "solver": solver,
                "steps": steps,
                "space_repo": str(discover_melodyflow_space_repo()),
            }
        )
    )


if __name__ == "__main__":
    main()