"""
tools/ai/run_audiox_backend.py
Run AudioX locally for text-to-music or audio-conditioned generation.

This script is intentionally minimal so the comparison CLI can call it as a
stable subprocess.

CLI:
  python tools/ai/run_audiox_backend.py --prompt "ambient synth pulse" \
      --output .tmp/audiox.wav --duration 10

Output JSON to stdout:
  {"success": true, "path": ".tmp/audiox.wav", "sample_rate": 44100}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def save_audio(path: Path, waveform, sample_rate: int) -> None:
    audio = waveform.detach().to("cpu").transpose(0, 1).numpy()
    sf.write(path, audio, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AudioX local inference")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="HKUSTAudio/AudioX-MAF")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--reference-audio", default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    args = parser.parse_args()

    import torch
    from einops import rearrange
    from audiox import get_pretrained_model
    from audiox.inference.generation import generate_diffusion_cond

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = get_pretrained_model(args.model_id)
    model = model.to(device)

    sample_rate = int(model_config["sample_rate"])
    sample_size = model_config["sample_size"]
    target_fps = int(model_config.get("video_fps", 25))
    requested_duration = max(1.0, float(args.duration))

    video_conditioner = None
    conditioner_module = getattr(model, "conditioner", None)
    conditioner_map = getattr(conditioner_module, "conditioners", None)
    if conditioner_map is not None and "video_prompt" in conditioner_map:
        video_conditioner = conditioner_map["video_prompt"]

    expected_frame_count = None
    if video_conditioner is not None:
        temp_pos_embedding = getattr(video_conditioner, "Temp_pos_embedding", None)
        if temp_pos_embedding is not None and temp_pos_embedding.ndim >= 2:
            expected_frame_count = int(temp_pos_embedding.shape[1])

    if expected_frame_count and target_fps > 0:
        conditioning_duration = expected_frame_count / float(target_fps)
    else:
        conditioning_duration = float(sample_size) / float(sample_rate)

    if args.reference_audio:
        from audiox.data.utils import load_and_process_audio
        audio_tensor = load_and_process_audio(args.reference_audio, sample_rate, 0, conditioning_duration)
    else:
        audio_tensor = torch.zeros((2, int(sample_rate * conditioning_duration)))

    frame_count = expected_frame_count or int(target_fps * conditioning_duration)
    video_tensors = torch.zeros((frame_count, 3, 224, 224), dtype=torch.float32)
    sync_features = torch.zeros((1, 240, 768), dtype=torch.float32, device=device)

    conditioning = [{
        "video_prompt": {"video_tensors": video_tensors.unsqueeze(0), "video_sync_frames": sync_features},
        "text_prompt": args.prompt,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": 0,
        "seconds_total": conditioning_duration,
    }]

    output = generate_diffusion_cond(
        model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32)
    output = output.div(torch.max(torch.abs(output)).clamp(min=1e-6))
    output = output.clamp(-1, 1)
    target_samples = int(sample_rate * requested_duration)
    if target_samples > 0:
        output = output[:, :target_samples]
    save_audio(output_path, output, sample_rate)

    print(json.dumps({"success": True, "path": str(output_path), "sample_rate": sample_rate}))


if __name__ == "__main__":
    main()