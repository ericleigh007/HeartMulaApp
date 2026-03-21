"""
tools/ai/run_heartmula_backend.py
Run HeartMuLa generation and save the decoded waveform without depending on torchaudio's torchcodec path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from heartlib import HeartMuLaGenPipeline


def summarize_parameter_dtypes(module: torch.nn.Module) -> str:
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        key = str(parameter.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + parameter.numel()
    if not counts:
        return "no-parameters"
    items = sorted(counts.items(), key=lambda item: item[0])
    return ", ".join(f"{dtype}={count}" for dtype, count in items)


def summarize_loaded_module_dtypes(module: torch.nn.Module | None, label: str) -> str:
    if module is None:
        return f"{label}[deferred]"
    return f"{label}[{summarize_parameter_dtypes(module)}]"


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CUDA unavailable"
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return (
        f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, "
        f"max_allocated={max_allocated:.2f} GiB, max_reserved={max_reserved:.2f} GiB"
    )


def apply_cuda_memory_limit(device: torch.device, max_vram_gb: float | None) -> str | None:
    if max_vram_gb is None or max_vram_gb <= 0:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return "Requested VRAM limit ignored because CUDA is unavailable."
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    indexed_device = torch.device(f"cuda:{device_index}")
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    fraction = min(max(max_vram_gb / total_gb, 0.0), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, indexed_device)
    return f"Applied CUDA per-process limit: target={max_vram_gb:.2f} GiB, total={total_gb:.2f} GiB, fraction={fraction:.4f}"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "y", "true", "t", "1"}:
        return True
    if normalized in {"no", "n", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected. Got: {value}")


def str2dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise argparse.ArgumentTypeError(f"Dtype not recognized: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--version", default="3B")
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--tags", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--max-audio-length-ms", type=int, default=240_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--mula-device", default="cuda")
    parser.add_argument("--codec-device", default="cuda")
    parser.add_argument("--mula-dtype", type=str2dtype, default="bfloat16")
    parser.add_argument("--codec-dtype", type=str2dtype, default="float32")
    parser.add_argument("--lazy-load", type=str2bool, default=True)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    parser.add_argument("--stage-codec", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    audio = waveform.detach().to(torch.float32).cpu().transpose(0, 1).numpy()
    sf.write(path, audio, sample_rate)


def main() -> None:
    args = parse_args()
    original_save = torchaudio.save
    mula_device = torch.device(args.mula_device)

    if args.stage_codec and not args.lazy_load:
        print("Staged codec mode requires lazy_load=true; overriding lazy_load to true.")
        args.lazy_load = True

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if mula_device.type == "cuda" and torch.cuda.is_available():
        limit_message = apply_cuda_memory_limit(mula_device, args.max_vram_gb)
        if limit_message:
            print(limit_message)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(mula_device)

    def patched_save(uri, src, sample_rate, *patched_args, **patched_kwargs):
        del patched_args, patched_kwargs
        save_audio(Path(uri), src, sample_rate)

    torchaudio.save = patched_save
    try:
        pipe = HeartMuLaGenPipeline.from_pretrained(
            args.model_path,
            device={
                "mula": torch.device(args.mula_device),
                "codec": torch.device(args.codec_device),
            },
            dtype={
                "mula": args.mula_dtype,
                "codec": args.codec_dtype,
            },
            version=args.version,
            lazy_load=args.lazy_load,
        )
        print(
            "Loaded model dtypes: "
            f"{summarize_loaded_module_dtypes(pipe._mula, 'HeartMuLa')} | "
            f"{summarize_loaded_module_dtypes(pipe._codec, 'HeartCodec')}"
        )
        print(f"CUDA after load: {cuda_memory_summary(mula_device)}")
        with torch.no_grad():
            if args.stage_codec:
                preprocess_kwargs, forward_kwargs, _postprocess_kwargs = pipe._sanitize_parameters(
                    max_audio_length_ms=args.max_audio_length_ms,
                    save_path=args.save_path,
                    topk=args.topk,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                )
                model_inputs = pipe.preprocess(
                    {
                        "lyrics": args.lyrics,
                        "tags": args.tags,
                    },
                    **preprocess_kwargs,
                )
                model_outputs = pipe._forward(model_inputs, **forward_kwargs)
                model_outputs["frames"] = model_outputs["frames"].cpu()
                print(f"CUDA after HeartMuLa stage: {cuda_memory_summary(mula_device)}")
                print(
                    "Loaded model dtypes after HeartMuLa stage: "
                    f"{summarize_loaded_module_dtypes(pipe._mula, 'HeartMuLa')} | "
                    f"{summarize_loaded_module_dtypes(pipe._codec, 'HeartCodec')}"
                )
                pipe.postprocess(model_outputs, args.save_path)
            else:
                pipe(
                    {
                        "lyrics": args.lyrics,
                        "tags": args.tags,
                    },
                    max_audio_length_ms=args.max_audio_length_ms,
                    save_path=args.save_path,
                    topk=args.topk,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                )
            print(f"CUDA after generation: {cuda_memory_summary(mula_device)}")
            print(
                "Loaded model dtypes after generation: "
                f"{summarize_loaded_module_dtypes(pipe._mula, 'HeartMuLa')} | "
                f"{summarize_loaded_module_dtypes(pipe._codec, 'HeartCodec')}"
            )
    finally:
        torchaudio.save = original_save

    print(f"Generated music saved to {args.save_path}")
    print(
        "HeartMuLa settings: "
        f"cfg_scale={args.cfg_scale}, lazy_load={args.lazy_load}, "
        f"mula_dtype={args.mula_dtype}, codec_dtype={args.codec_dtype}, "
        f"max_vram_gb={args.max_vram_gb}, stage_codec={args.stage_codec}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()