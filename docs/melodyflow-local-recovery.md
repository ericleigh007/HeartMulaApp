# MelodyFlow Local Recovery Notes

This note records the local MelodyFlow failure that produced loud buzz or hum instead of music, along with the fix that restored usable generation.

## Short Version

The bad local MelodyFlow output was not caused by WAV writing, prompt wording, or solver choice alone.

The root cause was that the local backend was trying to reconstruct MelodyFlow from an older local Audiocraft clone that did not actually contain the released MelodyFlow implementation. That custom path decoded the wrong latent contract and produced the characteristic garbage output.

The working fix was to stop using that reconstructed path and instead load the official MelodyFlow Hugging Face Space repo locally, then call the published `MelodyFlow` class directly.

## Symptoms

These were the recurring failure symptoms before the fix:

- generated MelodyFlow WAV files were structurally valid but sounded like buzz, hum, or "ka-ka"
- solver changes such as `dopri5` versus `midpoint` did not fix the output
- parameter-free latent split or channel-layout experiments also failed
- control files such as a 440 Hz reference tone played correctly, which ruled out the playback and save path

## What Actually Fixed It

The successful path was:

1. Clone the official MelodyFlow Space repo locally.
2. Keep using the local MelodyFlow model weights in `models/comparison/melodyflow/melodyflow-t24-30secs`.
3. Load the official `MelodyFlow` class from the cloned Space repo.
4. Apply a PyTorch 2.6 compatibility shim so checkpoint loads use `weights_only=False`.
5. Run generation through the official upstream implementation instead of the older custom local runner.

Clone command:

```powershell
git clone --depth 1 https://huggingface.co/spaces/facebook/MelodyFlow .tmp/MelodyFlowSpace
```

If you maintain your own fork, point `MELODYFLOW_SPACE_DIR` at that checkout and keep the official Space as the `upstream` remote so local fixes stay under your control while upstream changes remain easy to merge.

## Why The Older Local Path Failed

The older local backend was built on an Audiocraft clone that predated MelodyFlow support.

That older codebase did not include the official MelodyFlow pieces such as:

- the dedicated `MelodyFlow` model class
- the MelodyFlow-specific DiT loader path
- the official flow model schedule and generation logic used by the public Space
- the released `generate_audio` logic that distinguishes generator latents from direct VAE-style latents

Because of that mismatch, the custom local path was effectively guessing the latent-to-audio contract. The result was invalid decoded audio even though the files themselves were written correctly.

## PyTorch 2.6 Compatibility Note

The official MelodyFlow Space code predates the PyTorch 2.6 change that made `torch.load(..., weights_only=True)` the default.

Without a compatibility patch, loading the released MelodyFlow checkpoints can fail with an error similar to this:

```text
_pickle.UnpicklingError: Weights only load failed ...
Unsupported global: GLOBAL omegaconf.dictconfig.DictConfig
```

The local runner works around that by forcing checkpoint loads to use `weights_only=False` for trusted local model files.

## Working Local Components

The working local MelodyFlow path now depends on:

- `.venv-melodyflow`
- `.tmp/MelodyFlowSpace`
- optional `MELODYFLOW_SPACE_DIR` if you keep a maintained fork outside the default checkout path
- `models/comparison/melodyflow/melodyflow-t24-30secs`
- [tools/ai/run_melodyflow_backend.py](c:/Users/ericl/Documents/ai-agents/GitHub%20CoPilot/tools/ai/run_melodyflow_backend.py)
- [tools/ai/melodyflow_tiny_compare.py](c:/Users/ericl/Documents/ai-agents/GitHub%20CoPilot/tools/ai/melodyflow_tiny_compare.py)

## Verification Artifacts

These files were useful when verifying the recovery:

- `.tmp/melodyflow_backend_smoke.wav`
- `.tmp/melodyflow_tiny_compare/official_prompt_midpoint.wav`
- `.tmp/melodyflow_tiny_compare/official_prompt_euler.wav`
- `.tmp/melodyflow_tiny_compare/comparison_summary.json`

Useful checks:

- confirm the generated sample no longer has the old dominant hum profile
- confirm the tiny comparison bundle rebuilds successfully through the official path
- confirm the summary metrics look like ordinary audio rather than the earlier clipped garbage profile

## Current Scope

What is working now:

- local MelodyFlow text-to-music generation through the official upstream code path
- midpoint and euler solver control exposed through the local runner

What is not yet wired into the comparison GUI:

- reference-audio editing through MelodyFlow's regularized latent inversion flow

## Practical Guidance For Future Debugging

If MelodyFlow regresses again, check these in order:

1. Confirm the runner is importing MelodyFlow from `MELODYFLOW_SPACE_DIR`, `.tmp/MelodyFlowSpace`, or another maintained official-code checkout, not from the older `third_party/audiocraft` clone.
2. Confirm `torch.load` compatibility still forces `weights_only=False` for the trusted local checkpoints.
3. Confirm the model directory still contains both `state_dict.bin` and `compression_state_dict.bin`.
4. Re-run the tiny comparison script and compare the new `comparison_summary.json` against the last known good results.
5. Only investigate solver tuning after the import path and checkpoint-loading path are known to be correct.