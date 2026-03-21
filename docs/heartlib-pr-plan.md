# HeartLib PR Plan

This document defines which HeartMuLa changes are reasonable to propose upstream and which ones should remain in AIMusicApp.

## Current State

- The `heartlib` submodule is clean and has no local code changes yet.
- AIMusicApp currently relies on HeartMuLa internals from `tools/ai/run_heartmula_backend.py`.
- The most important non-PR-safe behavior today is direct use of pipeline private methods and private attributes.

## Working Policy

- AIMusicApp is the source of truth for local app behavior and local patches.
- The `heartlib` fork should remain clean by default.
- If we experiment with changes that might become upstreamable later, they should go onto a dedicated feature branch in the `heartlib` fork rather than onto `main`.
- No `heartlib` fork change should be treated as required for AIMusicApp until we decide it is worth maintaining even if no upstream PR is ever accepted.
- We should assume any upstream PR may be declined, delayed, or substantially revised.

This policy keeps the app shippable even if upstream never accepts the changes we would prefer.

## Current Risk Points

These are the current integration points that should not become the long-term upstream contract:

- Calling `pipe._sanitize_parameters(...)`
- Calling `pipe._forward(...)`
- Calling `pipe.postprocess(...)` as part of a custom staged flow assembled outside the pipeline
- Reading `pipe._mula` and `pipe._codec` for diagnostics
- Monkey-patching `torchaudio.save` in the wrapper runner

Those patterns work locally, but they are weak PR material because they rely on implementation details rather than a stable public API.

## PR-Safe Change Criteria

Only propose heartlib changes that meet all of these rules:

1. The change is useful without AIMusicApp.
2. The change improves the public API instead of exposing more private internals.
3. The change does not hard-code this app's environment variables, GUI workflow, or file layout.
4. The change is testable in heartlib itself.
5. The change can be described as a general inference or usability improvement.

## Good Candidates For Upstream PRs

### 1. Public staged generation API

Goal:
Provide a supported way to generate latent frames, unload HeartMuLa, and decode later without requiring external callers to stitch together `preprocess`, `_forward`, and `postprocess` manually.

Recommended shape:

- Add a public method such as `generate_frames(...)`
- Add a public method such as `decode_frames(...)`
- Or add a public option on `__call__` or a new method such as `generate(..., staged_decode=True)`

Why it is PR-safe:

- It is a general memory-reduction feature.
- It matches the library's existing lazy-load design.
- It removes the need for wrappers to depend on private methods.

Acceptance bar:

- Single-GPU memory-saving use case is documented.
- Existing default behavior remains unchanged.
- Unit or integration coverage proves staged decode produces valid output.

### 2. Public pipeline diagnostics

Goal:
Expose basic model-load and CUDA-memory reporting through supported helpers instead of requiring wrappers to inspect `_mula` and `_codec` directly.

Recommended shape:

- Add a `get_component_status()` method
- Add a `get_memory_summary()` method
- Or add an optional callback / logger hook during load, forward, unload, and decode

Why it is PR-safe:

- It helps all downstream users debug memory issues.
- It avoids private-attribute scraping.

Acceptance bar:

- Diagnostics are optional and low-overhead.
- Output is stable enough to document.

### 3. Official CLI support for seed and memory-oriented flags

Goal:
Extend `examples/run_music_generation.py` with a small set of generally useful flags.

Reasonable flags:

- `--seed`
- `--codec_dtype`
- `--lazy_load`
- Potentially `--staged_decode` if the pipeline gets a public staged API

Why it is PR-safe:

- These are generic inference controls, not app-specific controls.

Acceptance bar:

- Example script stays simple.
- Flags map cleanly onto public pipeline behavior.

### 4. Clear low-memory guidance for `codec_dtype`

Goal:
Document and expose `codec_dtype` as an explicit low-memory inference control, with `bfloat16` treated as an opt-in tradeoff for users who need more VRAM headroom.

Why it is PR-safe:

- It is a runtime configuration improvement, not a wrapper-specific hack.
- It avoids creating or maintaining a second derived model artifact.
- It is easy for upstream to accept independently of any separate model-hosting decision.

Acceptance bar:

- Guidance is framed as an option with tradeoffs, not as a universal default.
- The example script and docs explain when users should prefer `float32` versus `bfloat16`.
- The recommendation is backed by observed memory savings from real runs.

## Borderline Candidates

These may be acceptable later, but only after the public API is cleaned up.

### 1. Configurable audio save backend

The current wrapper patches `torchaudio.save` because the local environment needed a more reliable save path.

This is only PR-safe if it is reframed as:

- a generic output backend choice
- or a robust fallback save path when torchaudio output fails

It is not PR-safe if it is just a wrapper-specific workaround with no tests.

### 2. CUDA allocator cap support

The per-process VRAM cap is useful for experiments, but it is more of a benchmarking or wrapper concern than a core library feature.

This should only move upstream if:

- there is clear demand from general users
- the implementation is cross-platform safe
- the API is documented as best-effort rather than a hard partition

## Keep In AIMusicApp

These should remain in the app repo, not in heartlib:

- Tkinter GUI code
- model comparison dashboard logic
- ratings, notes, and comparison summaries
- screenshot generation and README screenshots
- environment auto-detection for multiple local virtualenvs
- setup checkers for multiple external backends
- app-specific presets like `Fast Preset` and `Low-Memory Preset`
- benchmark harnesses comparing HeartMuLa against MelodyFlow and ACE-Step
- local output-directory conventions and app env vars

## Recommended PR Sequence

1. Open a small PR for a public staged decode API.
2. Open a follow-up PR for public diagnostics helpers if still needed.
3. Open a final small PR updating `examples/run_music_generation.py` to use the new public API.

This order matters because the example script should consume the public API rather than introducing more special-case code first.

## Proposed Implementation Strategy

Phase 1:
Refactor AIMusicApp so its runner prefers a public heartlib API when available.

Phase 2:
Implement the public staged API in the `heartlib` fork.

Phase 3:
Replace private method usage in `tools/ai/run_heartmula_backend.py`.

Phase 4:
Only after that, consider whether diagnostics belong upstream or should stay in the wrapper.

## Definition Of A Good HeartLib PR

A good PR should read like this:

- "Adds a supported low-memory staged decode path for single-GPU inference"
- "Adds optional pipeline diagnostics for component load state and memory"
- "Updates the official example script to expose supported inference controls"

A bad PR would read like this:

- "Adds AIMusicApp-specific flags and logging"
- "Exposes private state so our GUI can inspect it"
- "Adds wrapper-specific environment variable behavior"

## Immediate Recommendation

Do not patch heartlib yet.

First, keep AIMusicApp working as-is while we use this plan as the bar for any library edits. When we do touch the submodule, the first target should be a narrow, public staged-decode API that lets us remove private pipeline calls from the wrapper.

If an experiment is needed before that point, create it on a dedicated `heartlib` fork branch reserved for upstream candidates, and keep AIMusicApp functional without requiring that branch.

Also, do not plan around publishing a separate quantized HeartCodec model unless `bfloat16` and staged decode stop being sufficient. The cleaner upstream recommendation is to support and document low-memory runtime settings first.