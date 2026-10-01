---
name: optimizing-euhedral-performance
description: Use when making Euhedral-Inference faster (a CUDA kernel, a frame or plan topology, lane placement, or host work at the token boundary), or when deciding whether a measured speedup is real enough to keep, commit, or put in a pull request.
---

# Optimizing Euhedral performance

## Overview

Treat every change as a hypothesis. Find where the time goes, change one thing, and prove its numerics. Then gate it against the previous commit in paired forks, and keep or revert it. Record the outcome either way.

Wins in this system are 0.3-1.5%. A single benchmark run on this desktop host cannot resolve them; only paired forks can.

Scripts are in `scripts/` next to this file. Run them as `bash scripts/X.sh` and `python3 scripts/X.py`.

## The loop

1. **Locate the time.** Profile before choosing a target.
2. **Make one change per commit.** If a commit holds two ideas, gate each idea separately.
3. **Prove the numerics** (see below).
4. **Gate it** against the previous commit.
5. **Keep it:** commit with the gate table and a `docs/FRAME_MODEL.md` entry, then push. **Reject it:** revert the code, but still record the result and its numbers in the doc.
6. **Before a PR:** run the end-to-end benchmark against `main`.

## 1. Locate the time

Before using the GPU, ask the user to stop `euhedral-inference-serve`. At the end, run `docker start euhedral-inference-serve` and wait until it reports healthy.

- **Decode:**
  - Run `bash scripts/profile.sh REPO OUT decode:64:128 d64`, then `python3 scripts/trace_report.py OUT/d64.sqlite`.
  - The report gives the token period, busy time, the idle gap at the token boundary, and exclusive time per kernel.
  - Rank by exclusive time. A kernel launched with programmatic dependent launch starts early and waits inside itself, so raw kernel durations overstate its cost.
  - A large boundary gap means host work is on the critical path: sampling, readback, admission.
- **Prefill:** profile with the same script, using `prefill:1024`.
- **One kernel:** use Nsight Compute, or an operator bench that rotates weight copies past L2. Weights that stay hot in L2 fake decode-shaped wins.

## 2. Experiment switches

For an A/B test of a switch, add a temporary environment toggle and mark it `// EXPERIMENT`. Gate both arms from the same build. Remove the toggle before committing.

## 3. Numerics

- **A changed result** (reordered accumulation, lower precision): keep the old kernel as an `_exact` twin, selected by host dispatch when `EUHEDRAL_EXACT=1` (`euhedral_cuda_exact_numerics()`). Bitwise route tests run under exact numerics. Never loosen a bitwise test to make a relaxed kernel pass.
- **Bound the relaxed kernel:**
  - **Operator error:** measure it against an FP64 or FP32 oracle in a native test.
  - **Model drift:** run `./gradlew :core:cudaIntegrationTest --tests '*RelaxedNumericsDrift*' --rerun`. You can tune it with `-Peuhedral.numerics.drift.steps`, `-Peuhedral.numerics.drift.prefix` and `-Peuhedral.numerics.drift.report`. Median hidden error, KL and top-1 must not grow with position.
  - With a 1024-token prefix, the drift test's window bound also fails on `main`; that limit is marginal there, not a regression.
- **Claimed-identical changes** (selection logic, refactors, file moves) need proof, not just green tests:
  - Write kernel tests against a host reference, including ties, NaN, infinities and unaligned addresses.
  - For moved or refactored CUDA, run `LD_LIBRARY_PATH=build/cuda-dev/linux-x64/runtime python3 scripts/ptx_compare.py OLD_SRC native/src old.cu=new/kernels.cu ...`. It must report no changed entries.
  - List only the NVRTC module roots (the `kernels.cu` files the host loads); each root compiles every header it includes, so moved headers need no mapping of their own.

## 4. Gate

1. **Build both arms.**
   - Control: create a worktree at the previous commit, which is HEAD while the change is still uncommitted (`git worktree add`). In it, run `rm -rf build/native && ./gradlew nativeBuild :benchmark:installDist`.
   - Candidate: the working tree, built the same way.
   - For an env-toggle experiment, use one tree with `CONTROL_ENV`/`CANDIDATE_ENV`.
2. **Run** `bash scripts/gate.sh OUT CONTROL CANDIDATE 6 'SCENARIOS'`, then `python3 scripts/summarize.py OUT`.
   - Decode changes: `"decode:64:128","decode:1024:128"`.
   - Prefill changes: `"prefill:64","prefill:256","prefill:1024"`.
   - Shared code: run both sets.
3. **Leave the host alone while it runs:** no builds, no tests, no other GPU work, and never rebuild a tree under test. CPU load also fails the lattice timing tests.
4. **Decide.** Keep a change only if it is positive on its target scenarios, the candidate is ahead in at least 5 of 6 forks, and nothing else regresses. Treat flat, mixed, or 4-of-6 results as a rejection.
5. **Attribute.** When a combined change wins, gate each half alone as well. Frame splits typically need lanes to pay off: gate split alone, spread lanes alone, and both together.

## 5. Record, validate, push

- **Commit message:** what changed, why, and the gate table (scenario, change, forks ahead). Add no Claude signature or model identifiers.
- **Docs:** `docs/FRAME_MODEL.md` gets the measurement, including rejected variants and their numbers.
- **Validate before pushing:**
  - `./gradlew spotlessCheck test nativeVerify nativePackage :api:bootJar`
  - `./gradlew cudaIntegrationTest --rerun`. Without `--rerun`, Gradle serves results from cache.
  - `cd native/tests && /home/brandon/src/Euhedral-Execution/.venv-cache-tuner/bin/python -m unittest discover -p 'test_*.py'`. Without numpy, the attention tests skip.
- **Push**, and watch CI after pushing.

## 6. Before a PR

1. Run `gate.sh` with a `main` worktree as control and 6 forks over `"decode:64:128","decode:1024:128","decode:4096:128","prefill:64","prefill:256","prefill:1024","prefill:2048"`.
2. Put the table in the PR body, with the per-change gates and the variants that were not kept.
3. Restart serving.

## Frames and lanes

- A frame boundary marks a causal or ownership boundary. Splitting a frame only exposes independent work; lanes are what overlap it.
- PATH placement keeps the critical chain on one lane. Cross-lane edges lose programmatic dependent launch.
- When storage is aliased, the plan must add hazard edges (`QwenExecutionPlan.withStorageHazards`).
- Euhedral has no scheduler. Workers take ready frames first come, first served, or the lattice routes a frame by its hash. Never write that Euhedral "schedules" or "picks a worker".

## Common mistakes

| Mistake | Instead |
|---|---|
| One benchmark run says +0.8%, so commit it | Gate it: 6 paired forks, at least 5 of 6 ahead |
| Microbenchmark with hot L2 weights | Rotate weight copies past L2 |
| Ranking decode kernels by raw duration | Rank by exclusive time from `trace_report.py` |
| Relaxing a bitwise test for a new kernel | Add an `_exact` twin and bound the relaxed kernel with the drift test |
| "Tests pass, so the move is neutral" | `ptx_compare.py` shows no changed entries |
| Gating a combined change only | Gate each half too; record the halves that lose |
| Rebuilding or running tests during a gate | Edit only; build after the gate finishes |
| `cudaIntegrationTest` green in 1 second | It came from cache; rerun with `--rerun` |
| Products test fails after adding a kernel file | Rebuild `nativePackage` |
| Recording only the wins | Record rejected variants and their numbers in `FRAME_MODEL.md` |
