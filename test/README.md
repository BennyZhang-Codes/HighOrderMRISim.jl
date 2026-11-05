# Package tests

`test/` contains package regression tests and their shared inputs only.
Full-demo experiments, performance tools, and historical reports are not part of
the package test suite.

```text
test/
├── runtests.jl       # Default CPU suite
├── unit/             # CPU compatibility, phantom, coil, and image utilities
├── gpu/              # Explicit CUDA regressions and sanitizer workload
└── support/          # Shared deterministic inputs; no tests run on inclusion
```

## Environment

Run commands from the repository root using the default Julia environment, with
this checkout registered through `Pkg.develop(path="/path/to/HighOrderMRISim.jl")`.
The current compatibility baseline uses Julia 1.12.7, KomaMRI 0.15.0,
KomaMRIBase 0.15.2, KomaMRICore 0.14.0, and CUDA.jl 6.4.2.
Includes resolve relative to the scripts, so absolute entry-point paths also
work from other directories. No test activates another Julia environment.

## Default CPU suite

```bash
julia --startup-file=no --threads=4 test/runtests.jl
```

This is also the entry point used by `Pkg.test`. It does not require a CUDA
device and never includes the explicit GPU suite. It covers modern KomaMRI
compatibility, simulation input/state ownership, high-order sequence sampling,
coil sensitivity, phantom construction, and center cropping/padding.
Coil-map tests include analytic complex-field interpolation in Float32/Float64,
outer half-voxel face/edge/corner extension, zero output beyond the source FOV,
zero-norm preservation, and the previously blind real-map boundary region.

## Explicit GPU suite

Expose exactly one physical GPU by UUID before starting Julia. The process-local
device index is then zero. Replace the placeholder below with that GPU's UUID.

```bash
export CUDA_VISIBLE_DEVICES="<GPU-UUID>"
julia --startup-file=no --threads=4 test/gpu/runtests.jl
```

The runner executes the five kernel/compatibility regression files and prepared
simulation lifecycle checks serially. A missing/nonfunctional
GPU is an error, not a skipped test or a silent CPU fallback. Scalar indexing
is disabled. Shared inputs are loaded once without rerunning CPU tests.

Individual files remain runnable with the same GPU selection:

| File under `test/gpu/` | Coverage |
| --- | --- |
| `test_gpu_compatibility.jl` | CPU/GPU signals, execution selection, matrix/raw/state outputs, and motion fallback |
| `test_gpu_kernels.jl` | Every high-order mask, analytic field-channel phase, RF, partial workgroups, and packed/strided outputs |
| `test_gpu_reduction.jl` | Independent coil reduction, cancellation, empty inputs, and workspace reuse |
| `test_gpu_fused_coils.jl` | Warp/tiled fusion, ADC/coil tails, tile/group validation, and final magnetization |
| `test_gpu_rf_buffers.jl` | Actual RF ADC capacity, zero-capacity buffers, RF acquisition, block splitting, guarded outputs, and reuse |

```bash
julia --startup-file=no --threads=4 test/gpu/test_gpu_rf_buffers.jl
```

Both f32 and f64 are tested. Signal comparisons retain precision-specific
tolerances and verify final states and input ownership. Tests do not change the
simulation's cuBLAS default, enable higher-order RF terms, or relax precision.

## Explicit multi-GPU suite

Run this separately from the single-device suite, exposing only approved GPUs:

```bash
CUDA_VISIBLE_DEVICES="<GPU-UUID-1>,<GPU-UUID-2>" \
  julia --startup-file=no --threads=4 test/gpu/runtests_multigpu.jl
```

The suite checks both precisions and every coil path, finite T1/T2 history,
multiple RF pulses with ADC during RF, ordered final-state assembly, shard-local
callbacks, device/math-mode selection, immutable input parameters, raw output,
invalid IDs, and worker failure propagation. Full native-grid demo comparisons
and performance/reconstruction experiments remain outside `test/` in ignored
`local/`; they are not prerequisites for `Pkg.test`.

The single/multi-GPU runners also check prepared-plan reuse, independent returned
arrays, snapshot ownership, matrix/state/raw output, block/auto synchronization
equivalence, callback observation and state edits, recovery after callback
failure, recursion rejection, do-block cleanup, idempotent close and closed-plan
errors. Both precisions and all coil paths are covered. CPU sampled-storage
tests check concrete homogeneous storage and mixed-vector compatibility.

## Optional device sanitizers

With NVIDIA Compute Sanitizer installed and the same single-GPU selection:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 julia --startup-file=no --threads=4 test/gpu/sanitize_gpu_fused_coils.jl
compute-sanitizer --tool racecheck --error-exitcode 99 julia --startup-file=no --threads=4 test/gpu/sanitize_gpu_fused_coils.jl
```

The small sanitizer workload uses padded spins, partial ADC/coil tiles, guarded
outputs, and zero/nonzero RF buffer capacities. It is not automatically run by
either regression entry point and requires no profiling-package dependencies.
