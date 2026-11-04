# Simulation validation

See [KERNEL_VALIDATION.md](KERNEL_VALIDATION.md) for measured full-resolution
performance, precision-oracle results, and the scope of the kernel implementation.

Use the default Julia environment with this repository registered through
`Pkg.develop(path="/path/to/HighOrderMRISim.jl")`. The compatibility baseline was
validated with Julia 1.12.7, KomaMRI 0.15.0, KomaMRIBase 0.15.2,
KomaMRICore 0.14.0, and CUDA.jl 6.4.2.

## CPU and GPU regression tests

```bash
julia --startup-file=no --threads=4 test/runtests.jl
CUDA_VISIBLE_DEVICES=<GPU-UUID> julia --startup-file=no --threads=4 test/test_gpu_compatibility.jl
```

The GPU test disables scalar indexing, checks actual CUDA arrays, and compares
CPU/GPU signals in both precisions with several high-order configurations.
GPU tests are separate from the default CPU suite; they require functional CUDA.

The dedicated kernel regression also covers every order mask, analytic phase
for all sixteen channels, RF frequency/phase, partial workgroups, and final
magnetization:

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> julia --startup-file=no --threads=4 test/test_gpu_kernels.jl
```

## Kernel execution

The public phantom, sequence, BlochHighOrder, and simulate interfaces are
preserved. Select an implementation through the simulation parameter dictionary:

- `"implementation" => "auto"` (default): use kernels for static CUDA
  BlochHighOrder simulations; otherwise use the reference array calculation.
- `"implementation" => "kernel"`: require the supported kernel configuration.
- `"implementation" => "array"`: retain the compatibility reference path.

Static coordinates are recognized by the default displacement functions.
Custom displacement functions, including custom zero-valued functions, remain
on the array path in auto mode and are rejected by an explicit kernel request.
No motion is silently ignored. CPU and ordinary Bloch() simulations retain the
array path. Higher-order terms remain disabled during RF excitation.

The kernels evolve each spin through a time block without constructing full
spin-by-time field and phase matrices. One reusable ADC-state buffer feeds the
existing complex multi-coil matrix multiplication; the CSM convention remains
transpose rather than complex conjugation.

## Full-resolution demos

Select one physical GPU by UUID before starting Julia. The process-local device
index is then zero. Set `HIGHORDERMRISIM_EXPECT_GPU_UUID` to the same UUID to
verify device selection. A headless matplotlib backend avoids GUI requirements.

```bash
export CUDA_VISIBLE_DEVICES=<GPU-UUID>
export HIGHORDERMRISIM_EXPECT_GPU_UUID=<GPU-UUID>
export MPLBACKEND=Agg
julia --startup-file=no --threads=4 test/validate_gpu_demos.jl 2d_nominal /path/to/output
julia --startup-file=no --threads=4 test/validate_gpu_demos.jl 2d_highorder /path/to/output
julia --startup-file=no --threads=4 test/validate_gpu_demos.jl 3d /path/to/output
```

The runner includes the existing demo scripts rather than duplicating their
input preparation. It disables plotting, artificial noise, reconstruction, and
optional reconstruction-data preparation. It preserves the simulation grids,
sequence events, physical effects, and number of shots. Temporal blocking
defaults to 32 and can be changed with `HIGHORDERMRISIM_DEMO_MAX_BLOCK_LENGTH`.
The validation runner defaults to an explicit kernel request. Set
`HIGHORDERMRISIM_DEMO_IMPLEMENTATION=array` to validate the reference path instead.

Each case saves a complex signal and input spin-density vector in MAT format,
plus a TOML summary containing the environment, GPU identity, dimensions, and
signal statistics. Keep generated results outside version control; directories
named `output/` are ignored.

Read back all three cases to audit saved dimensions, finite values, spin-density
lengths, and nonzero signal in every readout and coil:

```bash
julia --startup-file=no test/audit_gpu_demos.jl /path/to/output
```

## Validated compatibility baseline (checkpoint 8afdaeb)

On 2026-11-04, physical GPU2 (NVIDIA RTX 3090, 24 GiB) passed:

- CPU suite: 71 assertions.
- CUDA compatibility: 150 assertions.
- Full-resolution demo assertions: 5 per case.
- Independent saved-output audit: 24 assertions across all three cases.

| Case | Simulation grid | Precision | Signal shape |
| --- | --- | --- | --- |
| 2D nominal (`0000`) | 400 x 400 x 1 | f64 | 29,000 x 32 x 1 |
| 2D high-order (`1111`) | 400 x 400 x 1 | f64 | 29,000 x 32 x 1 |
| 3D high-order (`1111`) | 180 x 180 x 80 | f32 | 133,000 x 32 x 1 |

The 3D sequence contains 5 dummy and 35 acquired shots, with TR = 500 ms and a
total duration of 20 s. B0 and T1/T2 effects remain enabled. The 2D demo retains
its original disabled B0/T2 effects. Every saved signal was finite and had
nonzero signal in every acquired readout and coil.

This baseline uses the legacy array-based high-order GPU calculation, not a
specialized high-order kernel. Higher-order fields affect free precession only;
RF excitation retains nominal gradients. No speedup or whole-volume accuracy
claim follows from successful execution. Reconstruction, NRMSE/SSIM, matched
reconstruction grids, interactive plotting, and CI/docs deployment were not
validated in this run. In particular, the original 2D reconstruction grid is
200 x 200, whereas its simulation grid is 400 x 400.

## Performance and numerical comparisons

Full-demo comparisons run both implementations in the same Julia process.
Each path is warmed with the complete sequence and two spins; timed simulations
always use the original full-size phantom. Results include complex-signal L2
and peak differences, timings, allocation counts, and saved signals.

```bash
julia --startup-file=no --threads=4 test/benchmark_gpu_demos.jl 2d_nominal /path/to/output 2
julia --startup-file=no --threads=4 test/benchmark_gpu_demos.jl 2d_highorder /path/to/output 2
julia --startup-file=no --threads=4 test/benchmark_gpu_demos.jl 3d /path/to/output 1
```

Timings cover the simulate call, including sequence sampling and transfers, but
exclude input-file preparation, untimed warmup, package loading, and file writes.
One 3D sample is an observation, not a statistically robust benchmark.
Signal agreement with the array implementation is not reconstruction NRMSE.

For the supplied long 3D sequence, the f32 array baseline is not an exact oracle:
its measured peak difference from the full-volume f64 calculation is about
4.2e-4. The full-demo f32 array/kernel agreement bound is therefore 5e-4.
Independently, the kernel must agree with the f64 oracle within 1e-5 in both
relative signal L2 and normalized peak error. Small analytic and regression
tests retain their tighter precision-specific tolerances.

Generate an additional f64 reference in a separate directory, then compare the
two saved f32 outputs with it:

```bash
HIGHORDERMRISIM_DEMO_PRECISION=f64 julia --startup-file=no --threads=4 test/validate_gpu_demos.jl 3d /path/to/oracle
julia --startup-file=no test/compare_precision_oracle.jl /path/to/output /path/to/oracle/3d-signal.mat
```

Optional profiling tools are not package runtime dependencies. Install them in
a separate tooling environment, then expose it through
`HIGHORDERMRISIM_TOOL_ENV`. Keep the default Julia simulation environment active.

```bash
julia --startup-file=no -e 'using Pkg; Pkg.activate("output/tooling"); Pkg.add(["BenchmarkTools", "JET", "AllocCheck"])'
export HIGHORDERMRISIM_TOOL_ENV=/absolute/path/to/HighOrderMRISim.jl/output/tooling
julia --compiled-modules=existing --startup-file=no --threads=4 test/profile_highorder.jl /path/to/output 131072 both
julia --compiled-modules=existing --startup-file=no test/analyze_highorder.jl
julia --compiled-modules=existing --startup-file=no test/analyze_highorder.jl kernel
```

The profiling workload uses 32 time steps and 32 coils in both precisions,
warms each path, synchronizes GPU measurements, and captures host/device
profiles separately. The static analysis checks the scalar kernel mathematics
with JET and AllocCheck; it does not claim the complete simulation is
allocation-free. `--compiled-modules=existing` reuses available caches in the
same environment.
