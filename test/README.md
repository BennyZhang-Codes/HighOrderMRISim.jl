# Simulation validation

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

Each case saves a complex signal and input spin-density vector in MAT format,
plus a TOML summary containing the environment, GPU identity, dimensions, and
signal statistics. Keep generated results outside version control; directories
named `output/` are ignored.

## Validated compatibility baseline

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
