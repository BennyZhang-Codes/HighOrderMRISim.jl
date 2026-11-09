# HighOrderMRISim.jl v0.2.0 — Draft release notes

This release brings higher-order Bloch simulation to the KomaMRI 0.15 stack and
adds CUDA execution paths and reproducible 2D and 3D simulation-to-reconstruction
demos. The legacy simulation baseline remains available as `v0.1.0`.

## Highlights

- Updated sequence sampling, RF/ADC handling, and simulator integration for the
  modern KomaMRI stack while retaining the existing high-order simulation entry
  points. Sampling now includes high-order waveform breakpoints, so legacy and
  upgraded signals are not expected to be bitwise identical.
- Added CUDA kernels for high-order precession and excitation, with optional
  fused coil reduction. cuBLAS remains the default for coil combination.
- Added explicit CPU, single-GPU, and multi-GPU selection via `array_type` and
  `gpus`. Multi-GPU simulation partitions spins, with each shard following the
  complete sequence. `prepare_simulation` reuses workers and buffers for repeated
  runs; single-use multi-GPU simulation is not necessarily faster than one GPU.
- Aligned the simulation and reconstruction demos to the sequence-native grid,
  fixed receive-map interpolation at outer half-voxel boundaries, and exported
  the actual grid and encoding used by the simulator. Reconstruction uses CGNR
  with normalized density compensation, defaulting to LowRank in 2D and the
  blocked kernel operator in 3D.

## Upgrade and reproduction

- Use Julia 1.12 for the tested setup. The package lock uses KomaMRI 0.15.0.
  HighOrderMRISim versions are independent of KomaMRI versions.
- Existing `sim_params["gpu"]` and `sim_params["gpu_device"]` options remain
  supported. The explicit `array_type` and `gpus` keywords take precedence.
  GPU IDs are zero-based among devices visible to the Julia process.
- Follow the [demo setup](README.md#run-the-demos) to instantiate the standalone
  lockfile. It uses the exact GPU-validated source snapshots of HighOrderMRISim
  (`087f1b8`, before the version-metadata change) and HighOrderMRI (`245e430`).
  Reconstruction dependencies are separate from the simulator package.

## Validation

- [Julia 1.12.7 CPU CI](https://github.com/BennyZhang-Codes/HighOrderMRISim.jl/actions/runs/42487659540)
  passed both `Pkg.instantiate()` and `Pkg.test()` (298/298 assertions) on the
  `v0.2.0` version commit. CUDA and multi-GPU package tests are run separately
  on NVIDIA hardware.
- The isolated demo acceptance run passed 96/96 checks for the 2D and 3D default
  physical workflows and their pure spin-density controls. For the pure-density
  controls, magnitude NRMSE / SSIM were 1.27% / 0.99188 (2D) and
  1.20% / 0.99520 (3D), with no fitted intensity or phase scaling.
- The default 3D physical demo retains RF and relaxation effects. Its contrast
  is not expected to match pure spin density.
