# High-order CUDA kernel validation

Date: 2026-11-04. Compatibility checkpoint: `8afdaeb`.

Status: complete; full-resolution simulations and numerical comparisons passed.

## Environment and scope

- Default Julia 1.12.7 environment, with this repository registered by Pkg.develop.
- KomaMRI 0.15.0, KomaMRIBase 0.15.2, KomaMRICore 0.14.0.
- CUDA.jl 6.4.2 and KernelAbstractions 0.9.45.
- Physical GPU2: NVIDIA GeForce RTX 3090, 24 GiB, exposed alone by UUID.
- Four Julia threads and four OpenBLAS threads; GPU execution uses one host
  simulation partition, as in the compatibility implementation.
- Temporal block length: 32. No spatial downsampling or shot truncation.

The public containers and simulation call are preserved. Static CUDA
BlochHighOrder simulations select the kernel in auto mode. Array execution
remains available explicitly and is the fallback for CPU, ordinary Bloch(),
and custom displacement functions. Unsupported explicit kernel requests error.

Free precession evaluates all enabled spatial terms inside the kernel and
accumulates trapezoidal phase per spin. RF uses the existing nominal-gradient
hard-pulse model, carrier frequency, rotating-frame phase, and relaxation order.
High-order terms are not newly enabled during RF. Reusable ADC and packed coil
buffers feed cuBLAS without large per-block field/phase matrices.

## Executable checks

| Check | Result |
| --- | --- |
| Normal package loading and precompilation | Passed |
| CPU suite | 71 assertions passed |
| GPU precession, RF, and packed-output tests | 252 assertions passed |
| GPU compatibility and execution-selection tests | 213 assertions passed |
| Full-resolution demo and array/kernel checks | 27 assertions passed |
| Saved-output readback audit | 24 assertions passed |
| Full-volume precision-oracle comparison | 10 assertions passed |
| Scalar kernel math: JET | No optimization errors reported |
| Scalar kernel math: AllocCheck | Zero allocation/dynamic-call sites |

Tests include all sixteen order masks, analytic phase for every field channel,
both precisions, complex coil maps, partial workgroups, empty acquisition
blocks, RF phase/frequency, block subdivision, state/raw outputs, parameter
ownership, and explicit rejection/fallback of unsupported configurations.
The static-analysis result applies to scalar kernel mathematics, not the
allocation behavior of the complete simulation.

## Full-resolution performance

Each implementation is warmed with the complete sequence and two spins.
Timed calls always use the full original phantom. Timing includes sequence
sampling and host/device transfers but excludes package loading, demo input
preparation, compilation warmup, and result-file writes. GPU work is synchronized.

| Case | Precision | Array median | Kernel median | Speedup | Repeats per path |
| --- | --- | ---: | ---: | ---: | ---: |
| 2D nominal, 400 x 400 x 1 | f64 | 71.174 s | 3.983 s | 17.87x | 2 |
| 2D high-order, 400 x 400 x 1 | f64 | 72.505 s | 4.293 s | 16.89x | 2 |
| 3D high-order, 180 x 180 x 80 | f32 | 737.540 s | 16.525 s | 44.63x | 1 |

These are observations on this hardware and these inputs, not universal speedup
guarantees or statistically extensive benchmarks. The 3D input has 1,687,512
spins, 32 coils, 5 dummy shots and 35 acquired shots, TR = 500 ms, B0/T1/T2
effects enabled, and 133,000 ADC samples. The 2D input has 102,704 spins,
32 coils, 29,000 ADC samples, and its original disabled B0/T2 effects.

## Signal accuracy and the f32 reference floor

The full 2D f64 relative signal L2 differences from the array path were
1.456e-14 (nominal) and 1.421e-14 (high-order).

An initial 3D f32 comparison failed its original 2e-4 normalized peak bound.
The bound was not simply increased: a full-volume f64 reference was computed
using the same supplied sequence, field data, phantom, and physical effects.
The supplied measured-field preparation itself remains unchanged.

| f32 implementation | Relative signal L2 difference from f64 | Normalized peak difference from f64 |
| --- | ---: | ---: |
| Compatibility array path | 1.8141e-4 | 4.1958e-4 |
| New kernel path | 2.9870e-6 | 1.3355e-6 |

For these inputs, the kernel is closer to the higher-precision result than the
legacy f32 calculation. A complete-sequence eight-spin CPU f64 cross-check also
agreed with the scalar kernel algorithm within 3.0e-14 relative L2.

Full-demo f32 array/kernel agreement now uses 5e-4 to account for the measured
array-reference floor. An independent full-volume precision-oracle test requires
the kernel's relative L2 and normalized peak errors to remain below 1e-5 and
not exceed the array-reference errors beyond that allowance. All 10 oracle
assertions passed. Small analytic and regression tolerances remain unchanged.
This is a numerical signal comparison, not reconstruction NRMSE or spin-density
accuracy. The f64 reference is a higher-precision implementation check, not an
independent physical simulator.

## Performance diagnosis and retained design

A direct mul! into a strided legacy signal view selected generic GPU matmul in
the installed CUDA stack. A representative f64 multiplication took about
33.8 ms into that view versus 0.98 ms into a dense output. A reusable packed
output restored cuBLAS and reduced the view-target operation to about 0.87 ms.
Dedicated tests verify both dense backing storage and correct strided writes.

Fusion/group-reduction prototypes were kept outside the package and did not
pass compilation validation. They are not shipped and provide no valid speed
measurements. This version intentionally retains the validated packed-GEMM
route; it does not claim that a fused reduction would be slower.

Intermediate diagnostic timings were not used as the final isolated 3D
performance measurement. Raw logs and MAT/TOML results are retained outside
version control. See README.md in this directory for reproduction commands.

Reconstruction, NRMSE/SSIM, interactive plotting, GPU backends other than CUDA,
native Koma motion/flow integration, and CI/docs deployment are outside this
validation. The original reconstruction grids and physical effects were not
changed to improve these results.
