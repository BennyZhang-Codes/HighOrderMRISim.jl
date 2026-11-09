# 2D and 3D Demos

Each demo simulates a bundled spiral sequence and writes a MAT dataset for
reconstruction with [HighOrderMRI.jl](https://github.com/BennyZhang-Codes/HighOrderMRI.jl).
Run simulation first, then reconstruction. Both stages use the same
sequence-defined grid, receive maps, and field encoding.

## Install the pinned demo environment

Set up Python, Matplotlib, and the isolated Julia depot as in
[Getting Started](../getting-started.md). For the demos, use `demo/` instead of
the root Julia project. From the repository root, run:

```bash
julia --startup-file=no --project=demo -e 'using Pkg; Pkg.instantiate()'
```

The demo manifest pins HighOrderMRISim at `087f1b8` and HighOrderMRI at
`245e430`, including its `MRIGeometry` subpackage. The simulator pin predates
the version-only change to 0.2.0 and preserves the validated code. While the
simulator repository is private, its HTTPS URL requires GitHub credentials
with read access. If your credentials are configured through the system Git
helper, set `JULIA_PKG_USE_CLI_GIT=true` for the install command.

## Run simulation and reconstruction

Both stages default to `gpus = [0]`. Change `gpus` in the corresponding script
to use a different CUDA-visible device. The 3D simulation settings are in
`demo3D/3_run_sim.jl`. CPU simulation is also available by setting
`array_type = Array` and `gpus = []` in the simulation script; the documented
reconstruction scripts use CUDA.

### 2D spiral

```bash
julia --startup-file=no --project=demo --threads=4 demo2D/Sim2D.jl
julia --startup-file=no --project=demo --threads=4 demo2D/Recon2D.jl
```

The default simulation writes `output/2d/2d-0000-physical.mat`. Reconstruction
writes `output/2d/2d-0000-physical-lowrank-reconstruction.mat`.

### 3D spiral

```bash
julia --startup-file=no --project=demo --threads=4 demo3D/Sim3D.jl
julia --startup-file=no --project=demo --threads=8 demo3D/4_recon.jl
```

The default simulation writes `output/3d/3d-1111-physical.mat`. Reconstruction
writes `output/3d/3d-1111-physical-kernel-reconstruction.mat`.

| Demo | Native grid | Declared FOV | Simulation | Default operator |
| --- | --- | --- | --- | --- |
| 2D | 200 × 200 × 1 | 200 × 200 × 4 mm | Float64, `"0000"` | `HighOrderLowRankOp` |
| 3D | 100 × 100 × 35 | 200 × 200 × 70 mm | Float32, `"1111"` | Blocked `HighOrderKernelOp` |

Both scripts use CGNR with normalized sampling-density weights and zero L2
regularization. Edit `operator` in the reconstruction script to select
`"lowrank"` or `"kernel"`. Pass a dataset path as the sole script argument to
reconstruct a different export. LowRank requires centered Cartesian
coordinates on the active Fourier axes and equal samples per acquired readout.

## Validation and interpretation

The `"physical"` mode uses the demo's configured RF and relaxation settings.
In the 2D demo, B0 and T2 effects are disabled by default. In the multi-shot 3D
demo, RF and relaxation affect image contrast, so disagreement with unweighted
spin density is not an encoding-error measurement.

For a model-matched spin-density control, set `mode = "density"` in
`demo2D/Sim2D.jl` or `demo3D/3_run_sim.jl`, rerun simulation, then supply the
resulting file explicitly:

```bash
julia --startup-file=no --project=demo --threads=4 demo2D/Recon2D.jl \
    output/2d/2d-0000-density.mat
julia --startup-file=no --project=demo --threads=8 demo3D/4_recon.jl \
    output/3d/3d-1111-density.mat
```

Density mode disables T1 and T2 relaxation and prepares ideal transverse spin
density at each RF center, retaining the modeled phase through the RF tail.
It is a validation control, not a correction for physical data. The validated
reference results, without fitted intensity or phase scaling, were:

| Control | Magnitude NRMSE | Magnitude SSIM |
| --- | --- | --- |
| 2D spin density | 1.27% | 0.99188 |
| 3D spin density | 1.20% | 0.99520 |

The reconstruction MAT file contains `image`, `rho`, `mode`, and `metrics`.
Compare runs with the same mode, grid, field-order mask, and reconstruction
settings. These results validate the bundled workflows under the stated
conditions; they do not establish physical accuracy on an independent scanner.
