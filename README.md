This repository is an extension of [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) (a Julia package for highly efficient MR simulations) for **MR simulation** that can incorporate **dynamic field changes** associated with the gradients throughout the sequence.

## Features

* Supports **up to 2nd or 3rd order** spherical harmonic terms.

## CPU, single-GPU, and multi-GPU simulation

```julia
using HighOrderMRISim, CUDA

sim_params = Dict{String,Any}("sim_method" => BlochHighOrder("1111"),
    "precision" => "f32", "return_type" => "mat", "coil_combination" => "gemm")
sys = Scanner()
sig_gpu = simulate(obj, hoseq, sys; sim_params, array_type=CuArray, gpus=[0])
sig_multi = simulate(obj, hoseq, sys; sim_params,
    array_type=CuArray, gpus=[0, 1])
sig_cpu = simulate(obj, hoseq, sys; sim_params, array_type=Array)
```

`gpus` contains zero-based **visible** CUDA device IDs. Select physical devices
with `CUDA_VISIBLE_DEVICES` before starting Julia. Explicit CUDA selection fails
if CUDA/devices are unavailable. Existing calls using `sim_params["gpu"]` and
`sim_params["gpu_device"]` still work; explicit keywords take precedence.
The upgraded package samples sequences with current KomaMRI and includes
high-order waveform breakpoints, so signals need not be bitwise identical to
those produced by the legacy KomaMRI 0.8 stack.

Multi-GPU execution partitions spins, not shots or time. Every spin follows the
complete sequence. Partial signals are summed in device-list order, so rounding
can differ from one GPU. Callbacks receive shard-local data; see `?simulate` for
the callback contract and synchronization options.

### Reuse workers and buffers

For repeated simulations of the same inputs, prepare an explicit snapshot:

```julia
prepare_simulation(obj, hoseq, sys; sim_params,
    array_type=CuArray, gpus=[0, 1]) do plan
    sig = simulate(plan; verbose=false)
    sig_repeat = simulate(plan; verbose=false)
end
```

Each run starts from the initial magnetization and returns independent CPU
output. Preparation is timed separately and is required again when inputs
change. The do-block closes the plan automatically; otherwise use `close(plan)`
in `finally`. See `?prepare_simulation` for the full ownership contract.

## Run the demos

The default Julia environment must contain this checkout and its dependencies.
Reconstruction additionally requires a HighOrderMRI checkout that provides
normalized density weighting and the selected encoding operator.
RegularizedLeastSquares, NFFT and NonuniformFFTs are also required by the
reconstruction demos. None of these are simulation-package requirements.

Run simulation and reconstruction as separate processes so device selection is
explicit. Replace UUID placeholders with the devices you intend to use.

```bash
CUDA_VISIBLE_DEVICES="<simulation-GPU-UUID>" MPLBACKEND=Agg \
  julia --threads=4 demo2D/Sim2D.jl

CUDA_VISIBLE_DEVICES="<simulation-GPU-UUID>" MPLBACKEND=Agg \
  julia --threads=4 demo3D/Sim3D.jl

CUDA_VISIBLE_DEVICES="<reconstruction-GPU-UUID>" MPLBACKEND=Agg \
  julia --threads=4 demo2D/Recon2D.jl

CUDA_VISIBLE_DEVICES="<reconstruction-GPU-UUID>" MPLBACKEND=Agg \
  julia --threads=8 demo3D/4_recon.jl
```

The scripts use defaults defined in Julia. Simulation defaults to one CUDA
device and cuBLAS coil combination, with order mask `0000` and Float64 for 2D
and `1111` and Float32 for 3D. Edit `gpus` in the simulation or reconstruction
script to use additional visible devices, for example `[0, 1]`. For CPU
simulation, set `array_type = Array` and `gpus = []`. Set
`simulation_only = true` to skip data export.

The 2D demo defaults to `HighOrderLowRankOp`; the 3D demo defaults to the blocked
`HighOrderKernelOp`. Edit `operator` to choose `"lowrank"` or `"kernel"`.
Without arguments, the reconstruction scripts read the default datasets
`output/2d/2d-0000-physical.mat` and `output/3d/3d-1111-physical.mat`, respectively.
Pass a dataset path as the sole argument to reconstruct a different export.
The reconstruction helper accepts `iterations` (default 200) and `rank`
(default 30, for LowRank).
Both use CGNR with mean-one density compensation (unit RMS square-root weights)
and zero L2 regularization. The iteration budget is not a convergence guarantee.

The exported schema contains the actual simulation coordinates, support and
receive maps, RF-centred high-order coefficients/times from the same sampling,
and known ADC/RF-phase plus `1/sqrt(support)` normalization. There are no fitted
image scales, axis swaps, flips, or support expansion. NFFT requires centred
Cartesian coordinates on active Fourier axes; transformed grids are rejected by
the demo reconstruction helper.

Setting `mode = "density"` in the simulation script enables a validation
control: T1 and T2 are infinite and each RF event prepares ideal transverse spin
density at its centre, retaining nominal-gradient/B0 phase through the RF tail. The default
`physical` mode keeps the original RF and relaxation physics. Physical 3D
multi-shot data have flip-angle/saturation/relaxation weighting and must not be
interpreted as a pure-density error test. Metrics use HighOrderMRI's NRMSE/SSIM
functions without fitted intensity/phase scaling.

## Demo grids and receive maps

The demos use one sequence-defined grid for simulation, field/coil maps, and
reconstruction inputs:

| Demo | Matrix | Declared FOV | Grid spacing |
|---|---|---|---|
| 2D | 200 x 200 x 1 | 200 x 200 x 4 mm | 1 x 1 x 4 mm |
| 3D | 100 x 100 x 35 | 200 x 200 x 70 mm | 2 mm isotropic |

The single 2D plane remains at z=0; its 2 mm RF slice thickness is distinct from
the sequence's declared FOV thickness. Matching grids does not remove RF,
relaxation, or undersampling effects from a physical reconstruction.

Measured coil maps are linearly interpolated between source voxel centres.
Within the outer half-voxel cells, boundary values are extended to the voxel
faces. Outside the source physical FOV, sensitivities remain zero. Nonzero
receive vectors are normalized across coils; source zeros are not filled.

## Copyright & License Notice

This software is copyrighted by the Regents of the University of Minnesota and the Institute of Biophysics, Chinese Academy of Sciences. It can be freely used for educational and research purposes by non-profit institutions, US government agencies, and Chinese government agencies only.
Other organizations are allowed to use this software only for evaluation purposes, and any further uses will require prior approval. The software may not be sold or redistributed without prior approval.
One may make copies of the software for their use provided that the copies are not sold or distributed, and are used under the same terms and conditions.
As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied.
The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.
