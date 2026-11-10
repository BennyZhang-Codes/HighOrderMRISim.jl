# HighOrderMRISim.jl

[![Julia 1.12](https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white)](https://julialang.org/downloads/)
[![KomaMRI 0.15](https://img.shields.io/badge/KomaMRI-0.15-0B7285)](https://github.com/JuliaHealth/KomaMRI.jl)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/dev/)
[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/stable/)

HighOrderMRISim.jl is a Julia package for MRI signal simulation with time-varying,
higher-order magnetic fields. Built on
[KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl), it extends Bloch simulation
with a high-order field model originally developed and used in
[DataStitching](https://github.com/BennyZhang-Codes/DataStitching).
The package supports CPU and CUDA-accelerated single- and multi-GPU
execution, with 2D and 3D spiral demos linking simulation to reconstruction through
[HighOrderMRI.jl](https://github.com/BennyZhang-Codes/HighOrderMRI.jl).

## Features

- Time-varying fields through third-order spherical harmonics, with
  selectable field orders during free precession.
- KomaMRI sequence integration with measured or user-defined field waveforms.
- Multi-coil signal reception with coil-sensitivity maps, T1/T2 relaxation,
  and static B0 off-resonance.
- CPU and CUDA-accelerated single- and multi-GPU simulation in Float32
  and Float64.
- CUDA-accelerated reconstruction through
  [HighOrderMRI.jl](https://github.com/BennyZhang-Codes/HighOrderMRI.jl):
  `HighOrderKernelOp` for explicit encoding and
  `HighOrderLowRankOp` for low-rank/NFFT-based encoding.
- End-to-end 2D and 3D spiral demos with matched simulation/reconstruction
  grids and NRMSE/SSIM evaluation.

## How to Cite

If you use this package, please cite the original
[DataStitching](https://github.com/BennyZhang-Codes/DataStitching) paper and the
high-order MRI workshop abstract below.

### Plain Text

Zhang J, Zhang Z, Zuo Z, Xue R, Zhuo Y, Cushing C, Bratch A, Auerbach E,
Grant A, An J, Ugurbil K, Wu X. Data stitching for dynamic field monitoring
with NMR probes. *Magnetic Resonance in Medicine*. 2026;95(4):1944–1958.
[doi:10.1002/mrm.70164](https://doi.org/10.1002/mrm.70164).

Related workshop abstract:

Zhang J, Jaffray A, Qu S, Cushing C, Xue R, Ugurbil K, Rauscher A, Zhang Z,
Wu X. High Impact Project: A comprehensive open-source framework for high-order
MRI. In: *Proceedings of the 2026 ISMRM Workshop on Data Sampling and Image
Reconstruction*, Sedona, AZ, United States of America. Program #302-01-055.

Related work:

Zhang J, Zhen T, Li S, Zuo Z, Cushing C, An J, Xue R, Zhuo Y, Ugurbil K,
Wu X, Zhang Z. Generalized data stitching for dynamic field monitoring with NMR
probes: improving acquisition efficiency and accuracy. In: *Proceedings of the
2026 ISMRM-ISMRT Annual Meeting and Exhibition*, Cape Town, South Africa.
Program #605-02-004.

<details>
<summary>BibTeX</summary>

```bibtex
@article{Zhang2026DataStitching,
  author = {Zhang, Jinyuan and Zhang, Zihao and Zuo, Zhentao and Xue, Rong and Zhuo, Yan and Cushing, Cameron and Bratch, Alexander and Auerbach, Edward and Grant, Andrea and An, Jing and Ugurbil, Kamil and Wu, Xiaoping},
  title = {Data stitching for dynamic field monitoring with {NMR} probes},
  journal = {Magnetic Resonance in Medicine},
  year = {2026},
  month = apr,
  volume = {95},
  number = {4},
  pages = {1944--1958},
  doi = {10.1002/mrm.70164},
  url = {https://doi.org/10.1002/mrm.70164}
}

@inproceedings{Zhang2026HighOrderMRI,
  author = {Zhang, Jinyuan and Jaffray, Alexander and Qu, Shuxian and Cushing, Cameron and Xue, Rong and Ugurbil, Kamil and Rauscher, Alexander and Zhang, Zihao and Wu, Xiaoping},
  title = {High Impact Project: A comprehensive open-source framework for high-order {MRI}},
  booktitle = {2026 {ISMRM} Workshop on Data Sampling and Image Reconstruction},
  year = {2026},
  month = jan,
  address = {Sedona, AZ, United States of America},
  organization = {ISMRM},
  note = {Program Number: 302-01-055},
  url = {https://echo.ismrm.org/abstracts/view/234b9199-c446-4181-ad46-684761459f67}
}

@inproceedings{Zhang2026GeneralizedDataStitching,
  author = {Zhang, Jinyuan and Zhen, Tianxiang and Li, Sisi and Zuo, Zhentao and Cushing, Cameron and An, Jing and Xue, Rong and Zhuo, Yan and Ugurbil, Kamil and Wu, Xiaoping and Zhang, Zihao},
  title = {Generalized data stitching for dynamic field monitoring with {NMR} probes: improving acquisition efficiency and accuracy},
  booktitle = {2026 {ISMRM-ISMRT} Annual Meeting and Exhibition},
  year = {2026},
  month = may,
  address = {Cape Town, South Africa},
  organization = {ISMRM},
  note = {Program Number: 605-02-004},
  url = {https://echo.ismrm.org/abstracts/view/c6ce7146-215e-47e3-beed-12bd0bb2233b}
}
```

</details>

BibTeX file: [CITATION.bib](CITATION.bib).

## Installation

Use Julia 1.12. Install the latest code from the default branch (`main`)
in your Julia project:

```julia
using Pkg
Pkg.add(url="https://github.com/BennyZhang-Codes/HighOrderMRISim.jl.git")
using HighOrderMRISim
```

For GPU use, also add CUDA to the same project with `Pkg.add("CUDA")`.
See [Getting Started](docs/src/getting-started.md#install-from-a-checkout)
for detailed environment and plotting setup.

## Usage

### Simulate with your own inputs

Given an `HO_Phantom` (`obj`) and an `HO_Sequence` (`hoseq`):

```julia
using HighOrderMRISim, CUDA

sim_params = Dict{String,Any}("sim_method" => BlochHighOrder("1111"),
    "precision" => "f32", "return_type" => "mat")
signal = simulate(obj, hoseq, Scanner(); sim_params, array_type=CuArray, gpus=[0])
```

Use the zero-based device numbers listed by `CUDA.devices()`. Set
`gpus=[0, 1]` for two GPUs, or use `array_type=Array` and omit `gpus` for CPU
simulation.

### Run the demos

The full demos require a CUDA-capable NVIDIA GPU and use a
[separate Julia environment](demo/Project.toml) with pinned dependencies.
See the [demo guide](docs/src/guide/demos.md) for setup and validation. First,
clone the repository:

```bash
git clone https://github.com/BennyZhang-Codes/HighOrderMRISim.jl.git
cd HighOrderMRISim.jl
```

Complete the [environment setup](docs/src/getting-started.md#install-from-a-checkout),
then install the demo dependencies:

```bash
julia --startup-file=no --project=demo -e 'using Pkg; Pkg.instantiate()'
```

Run either pair below. Both stages default to `gpus = [0]`; edit `gpus` in the
corresponding script to select a different GPU. The 3D simulation settings are
in [3_run_sim.jl](demo3D/3_run_sim.jl).

```bash
# 2D spiral
julia --startup-file=no --project=demo demo2D/Sim2D.jl
julia --startup-file=no --project=demo demo2D/Recon2D.jl

# 3D spiral
julia --startup-file=no --project=demo demo3D/Sim3D.jl
julia --startup-file=no --project=demo demo3D/4_recon.jl
```

Results are saved under `output/`. The 2D demo uses LowRank reconstruction; the
3D demo uses kernel reconstruction.

See [local documentation builds](docs/README.md) and
[package tests](test/README.md) for validation details.

## License

[LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) ·
[Data sources and pending confirmations](DATA_SOURCES.md)
