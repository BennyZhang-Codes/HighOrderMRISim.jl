# HighOrderMRISim.jl

[![Julia 1.12](https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white)](https://julialang.org/downloads/)
[![KomaMRI 0.15](https://img.shields.io/badge/KomaMRI-0.15-0B7285)](https://github.com/JuliaHealth/KomaMRI.jl)
[![CPU CI](https://img.shields.io/badge/CPU_CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/BennyZhang-Codes/HighOrderMRISim.jl/actions/workflows/cpu-ci.yml)
[![License terms](https://img.shields.io/badge/License-See_terms-lightgrey)](LICENSE)

HighOrderMRISim.jl is a Julia package for high-order MRI signal simulation
built on [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl).

## Features

- Time-varying magnetic fields through third spatial order.
- CPU simulation and CUDA kernels for single- and multi-GPU execution.
- 2D and 3D spiral demos with reconstruction via
  [HighOrderMRI.jl](https://github.com/BennyZhang-Codes/HighOrderMRI.jl).

## Installation

Use Julia 1.12 and Python 3.12 with `venv` and a shared `libpython`. Clone the
repository and set up local dependencies without changing your default Julia
environment:

```bash
git clone https://github.com/BennyZhang-Codes/HighOrderMRISim.jl.git
cd HighOrderMRISim.jl

python3.12 -m venv local/demo-python
local/demo-python/bin/python -m pip install 'matplotlib==3.11.2'

export PYTHON="$PWD/local/demo-python/bin/python" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export JULIA_DEPOT_PATH="$PWD/local/demo-depot:" JULIA_LOAD_PATH='@:@stdlib'
export OPENBLAS_NUM_THREADS=1
```

To use the simulator on its own, install and launch it from the root project:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=.
```

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

The demos use a [separate Julia environment](demo/Project.toml) with pinned
simulation and reconstruction dependencies. Install it once in the same shell
instead of installing the root project.

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

See [package tests](test/README.md) for validation details.

## License

[LICENSE](LICENSE)
