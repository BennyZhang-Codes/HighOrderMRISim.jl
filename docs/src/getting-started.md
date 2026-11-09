# Getting Started

Use Julia 1.12 and Python 3.12 with `venv` and a shared `libpython` for PyPlot.
The example below runs on CPU and does not need an external dataset. For the
full simulation-to-reconstruction workflow, use the separate environment in
[2D and 3D Demos](@ref).

## Install from a checkout

Run these commands from the repository root. The local depot and Python
environment keep the setup separate from your default Julia environment.

```bash
python3.12 -m venv local/demo-python
local/demo-python/bin/python -m pip install 'matplotlib==3.11.2'

export PYTHON="$PWD/local/demo-python/bin/python" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export JULIA_DEPOT_PATH="$PWD/local/demo-depot:" JULIA_LOAD_PATH='@:@stdlib'

julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=.
```

## Simulate a small CPU example

The sequence contains a nominal 90° RF pulse followed by an eight-sample
readout. Attach a second-order ``xy`` field with a coefficient of 0.05 T/m²
during the readout. The phantom has two spins and one receive coil with unit
sensitivity. The two spins see opposite fields of ±1 μT.

```@example cpu
using HighOrderMRISim

sequence = Sequence()
τ = 1e-4
@addblock sequence += RF(1 / (4γ * τ), τ)
@addblock sequence += ADC(8, 1e-3)

hoseq = HO_Sequence(sequence)
hoseq.GR_dfc[5, 2] = Grad(0.05, 1e-3) # xy coefficient in readout block 2

obj = HO_Phantom(
    x = [-5e-3, 5e-3],
    y = [4e-3, 4e-3],
    ρ = [1.0, 0.7],
    T1 = [Inf, Inf],
    T2 = [Inf, Inf],
    csm = ones(ComplexF64, 2, 1),
)

scanner = Scanner()
sim_params = Dict{String,Any}(
    "sim_method" => BlochHighOrder("0010"),
    "precision" => "f64",
    "return_type" => "mat",
)
signal = simulate(obj, hoseq, scanner; sim_params, array_type=Array, verbose=false)
@assert size(signal) == (8, 1, 1) && all(isfinite, signal) # hide
@assert !isapprox(signal[1], signal[end]) # hide
size(signal)
```

The axes of `signal` are ADC samples, receive coils, and a trailing singleton
dimension retained for compatibility. Use `signal[:, :, 1]` when a
sample-by-coil matrix is needed.

## Run the same model on a GPU

With a functional CUDA installation, select a device by its zero-based number
in `CUDA.devices()`. Keep the same inputs and precision for a backend comparison.

```julia
using CUDA

signal_gpu = simulate(obj, hoseq, scanner; sim_params, array_type=CuArray, gpus=[0])
```

Use `gpus=[0, 1]` to distribute simulation across two visible devices. See
[High-Order Simulation](@ref) for backend selection and prepared execution.
