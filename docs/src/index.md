# HighOrderMRISim.jl

HighOrderMRISim extends [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
with Bloch simulation in time-varying magnetic fields through third spatial
order. It produces complex multi-coil signals from a sequence, a field model,
and a spin phantom. Image reconstruction is a separate step provided by
[HighOrderMRI.jl](https://github.com/BennyZhang-Codes/HighOrderMRI.jl).

## One model, several execution paths

| Path | Typical use |
| --- | --- |
| CPU array | Small reference problems and validation without CUDA hardware |
| Single-GPU CUDA | Static high-order simulations with one device |
| Multi-GPU CUDA | Spin partitioning for larger problems or repeated simulations |

All paths use the same field-order selection and RF model. The CUDA paths
change how the calculation is executed, not which field terms are included.

!!! note "Validation scope"
    Numerical agreement across backends and model-matched reconstruction
    checks do not establish independent scanner-level accuracy. See
    [Validation and interpretation](@ref) for what the demo results measure.

## Suggested reading order

1. [Getting Started](@ref): install the package and simulate a small CPU example.
2. [High-Order Simulation](@ref): choose field orders, devices, and execution paths.
3. [2D and 3D Demos](@ref): reproduce the bundled simulation and reconstruction runs.
4. [Migration from v0.1.0](@ref) and [API Reference](@ref): update existing scripts
   and look up the current interface.
