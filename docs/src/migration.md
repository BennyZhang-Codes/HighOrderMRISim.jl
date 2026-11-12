# Migration from v0.1.0

The v0.2 series targets the KomaMRI 0.15 stack and is validated on Julia 1.12.
`HO_Phantom`, `HO_Sequence`, and `BlochHighOrder` remain the high-order
simulation interface. Existing block-oriented field inputs and coil maps
stored in `obj.csm` do not need to move to a new data model.

## Keep environments separate

Do not resolve a legacy lockfile against the new dependency stack in place.
Keep the v0.1.0 project and results for reference, and instantiate the current
package or [pinned demo environment](@ref "Install the pinned demo environment")
in a separate project. The demo environment includes reconstruction dependencies;
the core package does not require HighOrderMRI.

## Update execution settings

| Existing setting | Current guidance |
| --- | --- |
| `sim_params["gpu"]` and `sim_params["gpu_device"]` | Still accepted. Prefer `array_type=Array` for CPU or `array_type=CuArray, gpus=[0]` for CUDA. Explicit keywords take precedence. |
| `sim_params["Nblocks"]` | Still accepted. Use `"max_block_length"` and `"max_rf_block_length"` for explicit limits; they take precedence over `"Nblocks"`. |
| `w` for the progress window | Retained for call compatibility, but the Blink window is no longer used. Use `verbose` for progress output or `callbacks` for programmatic observation. |
| `"mat"`, `"state"`, and `"raw"` returns | Still available. `"mat"` preserves the trailing singleton dimension. |
| `"precision"` | Set `"f32"` or `"f64"` explicitly for a controlled comparison. HighOrderMRISim does not support KomaMRI's CPU BigFloat option. |

A dictionary returned by `default_sim_params()` already contains maximum block
lengths. Set those limits directly rather than adding `"Nblocks"` on top.

`"implementation"` and `"coil_combination"` are HighOrderMRISim extensions,
not native KomaMRI switches. Start with `"auto"` and `"gemm"`. The former
selects the static CUDA kernel when supported and otherwise uses the array path.
An explicit `"kernel"` request rejects unsupported inputs rather than ignoring
motion. RF still uses nominal gradients; high-order fields affect precession.
See [Execution paths](@ref) for alternatives and their trade-offs.

This small CPU example exercises the current interface without a GPU:

```@example migration
using HighOrderMRISim

seq = Sequence()
τ = 1e-4
@addblock seq += RF(1 / (4γ * τ), τ)
@addblock seq += ADC(8, 1e-3)
hoseq = HO_Sequence(seq)
hoseq.GR_dfc[2:4, :] .= seq.GR
hoseq.GR_dfc[2, 2] = Grad(1e-3, 1e-3)

obj = HO_Phantom(x=Float32[-0.01, 0, 0.01],
    csm=ones(ComplexF32, 3, 1))
scanner = Scanner()
sim_params = Dict{String,Any}(
    "sim_method" => BlochHighOrder("1111"),
    "precision" => "f32", "return_type" => "mat",
    "implementation" => "auto", "coil_combination" => "gemm")
signal = simulate(obj, hoseq, scanner; sim_params,
    array_type=Array, gpus=[], verbose=false)
@assert size(signal) == (8, 1, 1)
@assert eltype(signal) == ComplexF32
@assert all(isfinite, signal)
nothing # hide
```

With the same inputs, select CUDA-visible GPU 0 explicitly:

```julia
using CUDA

signal = simulate(obj, hoseq, scanner; sim_params, array_type=CuArray, gpus=[0])
```

Device numbers refer to the GPUs visible to that Julia process, not necessarily
physical indices. Use `gpus=[0, 1]` for two visible GPUs. Multi-GPU execution
partitions spins through the complete sequence, not shots; each shard preserves
its RF and relaxation history. For repeated runs, use an explicitly owned
`prepare_simulation` plan and close it when finished. There is no implicit cache.

## Preserve field and receive-map conventions

- `GR_dfc` rows 1, 2–4, 5–9, and 10–16 hold zeroth-, first-, second-, and
  third-order **field amplitudes**, in T/mⁿ. They are not integrated trajectories.
- `"1111"` enables all four orders. Enabled first-order fields replace the
  nominal gradients during precession. Copy `seq.GR` into rows 2–4 before
  replacing selected blocks with measured fields. Use `"0000"` for nominal
  gradients only; it does not disable the phantom's off-resonance flags.
- Replace a `Grad` event with a newly constructed event when changing between
  scalar and waveform amplitudes. Current KomaMRI stores amplitude type in the
  event type; assigning a vector to a scalar event's `.A` field can fail. Preserve
  the intended duration, ramps, and delay when replacing it.
- Coordinates are in metres, T1/T2 in seconds, and `Δw` in rad/s. A CSM has
  shape `(number_of_spins, number_of_coils)`. Explicit complex sensitivities,
  including zeros, are used as supplied, without normalization.
- Omitting `csm` now means one ideal unit receive coil, not a zero-signal map.
  Pass the desired map explicitly when reproducing a legacy experiment.

See [Field coefficients and order selection](@ref) for the complete basis and
[Trajectories and field events](@ref) for integration, units, and RF timing.

Input validation now rejects inconsistent spin-property lengths, CSM rows,
field/block counts, and multiple transmit RF rows before simulation. Correct the
inputs rather than bypassing those checks. Stored `T2s` or diffusion fields do
not imply additional simulated physics; see [Input checks and model limits](@ref).

## Compare with legacy results

Hold the sequence, phantom, coil maps, field mask, working precision, and
reconstruction settings fixed when comparing simulation versions. Sampling now
uses current KomaMRI timing and includes attached field waveform breakpoints.
Small signal differences from the legacy KomaMRI 0.8 stack are expected; do
not require bitwise agreement.

For a backend comparison within the current package, set `"implementation" => "array"` as
the reference and compare it against `"kernel"` for supported static CUDA
inputs. Keep device count and precision fixed; a different spin partition can
change the floating-point summation order.

Check the exported coordinates, CSM, encoding coefficients, and sample times
before attributing a reconstruction difference to the simulator. Do not fit an
intensity scale or shift the image grid to hide a mismatch. Pure spin density
is a valid reference for the explicit `"density"` control, not generally for
multi-shot physical data with RF/relaxation contrast.

## Update demo workflows

Use the pinned environment in [2D and 3D Demos](@ref). Run simulation and
reconstruction in separate Julia processes. The reconstruction scripts expect
the current MAT export, which records the exact simulation grid, receive maps,
and RF-referenced encoding. Regenerate legacy datasets rather than assuming
that their field, phase, and normalization conventions match the new export.

The current 2D and 3D scripts both use Float32, `"1111"`, and a default
`HighOrderKernelOp` reconstruction; LowRank remains optional. These defaults
differ from the previous 2D Float64/`"0000"`/LowRank configuration, so its error
and timing figures are not an unchanged-configuration comparison. Current 2D
exports use `2d-1111-physical.mat` or `2d-1111-density.mat`.

The pinned demo project loads the released simulator, not uncommitted source.
To test changes to this checkout, use
[`demo/instantiate_checkout.jl`](@ref "Validate the current checkout instead")
with a new environment directory; keep the release-pinned demo lockfile intact.
