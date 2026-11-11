# High-Order Simulation

Pass an `HO_Phantom`, an `HO_Sequence`, and a KomaMRI `Scanner` to `simulate`.
The phantom holds one row of receive sensitivities per spin and one column per
coil. The sequence wraps a native KomaMRI `Sequence` and attaches field events
by block. [Getting Started](@ref) builds both inputs from scratch.

If `csm` is omitted, `HO_Phantom` uses one ideal receive coil with sensitivity
`1 + 0im` at every spin: unit amplitude, zero additional phase, and no signal
normalization. Default numeric arrays preserve Float32/Float64 coordinate precision, so both
`HO_Phantom(x=Float32[0])` and `HO_Phantom(x=Float64[0])` work without a CSM.
Explicit CSM values, including zeros and complex phases, are not normalized or
replaced. This receive-coil default does not change the transmit B1 model.

## Field coefficients and order selection

Each column of `hoseq.GR_dfc` corresponds to one sequence block. Row `n + 1`
contains the KomaMRI `Grad` event for real solid-harmonic term ``h_n``. During
free precession, the enabled time-varying terms contribute to the longitudinal
field as

```math
B_{\mathrm{dyn}}(\mathbf r,t) = \sum_n c_n(t) h_n(\mathbf r).
```

Coordinates are in metres and event times are in seconds. Coefficients are
field amplitudes, not time-integrated trajectories. With
``r^2 = x^2 + y^2 + z^2``, the basis in row order is:

| Order | Rows in `GR_dfc` | Basis in row order | Coefficient units |
| --- | --- | --- | --- |
| 0 | 1 | ``1`` | T |
| 1 | 2–4 | ``x``, ``y``, ``z`` | T/m |
| 2 | 5–9 | ``xy``, ``yz``, ``3z^2-r^2``, ``xz``, ``x^2-y^2`` | T/m² |
| 3 | 10–16 | ``3yx^2-y^3``, ``xyz``, ``(5z^2-r^2)y``, ``5z^3-3zr^2``, ``(5z^2-r^2)x``, ``(x^2-y^2)z``, ``x^3-3xy^2`` | T/m³ |

`BlochHighOrder("1111")` selects all four orders, in order from zeroth to third.
`BlochHighOrder("1000")` enables only the zeroth-order field in addition to the
nominal gradients. `BlochHighOrder("0000")` uses nominal gradients without
dynamic field terms. Phantom off-resonance (`obj.Δw`, in rad/s) is controlled
separately by the excitation and precession flags.

!!! important "First-order fields replace nominal gradients"
    When the second digit is `1`, rows 2–4 supply the full first-order field
    during precession. Put the nominal gradients in these rows for blocks or
    axes without a measured replacement. Leaving them zero is not equivalent
    to using nominal gradients. RF excitation uses nominal gradients regardless
    of the mask.

For example, with `hoseq = HO_Sequence(sequence)`, copy the nominal first-order
events before replacing selected entries with measured events:

```julia
hoseq.GR_dfc[2:4, :] .= sequence.GR
```

Alternatively, pass the event matrix directly to `HO_Sequence(sequence, fields)`.
It must have one column per sequence block. The constructor pads missing higher
rows with zero events. Sampling accounts for attached field durations and
waveform breakpoints without changing the input sequence's block durations.

## Execution paths

Set `array_type=Array` for CPU or `array_type=CuArray` for CUDA. GPU numbers in
`gpus` refer to the devices visible to Julia, starting at zero. Explicit CUDA
requests fail if CUDA or the requested devices are unavailable.

| Option in `sim_params` | Default | Effect |
| --- | --- | --- |
| `"implementation"` | `"auto"` | Selects the kernel for supported static CUDA high-order inputs; otherwise uses the array path. `"kernel"` requires kernel support and `"array"` selects the compatibility path. |
| `"coil_combination"` | `"gemm"` | Uses cuBLAS on the CUDA kernel path. `"reduction"`, `"fused_warp"`, and `"fused_tiled"` are alternative reductions. |
| `"precision"` | `"f32"` | Selects `"f32"` or `"f64"`. Keep it fixed when comparing backends. |

Multi-GPU simulation partitions spins, not readouts. Each partition sees the
complete RF and relaxation history. The simulator sums the partial signals in
device-list order, so small floating-point differences from a single GPU are
expected. Callbacks see partition-local data; see [API Reference](@ref) for
the callback contract.

## Reuse a prepared simulation

For repeated runs with the same static inputs, `prepare_simulation` reuses
sampled data, GPU workers, and buffers. The do-block closes the plan even if a
run fails:

```julia
using CUDA

first_signal, second_signal = prepare_simulation(obj, hoseq, scanner;
    sim_params, array_type=CuArray, gpus=[0, 1]) do plan
    sig1 = simulate(plan; verbose=false)
    sig2 = simulate(plan; verbose=false)
    return sig1, sig2
end
```

Each run starts from the initial magnetization and returns independent host
output. Prepare a new plan when model inputs change. For a plan created without
a do-block, call `close(plan)` in `finally`. Measure setup and repeated execution
separately; multi-GPU execution is not necessarily faster for a single run.
