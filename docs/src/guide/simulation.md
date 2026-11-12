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
For the bundled real receive map, `load_csm(:real_32cha, nCoil, ...)` selects
the first 1 to 32 channels without renormalizing the selected subset. Phantom
source voxel sizes come from `BrainPhantom` metadata in millimeters; simulation
coordinates and target voxel sizes remain in meters.
Source label voxels extend to their outer half-voxel faces; queries beyond that
physical FOV are zero. Nearest-neighbor label interpolation is unchanged inside
the source grid.

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

`"implementation"` and `"coil_combination"` are HighOrderMRISim options, not
native KomaMRI execution selectors. `"precision"` is inherited from KomaMRI.
The high-order kernels extend KomaMRI's simulation workflow; they do not enable
high-order fields in a native `Sequence` by themselves.

Keep `"gemm"` unless a workload-specific measurement supports another choice.
It was fastest in the tested 32-coil RTX 3090 workloads, but no strategy has a
universal speed or memory advantage. Fused modes can reduce ADC-state storage,
especially when RF has no ADC samples, at a throughput cost in those tests.
Compare complete warmed simulations at the same precision, spin/coil counts,
and block settings; measure peak memory separately. Alternative reductions do
not add physical effects or improve nominal accuracy.

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

## Trajectories and field events

`get_kspace(hoseq)` returns four matrices: nominal encoding, nominal encoding at
ADC times, attached-field encoding, and attached-field encoding at ADC times.
Nominal encoding has three columns in cycles/m. Attached-field encoding has
16 columns: order ``n`` has units cycles/mⁿ, including dimensionless cycles at
order zero. Multiply by ``2π`` only when radians are needed.

`grad2traj(field, dt)` integrates with the trapezoidal rule and retains the
sample count. `traj2grad(trajectory, dt)` computes consecutive differences and
returns one fewer sample along the selected dimension. Neither helper applies
``γ`` or a unit conversion; differentiating coefficients in cycles/mⁿ requires
division by ``γ`` to obtain T/mⁿ. The two numerical operations are not exact
inverses for arbitrary waveforms.

Map interval values to sequence events explicitly. Do not differentiate across
RF resets or assume that measured trajectory samples and field intervals have
the same time origin. `InterpTrajTime` uses seconds, subtracts `delTime` from
input knot times, and extrapolates out-of-range queries to zero. It preserves
the trajectory precision even when query times use another real precision.
The bundled demos retain their validated dataset-specific timing conventions.

## Input checks and model limits

Every spin-property vector must have the same length, and the CSM must have
one row per spin. Simulation rejects inconsistent storage before selecting a
device. Empty phantoms can be displayed and iterated, but cannot be simulated.
Nonempty phantoms with different displacement functions cannot be concatenated:
one global set of functions cannot preserve both motion models.

RF excitation uses nominal gradients; high-order fields apply during free
precession. Sampling and simulation require a single transmit RF channel;
additional native RF rows are rejected rather than silently ignored. Raw sequence
plots can still display multiple RF rows. Motion uses the array path, not the static CUDA kernel. `T2s` and
diffusion-related phantom fields are retained for compatibility, but the
implemented integrator evolves T1/T2 relaxation and does not add an independent
T2s or diffusion model. Stored fields do not imply additional simulated physics.
