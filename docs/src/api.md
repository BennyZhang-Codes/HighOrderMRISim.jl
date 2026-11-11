# API Reference

```@meta
CurrentModule = HighOrderMRISim
```

## Inputs

| Interface | Purpose |
| --- | --- |
| `HO_Phantom(; x, ...)` | Spin coordinates, density, relaxation, off-resonance, and receive sensitivities. Optional `csm` has shape `(nspin, ncoil)`; the default is one unit-sensitivity receive coil. |
| `HO_Sequence(sequence)` | Wrap a KomaMRI sequence with zero-valued dynamic field events. |
| `HO_Sequence(sequence, fields)` | Wrap a sequence and attach block-aligned `Grad` events, with one column per sequence block. |
| `BlochHighOrder(mask)` | Select the zeroth through third field orders using a four-digit string such as `"1111"`. |
| `BlochHighOrder(mask, Δw_excitation, Δw_precession)` | Also select whether phantom off-resonance is used during excitation and precession. |
| `Scanner()` | Construct the KomaMRI scanner input. |

See [Field coefficients and order selection](@ref) for units, row order, and
the rule for replacing nominal first-order gradients.

```@docs
HO_Phantom
HO_Sequence
BlochHighOrder
```

## Simulation

```@docs
simulate(::HO_Phantom, ::HO_Sequence, ::Scanner)
```

## Prepared execution

```@docs
prepare_simulation
```

Use `simulate(plan; callbacks=(), verbose=true)` to run a prepared simulation.
Its default return mode comes from the preparation parameters. Override it
with `return_type="mat"`, `"state"`, or `"raw"`. Use the do-block form of
`prepare_simulation` or close the plan explicitly with `close(plan)`.

## Trajectory and grid helpers

```@docs
get_kspace(::HO_Sequence)
grad2traj
traj2grad
InterpTrajTime
generate_grid
```

## Plotting

See [Plotting and Field Inspection](@ref) for executable examples and unit
conventions. These helpers return figures; they do not change simulation physics.

```@docs
plt_seq
plt_seqd
plt_kspace
plt_bfield
plt_kspha
plt_traj
plt_phantom
```

## Legacy real-image metrics

These helpers take the reference first and retain their existing fitted-scaling
default. The reconstruction demos use HighOrderMRI's norm-normalized diagnostics
instead; see [Validation and interpretation](@ref).

```@docs
HO_img_scale
HO_NRMSE
HO_MSE
HO_RMSE
HO_SSIM
normalization
standardization
```
