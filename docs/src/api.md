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
