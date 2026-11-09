# Migration from v0.1.0

The v0.2.0 code targets the KomaMRI 0.15 stack and is validated on Julia 1.12.
`HO_Phantom`, `HO_Sequence`, and `BlochHighOrder` remain the high-order
simulation interface. Existing block-oriented field inputs and coil maps
stored in `obj.csm` do not need to move to a new data model.

## Update execution settings

| Existing setting | Current guidance |
| --- | --- |
| `sim_params["gpu"]` and `sim_params["gpu_device"]` | Still accepted. Prefer `array_type=Array` for CPU or `array_type=CuArray, gpus=[0]` for CUDA. Explicit keywords take precedence. |
| `sim_params["Nblocks"]` | Still accepted. Use `"max_block_length"` and `"max_rf_block_length"` for explicit limits; they take precedence over `"Nblocks"`. |
| `w` for the progress window | Retained for call compatibility, but the Blink window is no longer used. Use `verbose` for progress output or `callbacks` for programmatic observation. |
| `"mat"`, `"state"`, and `"raw"` returns | Still available. `"mat"` preserves the trailing singleton dimension. |

For example, an existing call using `sim_params` can explicitly select GPU 0
without changing the sequence or phantom:

```julia
using HighOrderMRISim, CUDA

signal = simulate(obj, hoseq, scanner; sim_params, array_type=CuArray, gpus=[0])
```

## Compare with legacy results

Hold the sequence, phantom, coil maps, field mask, working precision, and
reconstruction settings fixed when comparing simulation versions. Sampling now
uses current KomaMRI timing and includes attached field waveform breakpoints.
Small signal differences from the legacy KomaMRI 0.8 stack are expected; do
not require bitwise agreement.

For a backend comparison within v0.2.0, set `"implementation" => "array"` as
the reference and compare it against `"kernel"` for supported static CUDA
inputs. Keep device count and precision fixed; a different spin partition can
change the floating-point summation order.

## Update demo workflows

Use the pinned environment in [2D and 3D Demos](@ref). Run simulation and
reconstruction in separate Julia processes. The reconstruction scripts expect
the current MAT export, which records the exact simulation grid, receive maps,
and RF-referenced encoding. Regenerate legacy datasets rather than assuming
that their field, phase, and normalization conventions match the new export.
