# Shared CPU/GPU inputs only; loading a fixture never runs a test suite.
using HighOrderMRISim

function compatibility_sequence()
    rf_duration = 1e-4
    gradients = [Grad(0.0, 0.0) for _ in 1:3, _ in 1:2]
    pulses = reshape([RF(1 / (4γ * rf_duration), rf_duration), RF(0.0, 0.0)], 1, :)
    adc = [ADC(0, 0.0), ADC(5, 1e-3)]
    return Sequence(gradients, pulses, adc, [rf_duration, 2e-3])
end

function compatibility_phantom()
    return HO_Phantom(
        x=[0.0, 0.01], ρ=[1.0, 0.7], T1=fill(Inf, 2), T2=fill(Inf, 2),
        csm=ComplexF64[1 im; 0.5-0.2im -0.3+0.1im],
    )
end
