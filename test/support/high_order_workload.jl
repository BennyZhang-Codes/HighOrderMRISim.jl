# Deterministic inputs shared by GPU regression, validation, and performance tools.
using HighOrderMRISim

function high_order_workload(; nspins=32768, nsteps=32, ncoils=32, rf=false)
    u = nspins == 1 ? [0.0] : collect(range(-1.0, 1.0; length=nspins))
    obj = HO_Phantom(
        x=0.08 .* u, y=0.06 .* sin.(3 .* u), z=0.04 .* cos.(5 .* u),
        ρ=0.8 .+ 0.2 .* cos.(u), T1=0.9 .+ 0.1 .* cos.(u),
        T2=0.07 .+ 0.01 .* cos.(u), Δw=80 .* u,
        csm=[cis(2π * coil * ui / ncoils) / (1 + coil / ncoils)
             for ui in u, coil in 1:ncoils],
    )
    t = collect(range(0.0, rf ? 2e-4 : 2e-3; length=nsteps+1))
    q = collect(range(0.0, 2π; length=length(t)))
    adc = trues(length(t))
    adc[1] = false
    seqd = DiscreteSequence(
        2e-4 .* sin.(q), 2e-4 .* cos.(q), fill(1e-4, length(t)),
        rf ? 8e-6 .* cis.(q ./ 4) : zeros(ComplexF64, length(t)),
        rf ? 100 .* sin.(q) : zeros(length(t)),
        rf ? 0.2 .* q : zeros(length(t)),
        adc, fill(rf, nsteps), t, diff(t),
    )
    fields = ntuple(16) do term
        scale = term == 1 ? 1e-7 : term <= 4 ? 2e-4 : term <= 9 ? 1e-3 : 1e-2
        scale .* sin.(q .+ term / 4)
    end
    hoseqd = HO_DiscreteSequence(seqd, fields...)
    state = Mag(complex.(0.3 .* obj.ρ, 0.6 .* obj.ρ), 0.5 .* obj.ρ)
    signal = zeros(ComplexF64, nsteps, ncoils)
    return (; obj, hoseqd, state, signal)
end

function convert_workload(workload, precision, backend=nothing)
    convert_precision = precision == "f32" ? f32 : f64
    values = map(convert_precision, Tuple(workload))
    if !isnothing(backend)
        values = map(x -> gpu(x, backend), values)
    end
    return NamedTuple{keys(workload)}(values)
end

function reset_workload!(workload)
    workload.state.xy .= complex.(0.3 .* workload.obj.ρ, 0.6 .* workload.obj.ρ)
    workload.state.z .= 0.5 .* workload.obj.ρ
    return nothing
end

function run_reference_block!(workload, rf)
    operation = rf ? HighOrderMRISim.run_spin_excitation! : HighOrderMRISim.run_spin_precession!
    operation(workload.obj, workload.hoseqd, workload.signal,
              workload.state, BlochHighOrder("1111"))
    return nothing
end
