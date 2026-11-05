module SimulationDemoSupport
using CUDA, MAT
import HighOrderMRISim as Sim
import KomaMRICore

function sampled_encoding(hoseq, params, orders)
    sampled = Sim.discretize(hoseq; sampling_params=params)
    ranges, rf = KomaMRICore.get_sim_ranges(sampled.seqd;
        max_block_length=params["max_block_length"], max_rf_block_length=params["max_rf_block_length"])
    ends = [last(ranges[i]) for i in eachindex(ranges) if rf[i] && (i == length(ranges) || !rf[i+1])]
    rf_starts = [first(ranges[i]) for i in eachindex(ranges) if rf[i] && (i == 1 || !rf[i-1])]
    isempty(ends) && error("The demo must contain an RF preparation.")
    native = Sim.simulation_sequence(hoseq)
    starts = Sim.KomaMRIBase.get_block_start_times(native)
    centre_times = [starts[i]+native.RF[1, i].delay+Sim.KomaMRIBase.rf_center(native.RF[1, i])
        for i in eachindex(native.DUR) if Sim.KomaMRIBase.is_RF_on(native.RF[1, i])]
    @assert length(centre_times) == length(ends)
    centres = [argmin(abs.(sampled.seqd.t .- t)) for t in centre_times]
    @assert all(abs(sampled.seqd.t[node]-t) < 1e-9 for (node, t) in zip(centres, centre_times))
    tail_coefficients = zeros(3, length(ends))
    tail_ages = zeros(length(ends))
    nominal_gradients = (sampled.seqd.Gx, sampled.seqd.Gy, sampled.seqd.Gz)
    for (event, (centre, last_node)) in enumerate(zip(centres, ends))
        @assert centre <= last_node
        tail_ages[event] = sampled.seqd.t[last_node]-sampled.seqd.t[centre]
        for interval in centre:last_node-1, axis in 1:3
            tail_coefficients[axis, event] += Sim.γ*sampled.seqd.Δt[interval]*
                (nominal_gradients[axis][interval]+nominal_gradients[axis][interval+1])/2
        end
    end
    # Independent unit-spin RF calculation determines only the known pulse
    # phase, never a fitted receive-data scale or a T1/sin(flip-angle) correction.
    P = eltype(sampled.seqd.Δt)
    unit = Sim.HO_Phantom(x=[0.0], ρ=[1.0], T1=[Inf], T2=[Inf],
        csm=ones(ComplexF64, 1, 1))
    unit = P === Float32 ? Sim.f32(unit) : Sim.f64(unit)
    rf_phases, ideal_rf_magnitudes = ComplexF64[], Float64[]
    for (first_node, last_node) in zip(rf_starts, ends)
        magnetization = Sim.Mag(zeros(Complex{P}, 1), ones(P, 1))
        Sim.run_spin_excitation!(unit, view(sampled, first_node:last_node),
            zeros(Complex{P}, 0, 1), magnetization, Sim.BlochHighOrder("0000"))
        value = only(magnetization.xy)
        abs(value) > 1e-6 || error("The RF phase reference has negligible transverse excitation.")
        push!(rf_phases, value/abs(value))
        push!(ideal_rf_magnitudes, abs(value))
    end
    isrf = falses(length(sampled.seqd.Δt))
    for (part, on) in zip(ranges, rf)
        isrf[first(part):last(part)-1] .= on
    end
    enabled = collect(orders) .== '1'
    fields = hcat((getfield(sampled, Symbol("h", term)) for term in 0:15)...)
    enabled[1] || (fields[:, 1] .= 0)
    if !enabled[2]
        fields[:, 2:4] .= hcat(sampled.seqd.Gx, sampled.seqd.Gy, sampled.seqd.Gz)
    end
    enabled[3] || (fields[:, 5:9] .= 0)
    enabled[4] || (fields[:, 10:16] .= 0)
    adc_nodes = findall(sampled.seqd.ADC)
    first(adc_nodes) > first(ends) || error("Acquisition before preparation is not supported here.")
    @assert !any(isrf[node-1] for node in adc_nodes)
    coefficients = zeros(Float64, 16, length(adc_nodes))
    ages, preparation = zeros(Float64, length(adc_nodes)), zeros(Int, length(adc_nodes))
    accumulated = zeros(16)
    last_reset, rf_index, column = 1, 0, 1
    for interval in eachindex(sampled.seqd.Δt)
        node = interval+1
        if node in ends
            fill!(accumulated, 0)
            rf_index += 1
            accumulated[2:4] .= tail_coefficients[:, rf_index] ./ Sim.γ
            last_reset = centres[rf_index]
        elseif !isrf[interval]
            for term in 1:16
                accumulated[term] += Float64(sampled.seqd.Δt[interval]) *
                    (fields[interval, term]+fields[node, term])/2
            end
        end
        if sampled.seqd.ADC[node]
            # Simulation uses negative precession phase; HighOrderMRI uses
            # positive phase. The ideal instantaneous preparation is at the RF
            # centre; its nominal-gradient tail preserves slice refocusing.
            coefficients[:, column] .= -Sim.γ .* accumulated
            ages[column] = sampled.seqd.t[node]-sampled.seqd.t[last_reset]
            preparation[column] = rf_index
            column += 1
        end
    end
    @assert column == length(adc_nodes)+1
    return (; sampled, ranges, rf, ends, centres, tail_coefficients, tail_ages,
        coefficients, ages, preparation, rf_phases, ideal_rf_magnitudes)
end

function density_callback(plan)
    return function (progress, blocks, data, _)
        node = last(blocks.parts[progress.block])
        event = searchsortedfirst(plan.ends, node)
        if event <= length(plan.ends) && plan.ends[event] == node
            T = eltype(data.obj.ρ)
            kx, ky, kz = T.(plan.tail_coefficients[:, event])
            data.Xt.xy .= data.obj.ρ .* cis.(-T(2π) .* (
                kx .* data.obj.x .+ ky .* data.obj.y .+ kz .* data.obj.z) .-
                data.obj.Δw .* T(plan.tail_ages[event]))
            fill!(data.Xt.z, zero(T))
            data.obj.x isa CUDA.CuArray && CUDA.synchronize()
        end
        return nothing
    end
end

function prepare_run!(obj, hoseq, params; mode="physical")
    # One explicit time-block policy shared by simulation and encoding export.
    pop!(params, "Nblocks", nothing)
    params["max_block_length"] = 32
    params["max_rf_block_length"] = Inf
    mode in ("physical", "density") || error("mode must be physical or density.")
    plan = sampled_encoding(hoseq, params, params["sim_method"].name)
    if mode == "density"
        # Validation control only: replace RF preparation, not production physics.
        fill!(obj.T1, Inf)
        fill!(obj.T2, Inf)
    end
    callbacks = mode == "density" ? (density_callback(plan),) : ()
    return (; plan, mode, callbacks)
end

function export_run(case, signal, obj, hoseq, params, run, rho, resolution, offcenter)
    T = params["precision"] == "f32" ? Float32 : Float64
    dims = size(rho)
    mask = rho .!= 0
    count(mask) == length(obj) || error("Density support and simulated spins differ.")
    T.(vec(rho)[vec(mask)]) == T.(obj.ρ) || error("Density ordering differs from simulation.")
    coordinates = Sim.generate_grid(collect(dims .* resolution), collect(dims))
    for (axis, actual) in enumerate((obj.x, obj.y, obj.z))
        coordinates[:, :, :, axis] .+= offcenter[axis]
        T.(vec(coordinates[:, :, :, axis])[vec(mask)]) == T.(actual) ||
            error("Simulation/export coordinates differ on axis $axis.")
    end
    xyz = T.(coordinates)
    csm = zeros(Complex{T}, dims..., size(obj.csm, 2))
    reshape(csm, :, size(obj.csm, 2))[vec(mask), :] .= Complex{T}.(obj.csm)
    fieldmap = zeros(T, dims)
    vec(fieldmap)[vec(mask)] .= -T.(obj.Δw ./ (2π))
    compensation = vec(Sim.get_adc_phase_compensation(Sim.simulation_sequence(hoseq)))
    rf_phase = run.mode == "density" ? ones(Complex{T}, length(compensation)) :
        Complex{T}.(run.plan.rf_phases[run.plan.preparation])
    data = signal[:, :, 1] ./ reshape(Complex{T}.(compensation) .* rf_phase, :, 1) ./ T(sqrt(count(mask)))
    output = normpath(joinpath(@__DIR__, "..", "output", case))
    mkpath(output)
    destination = joinpath(output, "$case-$(params["sim_method"].name)-$(run.mode).mat")
    MAT.matwrite(destination, Dict("schema_version" => 1, "data" => data,
        "signal" => signal, "coefficients" => T.(run.plan.coefficients),
        "times" => T.(run.plan.ages), "rf_preparation_index" => run.plan.preparation,
        "rf_phase_correction" => rf_phase, "rho" => T.(rho), "mask" => UInt8.(mask),
        "csm" => csm, "fieldmap_hz" => fieldmap, "xyz" => xyz,
        "dimensions" => collect(dims), "resolution_m" => collect(resolution),
        "mode" => run.mode, "orders" => params["sim_method"].name,
        "normalization_divisor" => T(sqrt(count(mask)))); compress=true)
    println("DEMO_DATA_SAVED: ", destination)
    return destination
end

end # module
