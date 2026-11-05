# The plan owns its workers and device allocations. No module-global GPU cache.
struct SimulationWorker
    device::Int
    jobs::Channel{Any}
    results::Channel{Any}
    task::Task
end

mutable struct PreparedSimulation{S}
    setup::S
    workers::Vector{SimulationWorker}
    ranges::Vector{UnitRange{Int}}
    lock::ReentrantLock
    closed::Bool
end

function SimulationWorker(initialize, device)
    jobs, results, ready = Channel{Any}(1), Channel{Any}(1), Channel{Any}(1)
    task = Threads.@spawn begin
        data = try
            CUDA.device!(device)
            Base.invokelatest(initialize)
        catch err
            put!(ready, CapturedException(err, catch_backtrace()))
            return nothing
        end
        put!(ready, nothing)
        try
            for job in jobs
                response = try
                    (:ok, Base.invokelatest(job, data))
                catch err
                    failure = CapturedException(err, catch_backtrace())
                    # Join pending work before reporting a failed callback/job.
                    try
                        CUDA.synchronize()
                    catch
                        # Preserve the original failure if the CUDA context failed too.
                    end
                    (:error, failure)
                end
                put!(results, response)
                job = nothing
                response = nothing
            end
        finally
            release_simulation_arrays!(data)
        end
        return nothing
    end
    failure = take!(ready)
    if !isnothing(failure)
        wait(task)
        throw(failure)
    end
    return SimulationWorker(device, jobs, results, task)
end

function collect_simulation_arrays!(arrays, x)
    if x isa CUDA.CuArray
        push!(arrays, x)
    elseif x isa SubArray
        collect_simulation_arrays!(arrays, parent(x))
    elseif x isa Union{HO_Phantom,HO_DiscreteSequence,DiscreteSequence,Mag,HighOrderGPUWorkspace}
        for name in fieldnames(typeof(x))
            collect_simulation_arrays!(arrays, getfield(x, name))
        end
    elseif x isa Union{Tuple,NamedTuple} || (x isa AbstractVector && !(eltype(x) <: Number))
        for value in x
            collect_simulation_arrays!(arrays, value)
        end
    end
    return nothing
end

function release_simulation_arrays!(data)
    CUDA.synchronize()
    arrays = Base.IdSet{Any}()
    collect_simulation_arrays!(arrays, data)
    for array in arrays
        CUDA.unsafe_free!(array)
    end
    return nothing
end

function prepare_simulation_shard(host_obj, host_hoseqd, setup, device)
    CUDA.device!(device)
    obj, hoseqd = gpu(host_obj, setup.backend), gpu(host_hoseqd, setup.backend)
    T = eltype(obj.ρ)
    Xt = Mag(CUDA.zeros(Complex{T}, length(obj)), similar(obj.ρ))
    sig = CUDA.zeros(Complex{T}, sim_output_dim(obj, setup.seq, setup.sys, setup.params["sim_method"])..., 1)
    workspace = HighOrderGPUWorkspace(obj, setup.plan.max_adc, setup.backend;
        coil_combination=get(setup.params, "coil_combination", "gemm"), max_rf_adc=setup.plan.max_rf_adc)
    blocks = [view(hoseqd, p) for p in setup.plan.parts]
    offsets = cumsum([1; setup.plan.adc_counts])
    signals = [view(sig, offsets[i]:(offsets[i + 1] - 1), :, 1) for i in eachindex(blocks)]
    params = copy(setup.params)
    params["gpu_device"], params["gpu"], params["Nthreads"] = device, true, 1
    params["implementation"] = "kernel"
    params["coil_combination"] = get(params, "coil_combination", "gemm")
    params["max_adc_per_block"] = setup.plan.max_adc
    params["max_rf_adc_per_block"] = setup.plan.max_rf_adc
    params["adc_state_buffer_samples"] = size(workspace.adc_states, 2)
    params["prepared"] = true
    return (; obj, hoseqd, Xt, sig, workspace, blocks, signals, host_obj, host_hoseqd,
            host_signal=zeros(Complex{T}, size(sig)), params)
end

# Restore values into existing storage, including any callback edits from a prior
# run. The host snapshots are private; changing the original inputs is harmless.
function restore_simulation_inputs!(device, host)
    for name in fieldnames(typeof(host))
        destination, source = getfield(device, name), getfield(host, name)
        if destination isa CUDA.CuArray
            copyto!(destination, source)
        elseif destination isa DiscreteSequence
            restore_simulation_inputs!(destination, source)
        end
    end
    return nothing
end

"""
    prepare_simulation(obj, hoseq, sys; sim_params=Dict(),
                       array_type=CUDA.CuArray, gpus=nothing)

Prepare an owned, reusable static CUDA high-order simulation. Sampling, device
inputs, magnetization, signal/work buffers, block views, and one persistent task
per device are reused by `simulate(plan; callbacks=(), return_type="mat")`.
The same precision, Bloch model, coil algorithm, and spin partition are used as
in one-shot `simulate`. CPU/array/moving-coordinate execution retains the
one-shot API and is not accepted here.

Inputs are snapshotted. Each run restores model/sequence values and resets
magnetization and signal buffers, preserving independent-run semantics even
after a callback failure. Callbacks may edit array values but must not rebind
arrays or change their shapes. Returned host arrays are independent snapshots.
Multiple calls on one plan serialize; recursive use from its callbacks errors.

Call `close(plan)` in `finally` to release owned device arrays and join workers.
The `prepare_simulation(f, obj, hoseq, sys; ...)` do-block form closes
automatically. Closed plans cannot be run again. No hidden global cache is used.
"""
function prepare_simulation(
    obj::HO_Phantom, hoseq::HO_Sequence, sys::Scanner;
    sim_params=Dict{String,Any}(), array_type=CUDA.CuArray, gpus=nothing,
)
    setup = setup_simulation(obj, hoseq, sys; sim_params, array_type, gpus)
    setup.backend isa CUDA.CUDABackend && setup.params["sim_method"] isa BlochHighOrder &&
        has_static_coordinates(obj) && get(setup.params, "implementation", "auto") != "array" ||
        throw(ArgumentError("Prepared simulation requires the static CUDA high-order kernel."))
    devices = something(setup.devices, [something(setup.params["gpu_device"], 0)])
    ranges = simulation_spin_ranges(length(obj), length(devices))
    precision = setup.params["precision"] == "f32" ? f32 : f64
    host_obj = precision(deepcopy(cpu(obj)))
    host_hoseqd = precision(setup.hoseqd)
    # Freeze raw output metadata as well as the sampled device waveforms.
    setup = merge(setup, (; obj=(; name=obj.name), seq=deepcopy(setup.seq), sys=deepcopy(sys)))
    workers = SimulationWorker[]
    caller_device = CUDA.device()
    try
        for (i, id) in enumerate(devices)
            push!(workers, SimulationWorker(id) do
                prepare_simulation_shard(host_obj[ranges[i]], host_hoseqd, setup, id)
            end)
        end
        return PreparedSimulation(setup, workers, ranges, ReentrantLock(), false)
    catch
        for worker in workers
            close(worker.jobs)
        end
        for worker in workers
            wait(worker.task)
        end
        rethrow()
    finally
        CUDA.device!(caller_device)
    end
end

function prepare_simulation(f::Function, obj::HO_Phantom, hoseq::HO_Sequence, sys::Scanner; kwargs...)
    plan = prepare_simulation(obj, hoseq, sys; kwargs...)
    try
        return f(plan)
    finally
        close(plan)
    end
end

function check_simulation_recursion(plan)
    any(w -> w.task === current_task(), plan.workers) &&
        throw(ArgumentError("A prepared simulation cannot be used recursively from its callback."))
    return nothing
end

function Base.close(plan::PreparedSimulation)
    check_simulation_recursion(plan)
    lock(plan.lock) do
        plan.closed && return nothing
        plan.closed = true
        for worker in plan.workers
            close(worker.jobs)
        end
        failures = Any[]
        for worker in plan.workers
            try
                wait(worker.task)
            catch err
                push!(failures, err)
            end
        end
        isempty(failures) || throw(CompositeException(failures))
        return nothing
    end
    return nothing
end

function simulate(
    plan::PreparedSimulation; callbacks=(), verbose=true,
    return_type=plan.setup.params["return_type"],
)
    return_type in ("mat", "state", "raw") || throw(ArgumentError("return_type must be mat, state, or raw."))
    check_simulation_recursion(plan)
    return lock(plan.lock) do
        plan.closed && throw(ArgumentError("The prepared simulation is closed."))
        callback_lock = ReentrantLock()
        mode, precision = CUDA.math_mode(), CUDA.math_precision()
        started = time_ns()
        for (i, worker) in enumerate(plan.workers)
            put!(worker.jobs, data -> run_prepared_shard!(data, plan, i, callbacks,
                                                       callback_lock, mode, precision, return_type))
        end
        results, failures = Any[], Any[]
        for worker in plan.workers
            status, payload = take!(worker.results)
            push!(status === :ok ? results : failures, payload)
        end
        isempty(failures) || throw(CompositeException(failures))
        sig = copy(first(results).signal)
        for result in Iterators.drop(results, 1)
            sig .+= result.signal
        end
        Xt = return_type == "state" ?
            Mag(vcat((r.state.xy for r in results)...), vcat((r.state.z for r in results)...)) : nothing
        params = copy(first(results).params)
        for key in ("spin_range", "shard_index", "nshards")
            delete!(params, key)
        end
        params["return_type"] = return_type
        params["distribution"] = "spins"
        params["gpus"] = [w.device for w in plan.workers]
        params["spin_ranges"] = [[first(r), last(r)] for r in plan.ranges]
        seconds = (time_ns() - started) / 1e9
        verbose && @info "Completed prepared high-order simulation" seconds devices=params["gpus"]
        result = (; signal=sig, state=Xt, params, seconds, device=join((r.device for r in results), "; "))
        return finalize_simulation!(result, plan.setup)
    end
end

function run_prepared_shard!(data, plan, i, callbacks, callback_lock, mode, precision, return_type)
    CUDA.device!(plan.workers[i].device)
    CUDA.math_mode!(mode; precision)
    restore_simulation_inputs!(data.obj, data.host_obj)
    restore_simulation_inputs!(data.hoseqd, data.host_hoseqd)
    fill!(data.Xt.xy, zero(eltype(data.Xt.xy)))
    copyto!(data.Xt.z, data.obj.ρ)
    fill!(data.sig, zero(eltype(data.sig)))
    params = copy(data.params)
    params["return_type"] = return_type
    params["shard_index"], params["nshards"] = i, length(plan.workers)
    params["spin_range"] = plan.ranges[i]
    local_callbacks = length(plan.workers) == 1 ? callbacks : map(callbacks) do callback
        (progress, blocks, device_data, actual) -> lock(callback_lock) do
            info = merge(progress, (; shard_index=i, nshards=length(plan.workers),
                gpu_device=plan.workers[i].device, spin_range=plan.ranges[i]))
            callback(info, blocks, device_data, actual)
        end
    end
    (; parts, excitation_bool, adc_counts) = plan.setup.plan
    run_sim_time_iter!(data.obj, data.hoseqd, data.sig, data.Xt, params["sim_method"];
        Nblocks=length(parts), Nthreads=1, parts, excitation_bool, adc_counts,
        backend=plan.setup.backend, workspace=data.workspace, callbacks=local_callbacks,
        sim_params=params, verbose=false, sequence_blocks=data.blocks, signal_blocks=data.signals)
    copyto!(data.host_signal, data.sig)
    state = return_type == "state" ? cpu(data.Xt) : nothing
    return (; signal=data.host_signal, state, params, device=KomaMRICore.device_name(plan.setup.backend))
end
