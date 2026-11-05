# Explicit device selection is opt-in. Legacy KomaMRI parameters remain supported.
function set_simulation_devices!(params, array_type, gpus, nspins)
    array_type in (nothing, Array, CUDA.CuArray) ||
        throw(ArgumentError("array_type must be Array or CUDA.CuArray."))
    isnothing(gpus) || gpus isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("gpus must be a vector or tuple of visible-device IDs."))
    if array_type === Array
        isnothing(gpus) || isempty(gpus) ||
            throw(ArgumentError("Nonempty gpus requires array_type=CUDA.CuArray."))
        params["gpu"] = false
        return nothing
    end
    isnothing(array_type) && isnothing(gpus) && return nothing
    ids = isnothing(gpus) ? [something(params["gpu_device"], 0)] : collect(gpus)
    !isempty(ids) && all(id -> id isa Integer && !(id isa Bool) && id >= 0, ids) ||
        throw(ArgumentError("gpus must contain nonnegative integer visible-device IDs."))
    length(unique(ids)) == length(ids) || throw(ArgumentError("gpus contains duplicate IDs."))
    length(ids) <= nspins || throw(ArgumentError("Each GPU needs at least one spin."))
    CUDA.functional() || throw(ArgumentError("Explicit CUDA execution requires functional CUDA."))
    ndevices = length(collect(CUDA.devices()))
    all(id -> id < ndevices, ids) || throw(ArgumentError("gpus contains an unavailable device ID."))
    params["gpu"] = true
    params["gpu_device"] = first(ids)
    params["gpus"] = Int.(ids)
    return Int.(ids)
end

function simulation_spin_ranges(nspins, nshards)
    1 <= nshards <= nspins || throw(ArgumentError("Require 1 <= nshards <= nspins."))
    return [fld((i-1)*nspins, nshards)+1:fld(i*nspins, nshards) for i in 1:nshards]
end

function simulate_distributed(
    obj, hoseqd, seq, sys, params, backend, plan, devices; callbacks=(), verbose=false,
)
    # Stage on the host before partitioning: a worker must not consume another
    # device's CuArray views. Phantom slices own their data and magnetization.
    to_precision = params["precision"] == "f32" ? f32 : f64
    host_obj = to_precision(cpu(obj))
    # Sampling is common to all shards. Convert it once on the host rather
    # than creating one full precision-converted sequence per worker.
    hoseqd = to_precision(hoseqd)
    ranges = simulation_spin_ranges(length(host_obj), length(devices))
    callback_lock = ReentrantLock()
    caller_device = CUDA.device()
    math_mode, math_precision = CUDA.math_mode(), CUDA.math_precision()
    tasks = Task[]
    started = time_ns()
    try
        @sync for (i, id) in enumerate(devices)
            task = Threads.@spawn begin
                # CUDA contexts and math modes are task-local, not thread IDs.
                CUDA.device!(id)
                CUDA.math_mode!(math_mode; precision=math_precision)
                shard_params = copy(params)
                shard_params["gpu_device"] = id
                shard_params["spin_range"] = ranges[i]
                shard_params["shard_index"] = i
                shard_params["nshards"] = length(devices)
                shard_callbacks = map(callbacks) do callback
                    (progress, blocks, data, p) -> lock(callback_lock) do
                        info = merge(progress, (; shard_index=i, nshards=length(devices),
                                                 gpu_device=id, spin_range=ranges[i]))
                        callback(info, blocks, data, p)
                    end
                end
                simulate_shard(host_obj[ranges[i]], hoseqd, seq, sys, shard_params,
                               backend, plan; callbacks=shard_callbacks, verbose=false)
            end
            push!(tasks, task)
        end
        results = fetch.(tasks)
        sig = copy(first(results).signal)
        for result in Iterators.drop(results, 1)
            sig .+= result.signal
        end
        Xt = Mag(vcat((r.state.xy for r in results)...), vcat((r.state.z for r in results)...))
        actual_params = copy(first(results).params)
        for key in ("spin_range", "shard_index", "nshards")
            delete!(actual_params, key)
        end
        actual_params["spin_ranges"] = [[first(r), last(r)] for r in ranges]
        actual_params["distribution"] = "spins"
        seconds = (time_ns() - started) / 1e9
        verbose && @info "Completed distributed high-order simulation" devices spins=length(obj) seconds
        return (; signal=sig, state=Xt, params=actual_params,
                device=join((r.device for r in results), "; "), seconds)
    finally
        CUDA.device!(caller_device)
    end
end
