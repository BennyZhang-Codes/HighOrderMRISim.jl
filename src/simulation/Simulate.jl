"""
    simulate(obj::HO_Phantom, hoseq::HO_Sequence, sys::Scanner;
             sim_params=Dict(), array_type=nothing, gpus=nothing,
             verbose=true, callbacks=(), w=nothing)

Run the legacy high-order Bloch model with current KomaMRI sequence sampling.
The `"mat"`, `"state"`, and `"raw"` return modes are preserved. Matrix output
retains its trailing singleton thread dimension for existing demo scripts.

Both the legacy `"Nblocks"` setting and the current `"max_block_length"` and
`"max_rf_block_length"` settings are accepted. Explicit maximum block lengths
take precedence. `w` is retained only for source compatibility; use callbacks
instead of the removed Blink progress interface.

High-order fields affect free precession, while RF excitation uses nominal
gradients, matching the original package. `"implementation"` selects `"array"`,
`"kernel"`, or `"auto"` (default). Auto uses the kernel for static CUDA high-order
simulations and otherwise preserves the array path. An explicit kernel request
errors if unsupported; it never silently discards motion or changes the model.
For supported kernels, `"coil_combination"` selects `"gemm"` (default, packed
cuBLAS), `"reduction"` (two-stage coil weighting and reduction), or experimental
`"fused_warp"` (precession with subgroup coil reduction; RF retains GEMM).
`"fused_tiled"` additionally reuses small ADC/coil tiles during precession.
The fused choices size reusable ADC/packed buffers from the maximum RF ADC
count, including zero capacity when RF has no acquisition. Callbacks and raw
metadata report `"max_adc_per_block"`, `"max_rf_adc_per_block"`, and
`"adc_state_buffer_samples"`; the caller's parameter dictionary is not changed.

Use `array_type=Array` for CPU execution or `array_type=CUDA.CuArray,
gpus=[0, 1]` for CUDA execution. GPU IDs are zero-based visible-device IDs,
not physical indices when `CUDA_VISIBLE_DEVICES` remaps devices. These keywords
override the legacy `"gpu"`/`"gpu_device"` settings; omitting them preserves
legacy device selection. Explicit CUDA requests fail rather than fall back to CPU.

Multiple GPUs simulate disjoint contiguous spin ranges through the complete
sequence, preserving relaxation and RF history. Signals are summed in device-list
order in the requested precision, and final magnetization is joined in original
spin order. Only summation order changes. Callbacks receive shard-local arrays;
their progress includes `shard_index`, `nshards`, `gpu_device`, and `spin_range`.
Callback invocations are serialized, but blocks from different shards can interleave.

`"synchronization" => "auto"` (default) relies on CUDA stream ordering between
kernel blocks and waits at the end. Callbacks still receive completed block data;
array execution retains block fences. Set `"synchronization" => "block"` to
retain the original per-block policy for diagnostics. Neither policy changes
sampling, arithmetic precision, RF physics, or coil summation.

For repeated runs, [`prepare_simulation`](@ref) provides explicit ownership and
reuse of device inputs, block views, workers, and buffers. Close the resulting
plan when finished; ordinary one-shot calls do not retain a hidden global cache.
"""
function simulate(
    obj::HO_Phantom, hoseq::HO_Sequence, sys::Scanner;
    sim_params=Dict{String,Any}(), array_type=nothing, gpus=nothing,
    verbose=true, callbacks=(), w=nothing,
)
    isnothing(w) || @warn "Blink progress windows are no longer supported; use callbacks."
    setup = setup_simulation(obj, hoseq, sys; sim_params, array_type, gpus, verbose)
    (; params, hoseqd, seq, backend, plan, devices) = setup
    result = if !isnothing(devices) && length(devices) > 1
        simulate_distributed(obj, hoseqd, seq, sys, params, backend, plan, devices;
                             callbacks, verbose)
    else
        simulate_shard(obj, hoseqd, seq, sys, params, backend, plan; callbacks, verbose)
    end
    return finalize_simulation!(result, setup)
end

# One sampling/validation contract for one-shot and explicitly reusable execution.
function setup_simulation(
    obj, hoseq, sys; sim_params=Dict{String,Any}(),
    array_type=nothing, gpus=nothing, verbose=false,
)
    sim_params isa AbstractDict && all(key -> key isa AbstractString, keys(sim_params)) ||
        throw(ArgumentError("sim_params must be a dictionary with string keys."))
    params = KomaMRICore.default_sim_params(Dict{String,Any}(sim_params))
    get(params, "synchronization", "auto") in ("auto", "block") ||
        throw(ArgumentError("synchronization must be auto or block."))
    sim_method = params["sim_method"]
    implementation = get(params, "implementation", "auto")
    implementation in ("auto", "array", "kernel") ||
        throw(ArgumentError("implementation must be auto, array, or kernel."))
    coil_combination = get(params, "coil_combination", "gemm")
    coil_combination in HIGH_ORDER_COIL_COMBINATIONS ||
        throw(ArgumentError("Unknown coil_combination: $coil_combination."))
    sim_method isa Union{Bloch,BlochHighOrder} ||
        throw(ArgumentError("HO_Phantom currently supports Bloch() and BlochHighOrder()."))
    params["return_type"] in ("mat", "state", "raw") ||
        throw(ArgumentError("return_type must be mat, state, or raw."))
    params["precision"] in ("f32", "f64") ||
        throw(ArgumentError("The legacy high-order path supports f32 and f64 precision."))
    params["Nthreads"] isa Integer && params["Nthreads"] > 0 ||
        throw(ArgumentError("Nthreads must be a positive integer."))
    validate_phantom(obj)
    isempty(obj.ρ) && throw(ArgumentError("The phantom must contain at least one spin."))
    if sim_method isa BlochHighOrder
        size(obj.csm, 2) > 0 || throw(ArgumentError("The CSM must contain at least one coil."))
    end
    validate_high_order_sequence(hoseq)
    devices = set_simulation_devices!(params, array_type, gpus, length(obj))

    hoseqd = discretize(hoseq; sampling_params=params)
    if haskey(sim_params, "Nblocks")
        nblocks = sim_params["Nblocks"]
        nblocks isa Integer && nblocks > 0 ||
            throw(ArgumentError("Nblocks must be a positive integer."))
        legacy_length = max(1, cld(length(hoseqd), nblocks))
        haskey(sim_params, "max_block_length") || (params["max_block_length"] = legacy_length)
        haskey(sim_params, "max_rf_block_length") || (params["max_rf_block_length"] = legacy_length)
    end
    for key in ("max_block_length", "max_rf_block_length")
        limit = params[key]
        (limit == Inf || (limit isa Integer && limit > 0)) ||
            throw(ArgumentError("$key must be a positive integer or Inf."))
    end
    parts, excitation_bool = KomaMRICore.get_sim_ranges(
        hoseqd.seqd; max_block_length=params["max_block_length"],
        max_rf_block_length=params["max_rf_block_length"],
    )
    isempty(parts) && throw(ArgumentError("The sequence must contain a simulation interval."))
    adc_counts = [count(view(hoseqd.seqd.ADC, first(p)+1:last(p))) for p in parts]
    max_adc = maximum(adc_counts)
    max_rf_adc = maximum((n for (n, rf) in zip(adc_counts, excitation_bool) if rf); init=0)
    seq = simulation_sequence(hoseq)
    sum(adc_counts) == sum(seq.ADC.N) ||
        throw(DimensionMismatch("Sampled ADC count does not match the native sequence."))
    t_sim_parts = [hoseqd.seqd.t[first(p)] for p in parts]
    push!(t_sim_parts, last(hoseqd.seqd.t))

    backend = KomaMRICore.get_backend(params["gpu"]; verbose)
    !isnothing(devices) && !(backend isa CUDA.CUDABackend) &&
        throw(ArgumentError("Explicit CUDA execution requires a functional CUDA backend."))
    plan = (; parts, excitation_bool, adc_counts, max_adc, max_rf_adc)
    return (; obj, hoseqd, seq, sys, params, backend, plan, devices, t_sim_parts)
end

# All device counts use this same integrator; distribution changes only spin ownership.
function simulate_shard(
    obj, hoseqd, seq, sys, params, backend, plan; callbacks=(), verbose=false,
)
    (; parts, excitation_bool, adc_counts, max_adc, max_rf_adc) = plan
    sim_method = params["sim_method"]
    implementation = get(params, "implementation", "auto")
    coil_combination = get(params, "coil_combination", "gemm")
    Xt, obj = initialize_spins_state(obj, sim_method)
    Ndims = sim_output_dim(obj, seq, sys, sim_method)
    params["gpu"] = backend isa KomaMRICore.KA.GPU
    params["gpu"] && (params["Nthreads"] = 1)
    T = params["precision"] == "f32" ? Float32 : Float64
    sig = zeros(Complex{T}, Ndims..., params["Nthreads"])
    to_precision = params["precision"] == "f32" ? f32 : f64
    obj, hoseqd, Xt, sig = map(to_precision, (obj, hoseqd, Xt, sig))
    if params["gpu"]
        isnothing(params["gpu_device"]) ||
            KomaMRICore.set_device!(backend, params["gpu_device"])
        obj, hoseqd, Xt, sig = map(x -> gpu(x, backend), (obj, hoseqd, Xt, sig))
    end
    kernel_supported = params["gpu"] && obj.x isa CUDA.CuArray &&
        sim_method isa BlochHighOrder && has_static_coordinates(obj)
    implementation == "kernel" && !kernel_supported && throw(ArgumentError(
        "The kernel requires CUDA, BlochHighOrder, and default static coordinates; " *
        "use implementation=array for other configurations.",
    ))
    use_kernel = implementation != "array" && kernel_supported
    coil_combination != "gemm" && !use_kernel && throw(ArgumentError(
        "Coil reduction requires the supported CUDA high-order kernel configuration.",
    ))
    params["implementation"] = use_kernel ? "kernel" : "array"
    params["coil_combination"] = coil_combination
    workspace = use_kernel ? HighOrderGPUWorkspace(
        obj, max_adc, backend; coil_combination, max_rf_adc,
    ) : nothing
    params["max_adc_per_block"] = max_adc
    params["max_rf_adc_per_block"] = max_rf_adc
    params["adc_state_buffer_samples"] = isnothing(workspace) ? 0 : size(workspace.adc_states, 2)
    if verbose
        device = params["gpu"] ? KomaMRICore.device_name(backend) : "CPU"
        @info "Running high-order simulation" device implementation=params["implementation"] spins=length(obj) time_points=length(hoseqd.seqd.t) sim_method koma_version=pkgversion(KomaMRICore)
    end
    timed_tuple = @timed run_sim_time_iter!(
        obj, hoseqd, sig, Xt, sim_method;
        Nblocks=length(parts), Nthreads=params["Nthreads"], parts,
        excitation_bool, adc_counts, backend, workspace, verbose, callbacks, sim_params=params,
    )
    sig = sum(cpu(sig); dims=length(Ndims)+1)
    Xt = cpu(Xt)
    device = params["gpu"] ? KomaMRICore.device_name(backend) : "nothing"
    if obj.x isa CUDA.CuArray
        CUDA.reclaim()
    end

    return (; signal=sig, state=Xt, params, device, seconds=timed_tuple.time)
end

function finalize_simulation!(result, setup)
    (; obj, seq, sys, t_sim_parts, plan) = setup
    (; excitation_bool, parts) = plan
    sim_method = setup.params["sim_method"]
    sig, Xt, params = result.signal, result.state, result.params
    sig .*= get_adc_phase_compensation(seq)
    params["return_type"] == "state" && return Xt
    params["return_type"] == "mat" && return sig
    raw_params = copy(params)
    raw_params["sim_method"] = string(sim_method)
    raw_params["sampling_rule"] = string(high_order_sampling_rule(params))
    raw_params["gpu_device"] = result.device
    raw_params["t_sim_parts"] = t_sim_parts
    raw_params["type_sim_parts"] = excitation_bool
    raw_params["Nblocks"] = length(parts)
    raw_params["sim_time_sec"] = result.seconds
    # Native raw-data trajectories describe XYZ encoding. High-order encoding
    # coefficients remain available separately through get_kspace(hoseq).
    return signal_to_raw_data(sig, seq; phantom_name=obj.name, sys, sim_params=raw_params)
end
