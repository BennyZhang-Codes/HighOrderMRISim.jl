"""
    simulate(obj::HO_Phantom, hoseq::HO_Sequence, sys::Scanner;
             sim_params=Dict(), verbose=true, callbacks=(), w=nothing)

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
"""
function simulate(
    obj::HO_Phantom, hoseq::HO_Sequence, sys::Scanner;
    sim_params=Dict{String,Any}(), verbose=true, callbacks=(), w=nothing,
)
    isnothing(w) || @warn "Blink progress windows are no longer supported; use callbacks."
    params = KomaMRICore.default_sim_params(copy(sim_params))
    method = params["sim_method"]
    implementation = get(params, "implementation", "auto")
    implementation in ("auto", "array", "kernel") ||
        throw(ArgumentError("implementation must be auto, array, or kernel."))
    method isa Union{Bloch,BlochHighOrder} ||
        throw(ArgumentError("HO_Phantom currently supports Bloch() and BlochHighOrder()."))
    params["return_type"] in ("mat", "state", "raw") ||
        throw(ArgumentError("return_type must be mat, state, or raw."))
    params["precision"] in ("f32", "f64") ||
        throw(ArgumentError("The legacy high-order path supports f32 and f64 precision."))
    params["Nthreads"] isa Integer && params["Nthreads"] > 0 ||
        throw(ArgumentError("Nthreads must be a positive integer."))
    isempty(obj.ρ) && throw(ArgumentError("The phantom must contain at least one spin."))
    if method isa BlochHighOrder
        size(obj.csm, 1) == length(obj) ||
            throw(DimensionMismatch("The CSM must have one row per spin."))
        size(obj.csm, 2) > 0 || throw(ArgumentError("The CSM must contain at least one coil."))
    end

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
    seq = simulation_sequence(hoseq)
    sum(adc_counts) == sum(seq.ADC.N) ||
        throw(DimensionMismatch("Sampled ADC count does not match the native sequence."))
    t_sim_parts = [hoseqd.seqd.t[first(p)] for p in parts]
    push!(t_sim_parts, last(hoseqd.seqd.t))

    Xt, obj = initialize_spins_state(obj, method)
    Ndims = sim_output_dim(obj, seq, sys, method)
    backend = KomaMRICore.get_backend(params["gpu"]; verbose)
    params["gpu"] = backend isa KomaMRICore.KA.GPU
    params["gpu"] && (params["Nthreads"] = 1)
    sig = zeros(ComplexF64, Ndims..., params["Nthreads"])
    to_precision = params["precision"] == "f32" ? f32 : f64
    obj, hoseqd, Xt, sig = map(to_precision, (obj, hoseqd, Xt, sig))
    if params["gpu"]
        isnothing(params["gpu_device"]) ||
            KomaMRICore.set_device!(backend, params["gpu_device"])
        obj, hoseqd, Xt, sig = map(x -> gpu(x, backend), (obj, hoseqd, Xt, sig))
    end
    kernel_supported = params["gpu"] && obj.x isa CUDA.CuArray &&
        method isa BlochHighOrder && has_static_coordinates(obj)
    implementation == "kernel" && !kernel_supported && throw(ArgumentError(
        "The kernel requires CUDA, BlochHighOrder, and default static coordinates; " *
        "use implementation=array for other configurations.",
    ))
    use_kernel = implementation != "array" && kernel_supported
    params["implementation"] = use_kernel ? "kernel" : "array"
    workspace = use_kernel ? HighOrderGPUWorkspace(obj, maximum(adc_counts), backend) : nothing
    if verbose
        device = params["gpu"] ? KomaMRICore.device_name(backend) : "CPU"
        @info "Running high-order simulation" device implementation=params["implementation"] spins=length(obj) time_points=length(hoseqd.seqd.t) sim_method=method koma_version=pkgversion(KomaMRICore)
    end
    timed_tuple = @timed run_sim_time_iter!(
        obj, hoseqd, sig, Xt, method;
        Nblocks=length(parts), Nthreads=params["Nthreads"], parts,
        excitation_bool, adc_counts, backend, workspace, verbose, callbacks, sim_params=params,
    )
    sig = sum(cpu(sig); dims=length(Ndims)+1)
    sig .*= get_adc_phase_compensation(seq)
    Xt = cpu(Xt)
    if obj.x isa CUDA.CuArray
        CUDA.reclaim()
    end

    params["return_type"] == "state" && return Xt
    params["return_type"] == "mat" && return sig
    raw_params = copy(params)
    raw_params["sim_method"] = string(method)
    raw_params["sampling_rule"] = string(high_order_sampling_rule(params))
    raw_params["gpu_device"] = params["gpu"] ? KomaMRICore.device_name(backend) : "nothing"
    raw_params["t_sim_parts"] = t_sim_parts
    raw_params["type_sim_parts"] = excitation_bool
    raw_params["Nblocks"] = length(parts)
    raw_params["sim_time_sec"] = timed_tuple.time
    # Native raw-data trajectories describe XYZ encoding. High-order encoding
    # coefficients remain available separately through get_kspace(hoseq).
    return signal_to_raw_data(
        sig, seq; phantom_name=obj.name, sys, sim_params=raw_params,
    )
end
