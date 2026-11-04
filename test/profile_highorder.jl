# Use HIGHORDERMRISIM_TOOL_ENV for optional development tools without changing
# the default active Julia environment or adding package runtime dependencies.
haskey(ENV, "HIGHORDERMRISIM_TOOL_ENV") && push!(LOAD_PATH, ENV["HIGHORDERMRISIM_TOOL_ENV"])
using CUDA
using HighOrderMRISim
using BenchmarkTools
using Profile
using TOML
using LinearAlgebra
import KomaMRICore
include("performance_workload.jl")

function run_profiled_block!(workspace, workload, rf)
    if isnothing(workspace)
        run_reference_block!(workload, rf)
    else
        operation = rf ? HighOrderMRISim.run_high_order_excitation! : HighOrderMRISim.run_high_order_precession!
        operation(workspace, workload.obj, workload.hoseqd, workload.signal,
                  workload.state, BlochHighOrder("1111"))
    end
    return nothing
end

function profile_high_order(output_directory; nspins=32768, implementation="array")
    implementation in ("array", "kernel") || error("Expected array or kernel.")
    CUDA.functional() || error("CUDA is required.")
    length(collect(CUDA.devices())) == 1 || error("Expose one GPU by UUID.")
    CUDA.device!(0)
    CUDA.allowscalar(false)
    backend = KomaMRICore.get_backend(true)
    mkpath(output_directory)
    results = Dict{String,Any}(
        "julia_version" => string(VERSION), "active_project" => Base.active_project(),
        "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
        "gpu_name" => CUDA.name(CUDA.device()), "spins" => nspins,
        "steps" => 32, "coils" => 32, "blas" => string(BLAS.get_config()),
        "implementation" => implementation,
        "julia_threads" => Threads.nthreads(),
    )
    for precision in ("f32", "f64"), rf in (false, true)
        label = precision * (rf ? "_rf" : "_precession")
        println("Profiling ", implementation, " / ", label)
        flush(stdout)
        workload = convert_workload(high_order_workload(; nspins, rf), precision, backend)
        workspace = implementation == "kernel" ?
            HighOrderMRISim.HighOrderGPUWorkspace(workload.obj, 32, backend) : nothing
        run_profiled_block!(workspace, workload, rf)
        CUDA.synchronize()
        trial = @benchmark (CUDA.@sync run_profiled_block!($workspace, $workload, $rf)) setup=(reset_workload!($workload); CUDA.synchronize()) samples=10 evals=1 seconds=5
        display(trial)
        estimate = median(trial)
        results[label] = Dict(
            "median_seconds" => estimate.time / 1e9,
            "cpu_bytes" => estimate.memory, "cpu_allocations" => estimate.allocs,
        )
        reset_workload!(workload)
        CUDA.@time run_profiled_block!(workspace, workload, rf)
        if precision == "f32"
            Profile.clear()
            @profile for _ in 1:5
                run_profiled_block!(workspace, workload, rf)
                CUDA.synchronize()
            end
            open(joinpath(output_directory, "$implementation-$label-cpu-profile.txt"), "w") do io
                Profile.print(io; format=:flat, sortedby=:count)
            end
            println("Device trace: ", label)
            display(CUDA.@profile run_profiled_block!(workspace, workload, rf))
        end
        CUDA.synchronize()
        CUDA.reclaim()
        flush(stdout)
    end
    open(joinpath(output_directory, "$implementation-block-baseline.toml"), "w") do io
        TOML.print(io, results; sorted=true)
    end
    println(uppercase(implementation), "_BLOCK_PROFILE_OK")
    return nothing
end

length(ARGS) in (1, 2, 3) || error("Expected OUTPUT_DIRECTORY [SPINS] [array|kernel|both].")
for implementation in (length(ARGS) == 3 && ARGS[3] == "both" ? ("array", "kernel") :
                       (length(ARGS) == 3 ? ARGS[3] : "array",))
    profile_high_order(abspath(ARGS[1]);
        nspins=length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32768, implementation,
    )
end
