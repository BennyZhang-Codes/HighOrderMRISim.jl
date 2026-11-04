# Full-resolution, same-process array/kernel comparison after untimed warmup.
# Usage: julia --threads=4 test/benchmark_gpu_demos.jl CASE OUTPUT_DIRECTORY [REPEATS]
include("validate_gpu_demos.jl")
using Statistics

function benchmark_loaded_demo(case, output_directory, repeats)
    obj = DemoInputs.obj
    hoseq = case == "3d" ? DemoInputs.hoseq : DemoInputs.hoseqMeasured
    sys = DemoInputs.sys
    base_params = DemoInputs.sim_params
    results = Dict{String,Any}(
        "case" => case, "julia_version" => string(VERSION),
        "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
        "precision" => base_params["precision"], "repeats" => repeats,
        "matrix_size" => collect(DemoInputs.target_matrix),
        "spins" => length(obj), "max_block_length" => base_params["max_block_length"],
        "timing_scope" => "simulate call including sampling and transfers, excluding package load, demo input preparation, compilation warmup, and file writes",
    )
    signals = Dict{String,Any}()
    for implementation in ("array", "kernel")
        params = merge(base_params, Dict("implementation" => implementation))
        println("Warmup: ", case, " / ", implementation, " (two spins, complete sequence)")
        flush(stdout)
        simulate(obj[1:2], hoseq, sys; sim_params=params, verbose=false)
        CUDA.synchronize()
        elapsed, allocations = Float64[], Int[]
        for repeat in 1:repeats
            GC.gc()
            CUDA.reclaim()
            callback = function (progress, _, _, actual)
                actual["implementation"] == implementation || error("Wrong execution path.")
                if progress.block == 1 || progress.block % 1000 == 0 || progress.block == progress.Nblocks
                    println(case, " / ", implementation, " / ", repeat,
                            ": block ", progress.block, "/", progress.Nblocks,
                            ", samples ", progress.samples-1)
                    flush(stdout)
                end
                return nothing
            end
            measured = @timed CUDA.@sync simulate(
                obj, hoseq, sys; sim_params=params, callbacks=(callback,), verbose=false,
            )
            push!(elapsed, measured.time)
            push!(allocations, measured.bytes)
            signals[implementation] = measured.value
            println("Timed simulation: ", implementation, " = ", measured.time, " s")
            flush(stdout)
        end
        results[implementation] = Dict(
            "seconds" => elapsed, "median_seconds" => median(elapsed),
            "cpu_bytes" => allocations,
        )
        MAT.matwrite(joinpath(output_directory, "$case-$implementation-signal.mat"),
                     Dict("signal" => signals[implementation]))
    end
    reference, candidate = signals["array"], signals["kernel"]
    difference = candidate - reference
    relative_error = norm(difference) / norm(reference)
    relative_peak = maximum(abs, difference) / maximum(abs, reference)
    results["relative_signal_l2_error"] = relative_error
    results["relative_signal_peak_error"] = relative_peak
    results["speedup"] = results["array"]["median_seconds"] / results["kernel"]["median_seconds"]
    open(joinpath(output_directory, "$case-array-kernel-comparison.toml"), "w") do io
        TOML.print(io, results; sorted=true)
    end
    # The long 3D array/f32 baseline itself differs from the Float64 oracle by
    # about 4.2e-4 at the peak. Do not treat that baseline as exact ground truth;
    # compare_precision_oracle.jl separately enforces a 1e-5 kernel accuracy bound.
    tolerance = base_params["precision"] == "f32" ? 5e-4 : 1e-9
    @testset "$case full-resolution array/kernel agreement" begin
        @test size(candidate) == size(reference)
        @test all(isfinite, candidate)
        @test relative_error < tolerance
        @test relative_peak < tolerance
    end
    println("Relative signal L2 error: ", relative_error)
    println("Same-process simulation speedup: ", results["speedup"])
    println("FULL_DEMO_KERNEL_COMPARISON_OK: ", case)
    return nothing
end

function benchmark_gpu_demo(case, output_directory, repeats)
    repeats > 0 || error("REPEATS must be positive.")
    validate_gpu_demo(case, output_directory)
    Base.invokelatest(benchmark_loaded_demo, case, output_directory, repeats)
    return nothing
end

length(ARGS) in (2, 3) || error("Expected CASE OUTPUT_DIRECTORY [REPEATS].")
benchmark_gpu_demo(ARGS[1], abspath(ARGS[2]), length(ARGS) == 3 ? parse(Int, ARGS[3]) : 2)
