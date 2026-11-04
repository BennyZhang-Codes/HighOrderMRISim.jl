# Compare the complete 3D f32 signals with a full-volume f64 reference.
# Usage: julia test/compare_precision_oracle.jl RESULT_DIRECTORY ORACLE_MAT_FILE
using MAT
using TOML
using LinearAlgebra
using Test

function compare_precision_oracle(directory, oracle_path)
    oracle = matread(oracle_path)["signal"]
    reference = matread(joinpath(directory, "3d-array-signal.mat"))["signal"]
    candidate = matread(joinpath(directory, "3d-kernel-signal.mat"))["signal"]
    metrics = Dict{String,Any}()
    for (label, signal) in (("array_f32", reference), ("kernel_f32", candidate))
        difference = signal-oracle
        metrics[label] = Dict(
            "relative_l2_vs_f64" => norm(difference)/norm(oracle),
            "relative_peak_vs_f64" => maximum(abs, difference)/maximum(abs, oracle),
        )
        println(label, ": ", metrics[label])
    end
    @testset "Full-volume 3D precision oracle" begin
        @test eltype(oracle) == ComplexF64
        @test eltype(candidate) == ComplexF32
        @test eltype(reference) == ComplexF32
        @test size(candidate) == size(reference) == size(oracle)
        @test all(isfinite, oracle)
        @test all(isfinite, candidate)
        @test metrics["kernel_f32"]["relative_l2_vs_f64"] < 1e-5
        @test metrics["kernel_f32"]["relative_peak_vs_f64"] < 1e-5
        @test metrics["kernel_f32"]["relative_l2_vs_f64"] <= max(1e-5, metrics["array_f32"]["relative_l2_vs_f64"])
        @test metrics["kernel_f32"]["relative_peak_vs_f64"] <= max(1e-5, metrics["array_f32"]["relative_peak_vs_f64"])
    end
    open(joinpath(directory, "precision-oracle-comparison.toml"), "w") do io
        TOML.print(io, metrics; sorted=true)
    end
    println("FULL_VOLUME_PRECISION_ORACLE_OK")
    return nothing
end

length(ARGS) == 2 || error("Expected RESULT_DIRECTORY ORACLE_MAT_FILE.")
compare_precision_oracle(abspath(ARGS[1]), abspath(ARGS[2]))
