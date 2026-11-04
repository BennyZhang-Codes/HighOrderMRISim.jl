# Recheck saved full-resolution demo outputs without CUDA or simulation.
# Usage: julia test/audit_gpu_demos.jl RESULT_DIRECTORY
using MAT
using TOML
using LinearAlgebra
using Test

function audit_gpu_demo_outputs(result_directory)
    audit = Dict{String,Any}()
    @testset "Saved GPU demo outputs" begin
        for case in ("2d_nominal", "2d_highorder", "3d")
            @testset "$case" begin
                summary = TOML.parsefile(joinpath(result_directory, "$case-summary.toml"))
                data = matread(joinpath(result_directory, "$case-signal.mat"))
                signal = data["signal"]
                nreadouts = summary["adc_readouts"]
                @test size(signal) == Tuple(summary["signal_shape"])
                @test all(isfinite, signal)
                @test length(data["spin_density"]) == summary["spins"]
                @test norm(signal) ≈ summary["signal_norm"]
                # These demo inputs use an equal sample count in every readout.
                @test size(signal, 1) % nreadouts == 0
                samples_per_readout = div(size(signal, 1), nreadouts)
                readouts = reshape(signal, samples_per_readout, nreadouts, size(signal, 2))
                readout_norms = [norm(view(readouts, :, shot, :)) for shot in 1:nreadouts]
                coil_norms = [norm(view(signal, :, coil, 1)) for coil in axes(signal, 2)]
                zero_samples = count(axes(signal, 1)) do sample
                    all(iszero, view(signal, sample, :, 1))
                end
                @test all(>(0), readout_norms)
                @test all(>(0), coil_norms)
                @test zero_samples == 0
                audit[case] = Dict(
                    "readouts" => nreadouts, "samples_per_readout" => samples_per_readout,
                    "readout_norms" => readout_norms, "coil_norms" => coil_norms,
                    "all_finite" => all(isfinite, signal), "zero_signal_samples" => zero_samples,
                )
                println(case, ": all readouts and coils have nonzero finite signal.")
            end
        end
    end
    open(joinpath(result_directory, "saved-output-audit.toml"), "w") do io
        TOML.print(io, audit; sorted=true)
    end
    println("SAVED_OUTPUT_AUDIT_OK")
    return nothing
end

length(ARGS) == 1 || error("Expected RESULT_DIRECTORY.")
audit_gpu_demo_outputs(abspath(only(ARGS)))
