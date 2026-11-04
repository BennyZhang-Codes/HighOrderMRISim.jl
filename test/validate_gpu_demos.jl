#=
Run an existing demo's full-resolution simulation without plotting or reconstruction.

Usage:
    CUDA_VISIBLE_DEVICES=<GPU UUID> MPLBACKEND=Agg julia --threads=4 \
        test/validate_gpu_demos.jl 2d_nominal|2d_highorder|3d OUTPUT_DIRECTORY

Use the default Julia environment with this repository registered via Pkg.develop.
HIGHORDERMRISIM_EXPECT_GPU_UUID optionally verifies the selected physical device.
HIGHORDERMRISIM_DEMO_MAX_BLOCK_LENGTH controls time blocking without changing the
sequence, phantom grid, number of shots, or physical model.
=#
using CUDA
using HighOrderMRISim
using MAT
using TOML
using Dates
using LinearAlgebra
using Test
import KomaMRICore

module DemoInputs end

function validate_gpu_demo(case, output_directory)
    case in ("2d_nominal", "2d_highorder", "3d") || error("Unknown demo case: $case")
    CUDA.functional() || error("CUDA is not functional.")
    length(collect(CUDA.devices())) == 1 ||
        error("Expose exactly one physical device with CUDA_VISIBLE_DEVICES.")
    CUDA.device!(0)
    CUDA.allowscalar(false)
    device_uuid = string(CUDA.uuid(CUDA.device()))
    expected_uuid = replace(get(ENV, "HIGHORDERMRISIM_EXPECT_GPU_UUID", ""), "GPU-" => "")
    isempty(expected_uuid) || device_uuid == expected_uuid ||
        error("The selected CUDA UUID does not match the expected physical GPU.")
    KomaMRICore.get_backend(true) isa KomaMRICore.KA.GPU || error("A GPU backend is required.")
    ENV["HIGHORDERMRISIM_DEMO_PLOTS"] = "false"
    ENV["HIGHORDERMRISIM_DEMO_SIMULATION_ONLY"] = "true"
    ENV["HIGHORDERMRISIM_DEMO_ORDERS"] = case == "2d_nominal" ? "0000" : "1111"
    get!(ENV, "HIGHORDERMRISIM_DEMO_MAX_BLOCK_LENGTH", "32")
    get!(ENV, "HIGHORDERMRISIM_DEMO_IMPLEMENTATION", "kernel")
    if haskey(ENV, "HIGHORDERMRISIM_DEMO_PRECISION")
        ENV["HIGHORDERMRISIM_DEMO_PRECISION"] in ("f32", "f64") ||
            error("HIGHORDERMRISIM_DEMO_PRECISION must be f32 or f64.")
    end
    root = normpath(joinpath(@__DIR__, ".."))
    mkpath(output_directory)
    println("Case: ", case)
    println("Julia: ", VERSION)
    println("Active project: ", Base.active_project())
    println("Source: ", pathof(HighOrderMRISim))
    println("Physical GPU UUID: ", device_uuid)
    println("GPU name: ", CUDA.name(CUDA.device()))
    println("Started: ", now())
    flush(stdout)
    wall_seconds = @elapsed if case == "3d"
        Base.include(DemoInputs, joinpath(root, "demo3D", "1_load_seq.jl"))
        Base.include(DemoInputs, joinpath(root, "demo3D", "2_load_phantom.jl"))
        Base.include(DemoInputs, joinpath(root, "demo3D", "3_run_sim.jl"))
    else
        Base.include(DemoInputs, joinpath(root, "demo2D", "Sim2D.jl"))
    end
    CUDA.synchronize()
    # include creates new global bindings in Julia 1.12. Read them only after
    # crossing a world-age boundary, rather than resolving them in this caller.
    return Base.invokelatest(
        finalize_gpu_demo, case, output_directory, wall_seconds, device_uuid,
    )
end

function finalize_gpu_demo(case, output_directory, wall_seconds, device_uuid)
    signal = DemoInputs.signal
    obj = DemoInputs.obj
    seq = case == "3d" ? DemoInputs.hoseq.SEQ : DemoInputs.hoseqMeasured.SEQ
    expected_adc = sum(seq.ADC.N)
    ncoils = size(obj.csm, 2)
    @testset "$case full-resolution GPU simulation" begin
        @test size(signal) == (expected_adc, ncoils, 1)
        @test all(isfinite, signal)
        @test norm(signal) > 0
        @test KomaMRICore.get_backend(true) isa KomaMRICore.KA.GPU
        @test string(CUDA.uuid(CUDA.device())) == device_uuid
    end
    summary = Dict(
        "case" => case,
        "completed_at" => string(now()),
        "julia_version" => string(VERSION),
        "active_project" => Base.active_project(),
        "source" => pathof(HighOrderMRISim),
        "gpu_uuid" => device_uuid,
        "gpu_name" => CUDA.name(CUDA.device()),
        "koma_mri_version" => string(pkgversion(HighOrderMRISim.KomaMRI)),
        "koma_core_version" => string(pkgversion(KomaMRICore)),
        "orders" => ENV["HIGHORDERMRISIM_DEMO_ORDERS"],
        "precision" => DemoInputs.sim_params["precision"],
        "implementation" => DemoInputs.sim_params["implementation"],
        "matrix_size" => collect(DemoInputs.target_matrix),
        "resolution_m" => collect(DemoInputs.target_res),
        "spins" => length(obj),
        "coils" => ncoils,
        "adc_samples" => expected_adc,
        "adc_readouts" => count(adc -> adc.N > 0, seq.ADC),
        "sequence_duration_s" => dur(seq),
        "max_block_length" => DemoInputs.sim_params["max_block_length"],
        "signal_shape" => collect(size(signal)),
        "signal_norm" => norm(signal),
        "signal_max_abs" => maximum(abs, signal),
        "all_finite" => all(isfinite, signal),
        "wall_seconds_including_setup_and_compilation" => wall_seconds,
        "reconstruction_performed" => false,
    )
    signal_path = joinpath(output_directory, "$case-signal.mat")
    MAT.matwrite(signal_path, Dict("signal" => signal, "spin_density" => obj.ρ))
    summary_path = joinpath(output_directory, "$case-summary.toml")
    open(summary_path, "w") do io
        TOML.print(io, summary; sorted=true)
    end
    println("Signal: ", signal_path)
    println("Summary: ", summary_path)
    println("DEMO_GPU_VALIDATION_OK: ", case)
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Expected a demo case and an output directory.")
    validate_gpu_demo(ARGS[1], abspath(ARGS[2]))
end
