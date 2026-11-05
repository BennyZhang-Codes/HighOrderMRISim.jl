# Run separately from the single-device regression suite.
using Test, CUDA, HighOrderMRISim, LinearAlgebra
include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

function test_distributed_simulation()
    CUDA.functional() || error("Functional CUDA is required.")
    devices = collect(0:length(collect(CUDA.devices()))-1)
    length(devices) >= 2 || error("Expose at least two GPUs.")
    CUDA.device!(first(devices))
    CUDA.allowscalar(false)
    CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    @testset "Distributed spin simulation" begin
        u = collect(range(-0.01, 0.01; length=11))
        obj = HO_Phantom(x=u, y=sin.(u), z=cos.(u)./100,
            ρ=1 .+ u, T1=0.8 .+ u, T2=0.07 .+ u, Δw=80 .* u,
            csm=[cis(c*x) for x in u, c in 1:3])
        density = copy(obj.ρ)
        # Two RF pulses, finite relaxation, and ADC within both RF pulses.
        hoseq = HO_Sequence(rf_acquisition_sequence(3, 2, 11))
        for block in axes(hoseq.GR_dfc, 2), term in 1:16
            scale = term == 1 ? 1e-7 : term <= 4 ? 2e-4 : term <= 9 ? 1e-3 : 1e-2
            hoseq.GR_dfc[term, block] = Grad(scale*sin(term+block), hoseq.SEQ.DUR[block])
        end
        for precision in ("f32", "f64"), mode in ("gemm", "reduction", "fused_warp", "fused_tiled")
            params = Dict{String,Any}("sim_method" => BlochHighOrder("1111"),
                "precision" => precision, "return_type" => "mat",
                "max_block_length" => 4, "max_rf_block_length" => 5,
                "implementation" => "kernel", "coil_combination" => mode)
            original = copy(params)
            reference = simulate(obj, hoseq, Scanner(); sim_params=params,
                array_type=CuArray, gpus=[0], verbose=false)
            seen = Dict{Int,Vector{Int}}()
            function callback(progress, _, data, actual)
                @test CUDA.deviceid(CUDA.device()) == progress.gpu_device
                @test CUDA.math_mode() == CUDA.PEDANTIC_MATH
                @test length(data.obj) == length(progress.spin_range)
                @test actual["shard_index"] == progress.shard_index
                push!(get!(seen, progress.shard_index, Int[]), progress.block)
                return nothing
            end
            signal = simulate(obj, hoseq, Scanner(); sim_params=params,
                array_type=CuArray, gpus=devices, callbacks=(callback,), verbose=false)
            tol = precision == "f32" ? 5e-5 : 2e-12
            @test signal ≈ reference rtol=tol atol=tol
            @test eltype(signal) == (precision == "f32" ? ComplexF32 : ComplexF64)
            @test length(seen) == length(devices)
            @test all(v == collect(1:length(v)) for v in values(seen))
            @test CUDA.deviceid(CUDA.device()) == 0
            state_params = merge(params, Dict("return_type" => "state"))
            single = simulate(obj, hoseq, Scanner(); sim_params=state_params,
                              array_type=CuArray, gpus=[0], verbose=false)
            multi = simulate(obj, hoseq, Scanner(); sim_params=state_params,
                             array_type=CuArray, gpus=reverse(devices), verbose=false)
            @test multi.xy == single.xy
            @test multi.z == single.z
            @test params == original
            @test obj.ρ == density
        end
        params = Dict{String,Any}("sim_method" => BlochHighOrder("1111"),
            "return_type" => "raw", "precision" => "f64", "implementation" => "kernel")
        raw = simulate(obj, hoseq, Scanner(); sim_params=params,
                       array_type=CuArray, gpus=devices, verbose=false)
        @test sum(size(p.data, 1) for p in raw.profiles) == 16
        @test all(size(p.data, 2) == 3 for p in raw.profiles)
        @test_throws ArgumentError simulate(obj, hoseq, Scanner(); gpus=[length(devices)], verbose=false)
        failing = (_, _, _, _) -> error("intentional callback failure")
        @test_throws CompositeException simulate(obj, hoseq, Scanner();
            sim_params=merge(params, Dict("return_type" => "mat")),
            array_type=CuArray, gpus=devices, callbacks=(failing,), verbose=false)
        @test CUDA.deviceid(CUDA.device()) == 0
    end
    println("MULTIGPU_SIMULATION_TESTS_OK")
end

test_distributed_simulation()
include("test_prepared_simulation.jl")
test_prepared_simulation([0])
test_prepared_simulation(collect(0:length(collect(CUDA.devices()))-1))
