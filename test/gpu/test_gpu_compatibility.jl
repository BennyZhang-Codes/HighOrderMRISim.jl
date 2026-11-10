# Standalone GPU regression test. Select the physical GPU before starting Julia:
# CUDA_VISIBLE_DEVICES=<GPU UUID> julia --threads=4 test/gpu/test_gpu_compatibility.jl
using Test
using CUDA
using HighOrderMRISim

isdefined(@__MODULE__, :compatibility_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

CUDA.functional() || error("A functional CUDA device is required for this test.")
CUDA.allowscalar(false)
CUDA.device!(0)
println("GPU regression device: ", CUDA.name(CUDA.device()), " / ", CUDA.uuid(CUDA.device()))

@testset "Array coil combination with CUDA views" begin
    # Coherent terms expose the rounding error of a long sequential sum.
    nspins = 2^19
    weights = ComplexF32[0.7-0.13im 0.5+0.25im 0.125-0.25im]
    state = ComplexF32(0.51, 0.24)
    for rows in (3:nspins+2, 3:2:nspins+2)
        csm = CUDA.fill(ComplexF32(-2, 3), nspins + 4, 3)
        view(csm, rows, :) .= CuArray(weights)
        coils = view(csm, rows, :)
        for nadc in (1, 2)
            states = CUDA.fill(state, length(rows), nadc)
            buffer = CUDA.fill(ComplexF32(-2, 3), nadc + 4, 3, 1)
            signal = view(buffer, 3:nadc+2, :, 1)
            HighOrderMRISim.combine_array_signal!(signal, nadc == 1 ? vec(states) : states, coils)
            expected = repeat(length(rows) * ComplexF64(state) * ComplexF64.(weights), nadc, 1)
            @test Array(signal) ≈ expected rtol=1e-4
            @test all(==(ComplexF32(-2, 3)), Array(buffer)[[1, 2, nadc+3, nadc+4], :, :])
        end
    end
end

@testset "Array coil combination with reversed coil columns" begin
    for T in (Float32, Float64)
        # Distinct complex weights detect coil reordering and conjugation errors.
        csm = Complex{T}[complex(T(i)/16, T(j-i)/32) for i in 1:12, j in 1:4]
        source = CuArray(csm)
        coils = view(source, 3:10, 4:-1:1)
        @test stride(coils, 1) == 1
        @test stride(coils, 2) < 0
        for nadc in (1, 3)
            states = Complex{T}[complex(T(i+j)/16, T(2j-i)/32) for i in 1:8, j in 1:nadc]
            device_states = CuArray(states)
            buffer = CUDA.fill(Complex{T}(-2, 3), nadc+4, 4, 1)
            signal = view(buffer, 3:nadc+2, :, 1)
            HighOrderMRISim.combine_array_signal!(signal,
                nadc == 1 ? vec(device_states) : device_states, coils)
            expected = transpose(states) * csm[3:10, 4:-1:1]
            tolerance = T == Float32 ? 3e-5 : 1e-11
            @test Array(signal) ≈ expected atol=tolerance rtol=tolerance
            @test all(==(Complex{T}(-2, 3)), Array(buffer)[[1, 2, nadc+3, nadc+4], :, :])
            @test Array(source) == csm
            @test Array(device_states) == states
        end
    end
end

@testset "CUDA high-order compatibility" begin
    for precision in ("f32", "f64"), orders in ("0000", "1000", "1111"),
        coil_combination in HighOrderMRISim.HIGH_ORDER_COIL_COMBINATIONS
        @testset "$precision / $orders / $coil_combination" begin
            seq = compatibility_sequence()
            hoseq = HO_Sequence(seq)
            for term in 1:16
                hoseq.GR_dfc[term, 2] = Grad(term * 1e-7, 1e-3)
            end
            obj = compatibility_phantom()
            obj.Δw .= [5.0, 25.0]
            density = copy(obj.ρ)
            params = Dict{String,Any}(
                "gpu" => false, "precision" => precision, "Nthreads" => 1,
                "sim_method" => BlochHighOrder(orders), "return_type" => "mat",
                "max_block_length" => 2, "max_rf_block_length" => 1,
            )
            reference = simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false)
            gpu_blocks = Ref(0)
            callback = function (_, _, data, actual_params)
                @test actual_params["gpu"]
                @test actual_params["implementation"] == "kernel"
                @test actual_params["coil_combination"] == coil_combination
                @test data.obj.x isa CUDA.CuArray
                @test parent(data.sig) isa CUDA.CuArray
                gpu_blocks[] += 1
                return nothing
            end
            signal = simulate(
                obj, hoseq, Scanner(); verbose=false, callbacks=(callback,),
                sim_params=merge(params, Dict("gpu" => true, "gpu_device" => 0,
                                             "coil_combination" => coil_combination)),
            )
            tolerance = precision == "f32" ? 3e-5 : 1e-11
            @test signal ≈ reference atol=tolerance rtol=tolerance
            @test all(isfinite, signal)
            @test gpu_blocks[] > 0
            @test obj.ρ == density
            array_signal = simulate(
                obj, hoseq, Scanner(); verbose=false,
                sim_params=merge(params, Dict("gpu" => true, "gpu_device" => 0,
                                             "implementation" => "array")),
            )
            @test signal ≈ array_signal atol=tolerance rtol=tolerance
        end
    end
end

@testset "Execution selection and kernel API" begin
    obj, hoseq = compatibility_phantom(), HO_Sequence(compatibility_sequence())
    params = Dict{String,Any}(
        "gpu" => true, "gpu_device" => 0, "precision" => "f64",
        "sim_method" => BlochHighOrder("1111"), "return_type" => "mat",
        "implementation" => "kernel", "max_block_length" => 1,
    )
    original = copy(params)
    reference = simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false)
    @test params == original
    for block_length in (2, 32, 128)
        signal = simulate(obj, hoseq, Scanner(); verbose=false,
                          sim_params=merge(params, Dict("max_block_length" => block_length)))
        @test signal ≈ reference atol=1e-11 rtol=1e-11
    end
    for (implementation, coil_combination) in (("array", "gemm"),
        (("kernel", mode) for mode in HighOrderMRISim.HIGH_ORDER_COIL_COMBINATIONS)...)
        state = simulate(obj, hoseq, Scanner(); verbose=false,
                         sim_params=merge(params, Dict("implementation" => implementation,
                                                       "coil_combination" => coil_combination,
                                                       "return_type" => "state")))
        @test state.xy ≈ im .* obj.ρ atol=1e-11
        raw = simulate(obj, hoseq, Scanner(); verbose=false,
                       sim_params=merge(params, Dict("implementation" => implementation,
                                                     "coil_combination" => coil_combination,
                                                     "return_type" => "raw")))
        @test sum(size(profile.data, 1) for profile in raw.profiles) == 5
        @test all(size(profile.data, 2) == 2 for profile in raw.profiles)
    end
    @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("implementation" => "unknown")))
    @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("gpu" => false)))
    @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("coil_combination" => "unknown")))
    @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("coil_combination" => "reduction", "implementation" => "array")))
    for mode in ("fused_warp", "fused_tiled")
        @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(params, Dict("coil_combination" => mode, "implementation" => "array")))
        @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(params, Dict("coil_combination" => mode, "implementation" => "auto", "gpu" => false)))
    end
    obj.ux = (x, y, z, t) -> 0
    @test_throws ArgumentError simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false)
    @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("implementation" => "auto", "coil_combination" => "reduction")))
    for mode in ("fused_warp", "fused_tiled")
        @test_throws ArgumentError simulate(obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(params, Dict("implementation" => "auto", "coil_combination" => mode)))
    end
    paths = String[]
    callback = (_, _, _, actual) -> push!(paths, actual["implementation"])
    fallback = simulate(obj, hoseq, Scanner(); verbose=false, callbacks=(callback,),
                        sim_params=merge(params, Dict("implementation" => "auto")))
    @test !isempty(paths) && all(==("array"), paths)
    @test fallback ≈ reference atol=1e-11
end
