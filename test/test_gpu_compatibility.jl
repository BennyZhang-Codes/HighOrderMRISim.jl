# Standalone GPU regression test. Select the physical GPU before starting Julia:
# CUDA_VISIBLE_DEVICES=<GPU UUID> julia --threads=4 test/test_gpu_compatibility.jl
using Test
using CUDA
using HighOrderMRISim

isdefined(@__MODULE__, :compatibility_sequence) ||
    include("test_koma_compatibility.jl")

CUDA.functional() || error("A functional CUDA device is required for this test.")
CUDA.allowscalar(false)
CUDA.device!(0)
println("GPU regression device: ", CUDA.name(CUDA.device()), " / ", CUDA.uuid(CUDA.device()))

@testset "CUDA high-order compatibility" begin
    for precision in ("f32", "f64"), orders in ("0000", "1000", "1111")
        @testset "$precision / $orders" begin
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
                @test data.obj.x isa CUDA.CuArray
                @test parent(data.sig) isa CUDA.CuArray
                gpu_blocks[] += 1
                return nothing
            end
            signal = simulate(
                obj, hoseq, Scanner(); verbose=false, callbacks=(callback,),
                sim_params=merge(params, Dict("gpu" => true, "gpu_device" => 0)),
            )
            tolerance = precision == "f32" ? 3e-5 : 1e-11
            @test signal ≈ reference atol=tolerance rtol=tolerance
            @test all(isfinite, signal)
            @test gpu_blocks[] > 0
            @test obj.ρ == density
        end
    end
end
