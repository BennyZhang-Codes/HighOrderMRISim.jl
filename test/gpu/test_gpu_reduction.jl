# Run with one GPU exposed by UUID; the default Julia environment is retained.
using Test, CUDA, HighOrderMRISim, LinearAlgebra, Random
import KomaMRICore
isdefined(@__MODULE__, :high_order_workload) ||
    include(joinpath(@__DIR__, "..", "support", "high_order_workload.jl"))

function check_coil_reduction(T, nspins, nadc, ncoils; items=8, groupsize=256)
    rng = MersenneTwister(42)
    states = randn(rng, Complex{T}, nspins, nadc)
    csm = randn(rng, Complex{T}, nspins, ncoils)
    reference = transpose(ComplexF64.(states)) * ComplexF64.(csm)
    device_states, device_csm = CuArray(states), CuArray(csm)
    partials = HighOrderMRISim.high_order_coil_partials(device_states, ncoils; items, groupsize)
    # Deliberately non-contiguous coil columns and protected adjacent rows.
    parent = CUDA.fill(Complex{T}(7, -3), nadc + 4, ncoils, 1)
    sig = view(parent, 3:nadc+2, :, 1)
    HighOrderMRISim.reduce_high_order_signal!(
        sig, device_states, device_csm, partials, KomaMRICore.get_backend(true);
        items, groupsize,
    )
    CUDA.synchronize()
    tolerance = T == Float32 ? 5e-6 : 5e-13
    scale = max(norm(reference), sqrt(nspins * nadc * ncoils), 1)
    @test norm(Array(sig) - reference) / scale < tolerance
    @test all(isfinite, Array(sig))
    @test all(==(Complex{T}(7, -3)), Array(parent)[[1, 2, nadc+3, nadc+4], :, :])
    @test Array(device_states) == states
    @test Array(device_csm) == csm
    return nothing
end

function check_reduction_block(precision, rf, nadc, ncoils)
    source = high_order_workload(; nspins=2051, ncoils, rf)
    fill!(source.hoseqd.seqd.ADC, false)
    source.hoseqd.seqd.ADC[2:nadc+1] .= true
    source = merge(source, (; signal=zeros(ComplexF64, nadc, ncoils)))
    backend = KomaMRICore.get_backend(true)
    reference, candidate = (convert_workload(source, precision, backend) for _ in 1:2)
    operation = rf ? HighOrderMRISim.run_high_order_excitation! : HighOrderMRISim.run_high_order_precession!
    # A shorter ADC block must reuse a workspace sized for the maximum block.
    for (workload, mode) in ((reference, "gemm"), (candidate, "reduction"))
        workspace = HighOrderMRISim.HighOrderGPUWorkspace(workload.obj, 32, backend;
                                                         coil_combination=mode)
        operation(workspace, workload.obj, workload.hoseqd, workload.signal,
                  workload.state, BlochHighOrder("1111"))
    end
    CUDA.synchronize()
    tolerance = precision == "f32" ? 3e-5 : 1e-11
    @test Array(candidate.signal) ≈ Array(reference.signal) atol=tolerance rtol=tolerance
    @test Array(candidate.state.xy) == Array(reference.state.xy)
    @test Array(candidate.state.z) == Array(reference.state.z)
    return nothing
end

function check_reduction_cancellation(T)
    states = fill(Complex{T}(1, -2), 4097, 3)
    states[2:2:4096, :] .*= -1
    states[end, :] .= Complex{T}(1e-3, 2e-3)
    csm = fill(Complex{T}(0.5, 0.25), 4097, 3)
    reference = transpose(ComplexF64.(states)) * ComplexF64.(csm)
    device_states, device_csm = CuArray(states), CuArray(csm)
    sig = CUDA.zeros(Complex{T}, 3, 3)
    partials = HighOrderMRISim.high_order_coil_partials(device_states, 3)
    HighOrderMRISim.reduce_high_order_signal!(sig, device_states, device_csm,
        partials, KomaMRICore.get_backend(true))
    @test Array(sig) ≈ reference atol=(T == Float32 ? 1e-7 : 1e-14) rtol=0
    return nothing
end

function test_gpu_reduction()
    CUDA.functional() || error("CUDA is required.")
    length(collect(CUDA.devices())) == 1 || error("Expose one GPU by UUID.")
    CUDA.device!(0)
    CUDA.allowscalar(false)
    println("Julia=", VERSION, " CUDA=", pkgversion(CUDA),
            " GPU=", CUDA.name(CUDA.device()), " UUID=", CUDA.uuid(CUDA.device()))
    flush(stdout)
    @testset "Fused coil weighting and reduction" begin
        for T in (Float32, Float64), nspins in (0, 1, 259, 2049, 32769),
            (nadc, ncoils) in ((0, 3), (1, 1), (15, 3), (32, 32))
            check_coil_reduction(T, nspins, nadc, ncoils)
        end
        for items in (1, 4, 16), groupsize in (64, 128, 256)
            check_coil_reduction(Float64, 2051, 7, 3; items, groupsize)
        end
        check_reduction_cancellation.((Float32, Float64))
        # More than 65,535 output channels must not hit CUDA's grid-y limit.
        check_coil_reduction(Float32, 1, 257, 257)
        @test_throws ArgumentError HighOrderMRISim.high_order_coil_partials(CUDA.zeros(ComplexF32, 1, 1), 1; groupsize=3)
        @test_throws ArgumentError HighOrderMRISim.high_order_coil_partials(CUDA.zeros(ComplexF32, 1, 1), 1; items=0)
    end
    @testset "Reduction in high-order precession and RF blocks" begin
        for precision in ("f32", "f64"), rf in (false, true), nadc in (0, 1, 7, 32),
            ncoils in (1, 3, 32)
            check_reduction_block(precision, rf, nadc, ncoils)
        end
    end
    println("GPU_COIL_REDUCTION_TESTS_OK")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_gpu_reduction()
end
