# Standalone CUDA kernel regression; select one physical GPU by UUID.
using Test
using CUDA
using HighOrderMRISim
using LinearAlgebra
import KomaMRICore
isdefined(@__MODULE__, :high_order_workload) ||
    include(joinpath(@__DIR__, "..", "support", "high_order_workload.jl"))

CUDA.functional() || error("CUDA is required.")
length(collect(CUDA.devices())) == 1 || error("Expose one GPU by UUID.")
CUDA.device!(0)
CUDA.allowscalar(false)
const kernel_backend = KomaMRICore.get_backend(true)

function check_precession_kernel(precision, orders; ncoils=32, adc=true, off_resonance=true)
    reference = high_order_workload(; nspins=259, ncoils)
    adc || fill!(reference.hoseqd.seqd.ADC, false)
    if !adc
        reference = merge(reference, (; signal=zeros(ComplexF64, 0, ncoils)))
    end
    cpu = convert_workload(reference, precision)
    device = convert_workload(reference, precision, kernel_backend)
    method = BlochHighOrder(orders, true, off_resonance)
    HighOrderMRISim.run_spin_precession!(cpu.obj, cpu.hoseqd, cpu.signal, cpu.state, method)
    workspace = HighOrderMRISim.HighOrderGPUWorkspace(device.obj, size(device.signal, 1), kernel_backend)
    HighOrderMRISim.run_high_order_precession!(
        workspace, device.obj, device.hoseqd, device.signal, device.state, method,
    )
    CUDA.synchronize()
    tolerance = precision == "f32" ? 3e-5 : 1e-11
    @test Array(device.signal) ≈ cpu.signal atol=tolerance rtol=tolerance
    @test Array(device.state.xy) ≈ cpu.state.xy atol=tolerance rtol=tolerance
    @test Array(device.state.z) ≈ cpu.state.z atol=tolerance rtol=tolerance
    @test Array(device.obj.ρ) == cpu.obj.ρ
    return nothing
end

@testset "High-order GPU precession kernel" begin
    for precision in ("f32", "f64")
        for mask in 0:15
            orders = lpad(string(mask; base=2), 4, '0')
            @testset "$precision / $orders" begin
                check_precession_kernel(precision, orders)
            end
        end
        for ncoils in (1, 3)
            check_precession_kernel(precision, "1111"; ncoils)
        end
        check_precession_kernel(precision, "1111"; adc=false)
        check_precession_kernel(precision, "1111"; off_resonance=false)
    end

    @testset "Analytic phase for each field channel" begin
        source = high_order_workload(; nspins=3, nsteps=4, ncoils=3)
        source.obj.T1 .= Inf
        source.obj.T2 .= Inf
        source.obj.Δw .= 0
        for term in 0:15
            getfield(source.hoseqd, Symbol("h", term)) .= 0
        end
        # All measured terms are enabled, so nominal gradients must not be added.
        for term in 1:16
            field = getfield(source.hoseqd, Symbol("h", term-1))
            field .= 2e-5
            device = convert_workload(source, "f64", kernel_backend)
            workspace = HighOrderMRISim.HighOrderGPUWorkspace(device.obj, 4, kernel_backend)
            HighOrderMRISim.run_high_order_precession!(
                workspace, device.obj, device.hoseqd, device.signal, device.state,
                BlochHighOrder("1111"),
            )
            CUDA.synchronize()
            basis = map(source.obj.x, source.obj.y, source.obj.z) do x, y, z
                r² = x^2 + y^2 + z^2
                (1.0, x, y, z, x*y, z*y, 3z^2-r², x*z, x^2-y^2,
                 3y*x^2-y^3, x*z*y, (5z^2-r²)*y, 5z^3-3z*r²,
                 (5z^2-r²)*x, x^2*z-y^2*z, x^3-3x*y^2)[term]
            end
            phase = -2π * γ .* basis .* 2e-5 .* transpose(source.hoseqd.seqd.t[2:end])
            states = source.state.xy .* cis.(phase)
            @test Array(device.signal) ≈ transpose(states) * source.obj.csm atol=1e-11 rtol=1e-11
            @test Array(device.state.xy) ≈ states[:, end] atol=1e-11 rtol=1e-11
            field .= 0
        end
    end
end

println("GPU_PRECESSION_KERNEL_TESTS_OK")

@testset "High-order GPU excitation kernel" begin
    for precision in ("f32", "f64"), ncoils in (1, 3, 32), off in (false, true)
        source = high_order_workload(; nspins=259, ncoils, rf=true)
        reference = convert_workload(source, precision)
        device = convert_workload(source, precision, kernel_backend)
        method = BlochHighOrder("1111", off, true)
        HighOrderMRISim.run_spin_excitation!(
            reference.obj, reference.hoseqd, reference.signal, reference.state, method,
        )
        workspace = HighOrderMRISim.HighOrderGPUWorkspace(device.obj, 32, kernel_backend)
        HighOrderMRISim.run_high_order_excitation!(
            workspace, device.obj, device.hoseqd, device.signal, device.state, method,
        )
        CUDA.synchronize()
        tolerance = precision == "f32" ? 3e-5 : 1e-11
        @test Array(device.signal) ≈ reference.signal atol=tolerance rtol=tolerance
        @test Array(device.state.xy) ≈ reference.state.xy atol=tolerance rtol=tolerance
        @test Array(device.state.z) ≈ reference.state.z atol=tolerance rtol=tolerance
    end
end

println("GPU_EXCITATION_KERNEL_TESTS_OK")

@testset "Packed GEMM output and strided signal views" begin
    for precision in ("f32", "f64"), nsteps in (1, 15, 32)
        source = high_order_workload(; nspins=259, nsteps, ncoils=3)
        reference = convert_workload(source, precision)
        device = convert_workload(source, precision, kernel_backend)
        parent_signal = CUDA.zeros(eltype(device.signal), 100, 3, 1)
        output = view(parent_signal, 17:16+nsteps, :, 1)
        workspace = HighOrderMRISim.HighOrderGPUWorkspace(device.obj, 32, kernel_backend)
        packed = reshape(view(workspace.coil_signal, 1:length(output)), size(output))
        @test packed isa CUDA.CuArray
        method = BlochHighOrder("1111")
        HighOrderMRISim.run_spin_precession!(
            reference.obj, reference.hoseqd, reference.signal, reference.state, method,
        )
        HighOrderMRISim.run_high_order_precession!(
            workspace, device.obj, device.hoseqd, output, device.state, method,
        )
        CUDA.synchronize()
        tolerance = precision == "f32" ? 3e-5 : 1e-11
        @test Array(output) ≈ reference.signal atol=tolerance rtol=tolerance
        @test all(iszero, Array(parent_signal[1:16, :, :]))
        @test all(iszero, Array(parent_signal[17+nsteps:end, :, :]))
    end
end

println("GPU_PACKED_SIGNAL_TESTS_OK")
