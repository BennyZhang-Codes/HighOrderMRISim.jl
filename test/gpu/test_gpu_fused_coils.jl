using Test, CUDA, HighOrderMRISim, LinearAlgebra
import KomaMRICore
isdefined(@__MODULE__, :high_order_workload) ||
    include(joinpath(@__DIR__, "..", "support", "high_order_workload.jl"))

function check_fused_coils(precision, mode; nspins=259, ncoils=3, nadc=7,
                           orders="1111", rf=false, groupsize=256,
                           adc_tile=8, coil_tile=1)
    source = high_order_workload(; nspins, ncoils, rf)
    fill!(source.hoseqd.seqd.ADC, false)
    # Irregular ADC spacing, including a final node without acquisition.
    indices = round.(Int, range(2, nadc == 32 ? 33 : 32; length=max(nadc, 2)))[1:nadc]
    source.hoseqd.seqd.ADC[indices] .= true
    source = merge(source, (; signal=zeros(ComplexF64, nadc, ncoils)))
    backend = KomaMRICore.get_backend(true)
    reference, candidate = (convert_workload(source, precision, backend) for _ in 1:2)
    parent = CUDA.fill(eltype(candidate.signal)(7, -3), nadc+4, ncoils, 1)
    candidate = merge(candidate, (; signal=view(parent, 3:nadc+2, :, 1)))
    operation = rf ? HighOrderMRISim.run_high_order_excitation! : HighOrderMRISim.run_high_order_precession!
    for (w, strategy) in ((reference, "gemm"), (candidate, mode))
        workspace = HighOrderMRISim.HighOrderGPUWorkspace(w.obj, 32, backend;
            coil_combination=strategy, groupsize, adc_tile, coil_tile,
            max_rf_adc=rf ? nadc : 0)
        capacity = strategy == "gemm" ? 32 : rf ? nadc : 0
        @test size(workspace.adc_states) == (nspins, capacity)
        @test length(workspace.coil_signal) == capacity*ncoils
        operation(workspace, w.obj, w.hoseqd, w.signal, w.state, BlochHighOrder(orders))
    end
    CUDA.synchronize()
    tolerance = precision == "f32" ? 3e-5 : 1e-11
    @test Array(candidate.signal) ≈ Array(reference.signal) atol=tolerance rtol=tolerance
    @test Array(candidate.state.xy) == Array(reference.state.xy)
    @test Array(candidate.state.z) == Array(reference.state.z)
    @test Array(candidate.obj.ρ) == Array(reference.obj.ρ)
    @test all(==(eltype(parent)(7, -3)), Array(parent)[[1, 2, nadc+3, nadc+4], :, :])
    return nothing
end

function test_fused_coils(modes=("fused_warp", "fused_tiled"))
    length(collect(CUDA.devices())) == 1 || error("Expose one GPU by UUID.")
    CUDA.device!(0)
    CUDA.allowscalar(false)
    println("Julia ", VERSION, " / ", CUDA.name(CUDA.device()), " / ", CUDA.uuid(CUDA.device()))
    flush(stdout)
    @testset "Fused CUDA coil reduction" begin
        for mode in modes, precision in ("f32", "f64"), ncoils in (1, 3, 32),
            nadc in (0, 1, 7, 15, 31, 32)
            check_fused_coils(precision, mode; ncoils, nadc)
        end
        for mode in modes, precision in ("f32", "f64"), mask in 0:15
            check_fused_coils(precision, mode; orders=lpad(string(mask; base=2), 4, '0'))
        end
        for mode in modes, precision in ("f32", "f64"), ncoils in (1, 3, 32)
            check_fused_coils(precision, mode; ncoils, rf=true)
        end
        for mode in modes, precision in ("f32", "f64"), nspins in (1, 31, 32, 33, 255, 256, 257),
            groupsize in (64, 128, 256)
            check_fused_coils(precision, mode; nspins, groupsize)
        end
        for precision in ("f32", "f64"), adc_tile in (2, 4, 8), coil_tile in (1, 2, 4)
            check_fused_coils(precision, "fused_tiled"; nspins=2051, ncoils=33, adc_tile, coil_tile)
        end
        w = convert_workload(high_order_workload(; nspins=1), "f64", KomaMRICore.get_backend(true))
        @test_throws ArgumentError HighOrderMRISim.HighOrderGPUWorkspace(w.obj, 32,
            KomaMRICore.get_backend(true); coil_combination="fused_warp", groupsize=48)
        for (adc_tile, coil_tile) in ((3, 1), (4.0, 1), (4, 0))
            @test_throws ArgumentError HighOrderMRISim.HighOrderGPUWorkspace(w.obj, 32,
                KomaMRICore.get_backend(true); coil_combination="fused_tiled", adc_tile, coil_tile)
        end
        @test_throws ArgumentError HighOrderMRISim.HighOrderGPUWorkspace(w.obj, 32,
            KomaMRICore.get_backend(true); coil_combination="fused_tiled", groupsize=1024)
        for max_rf_adc in (-1, 33, 2.5)
            @test_throws ArgumentError HighOrderMRISim.HighOrderGPUWorkspace(w.obj, 32,
                KomaMRICore.get_backend(true); coil_combination="fused_tiled", max_rf_adc)
        end
    end
    println("FUSED_COIL_TESTS_OK")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_fused_coils()
end
