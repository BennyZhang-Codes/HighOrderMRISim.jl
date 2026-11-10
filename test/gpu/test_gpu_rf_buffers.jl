# RF-aware workspace allocation and reuse, including acquisition during RF.
using Test, CUDA, HighOrderMRISim, LinearAlgebra
import KomaMRICore
isdefined(@__MODULE__, :high_order_workload) ||
    include(joinpath(@__DIR__, "..", "support", "high_order_workload.jl"))
isdefined(@__MODULE__, :rf_acquisition_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

function check_rf_workspace_reuse(precision, mode, max_rf_adc, ncoils)
    backend = KomaMRICore.get_backend(true)
    source = high_order_workload(; nspins=259, ncoils)
    low, retained, reference = (convert_workload(source, precision, backend) for _ in 1:3)
    compact = HighOrderMRISim.HighOrderGPUWorkspace(low.obj, 32, backend;
        coil_combination=mode, max_rf_adc)
    previous = HighOrderMRISim.HighOrderGPUWorkspace(retained.obj, 32, backend;
        coil_combination=mode) # Conservative pre-optimization capacity.
    gemm = HighOrderMRISim.HighOrderGPUWorkspace(reference.obj, 32, backend; max_rf_adc=0)
    states, packed, partials = compact.adc_states, compact.coil_signal, compact.fusion.partials
    @test size(states) == (259, max_rf_adc)
    @test length(packed) == max_rf_adc*ncoils
    @test size(partials, 2) == 32*ncoils
    @test size(gemm.adc_states, 2) == 32
    for (rf, nadc) in ((true, max_rf_adc), (false, 32), (true, 0),
                       (false, 7), (true, min(1, max_rf_adc)), (false, 0), (true, max_rf_adc))
        block = high_order_workload(; nspins=259, ncoils, rf)
        fill!(block.hoseqd.seqd.ADC, false)
        if rf && nadc > 0
            # Acquire at the final RF node as well as interior nodes. A single
            # RF sample is deliberately at the end, not the first time step.
            nodes = nadc == 1 ? [33] : round.(Int, range(2, 33; length=nadc))
            block.hoseqd.seqd.ADC[nodes] .= true
        else
            block.hoseqd.seqd.ADC[2:nadc+1] .= true
        end
        convert_precision = precision == "f32" ? f32 : f64
        sampled = gpu(convert_precision(block.hoseqd), backend)
        T = eltype(low.signal)
        parent = CUDA.fill(T(7, -3), nadc+4, ncoils, 1)
        signal = view(parent, 3:nadc+2, :, 1)
        old_signal, gemm_signal = CUDA.zeros(T, nadc, ncoils), CUDA.zeros(T, nadc, ncoils)
        operation = rf ? HighOrderMRISim.run_high_order_excitation! : HighOrderMRISim.run_high_order_precession!
        for (workspace, w, output) in ((compact, low, signal), (previous, retained, old_signal),
                                       (gemm, reference, gemm_signal))
            operation(workspace, w.obj, sampled, output, w.state, BlochHighOrder("1111"))
        end
        CUDA.synchronize()
        tolerance = precision == "f32" ? 3e-5 : 1e-11
        @test Array(signal) == Array(old_signal)
        @test Array(signal) ≈ Array(gemm_signal) atol=tolerance rtol=tolerance
        @test Array(low.state.xy) == Array(retained.state.xy) == Array(reference.state.xy)
        @test Array(low.state.z) == Array(retained.state.z) == Array(reference.state.z)
        @test all(==(T(7, -3)), Array(parent)[[1, 2, nadc+3, nadc+4], :, :])
        @test compact.adc_states === states && compact.coil_signal === packed && compact.fusion.partials === partials
    end
    return nothing
end

function check_public_rf_buffers(precision, rf_limit, scenario)
    nrf_first, nrf_last, npre, rf_on = scenario
    native = rf_acquisition_sequence(nrf_first, nrf_last, npre; rf_on)
    hoseq = HO_Sequence(native)
    obj = HO_Phantom(x=[-0.01, 0.0, 0.01], y=[0.02, -0.01, 0.0],
        ρ=[1.0, 0.7, 0.2], T1=[0.9, 1.0, 1.1], T2=[0.07, 0.08, 0.09], Δw=[-20.0, 5.0, 25.0],
        csm=ComplexF64[1 im 0.5; 0.5-0.2im -0.3+0.1im 0.2im; 0.7 0.2im 0.4-0.3im])
    params = Dict{String,Any}("gpu" => true, "gpu_device" => 0, "precision" => precision,
        "implementation" => "kernel", "sim_method" => BlochHighOrder("1111"),
        "return_type" => "mat", "max_block_length" => 32, "max_rf_block_length" => rf_limit)
    reports = Dict{String,Any}()
    function capture(_, _, _, actual)
        get!(reports, actual["coil_combination"]) do
            (maximum=actual["max_adc_per_block"], rf=actual["max_rf_adc_per_block"],
             capacity=actual["adc_state_buffer_samples"])
        end
        return nothing
    end
    reference = simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false, callbacks=(capture,))
    cpu = simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("gpu" => false, "implementation" => "array")))
    array = simulate(obj, hoseq, Scanner(); verbose=false,
        sim_params=merge(params, Dict("implementation" => "array")))
    tolerance = precision == "f32" ? 3e-5 : 1e-11
    @test reference ≈ cpu atol=tolerance rtol=tolerance
    @test array ≈ cpu atol=tolerance rtol=tolerance
    @test size(reference) == (nrf_first+nrf_last+npre, 3, 1)
    expected_rf = rf_on ? min(max(nrf_first, nrf_last), rf_limit == Inf ? typemax(Int) : rf_limit) : 0
    @test reports["gemm"].rf == expected_rf
    @test reports["gemm"].capacity == reports["gemm"].maximum
    state_params = merge(params, Dict("return_type" => "state"))
    reference_state = simulate(obj, hoseq, Scanner(); sim_params=state_params, verbose=false)
    for mode in ("fused_warp", "fused_tiled")
        candidate_params = merge(params, Dict("coil_combination" => mode))
        owned = copy(candidate_params)
        signal = simulate(obj, hoseq, Scanner(); sim_params=candidate_params, verbose=false, callbacks=(capture,))
        @test signal ≈ reference atol=tolerance rtol=tolerance
        @test all(isfinite, signal)
        @test reports[mode].rf == expected_rf
        @test reports[mode].capacity == expected_rf
        @test reports[mode].maximum == reports["gemm"].maximum
        @test candidate_params == owned
        state = simulate(obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(candidate_params, Dict("return_type" => "state")))
        @test state.xy == reference_state.xy
        @test state.z == reference_state.z
        if expected_rf > 0
            raw = simulate(obj, hoseq, Scanner(); verbose=false,
                sim_params=merge(candidate_params, Dict("return_type" => "raw")))
            @test sum(size(profile.data, 1) for profile in raw.profiles) == nrf_first+nrf_last+npre
            @test all(size(profile.data, 2) == 3 for profile in raw.profiles)
        end
    end
    return nothing
end

function test_gpu_rf_buffers()
    length(collect(CUDA.devices())) == 1 || error("Expose one GPU by UUID.")
    CUDA.device!(0)
    CUDA.allowscalar(false)
    @testset "RF-aware workspace reuse" begin
        for precision in ("f32", "f64"), mode in ("fused_warp", "fused_tiled"),
            max_rf_adc in (0, 3), ncoils in (3, 32)
            check_rf_workspace_reuse(precision, mode, max_rf_adc, ncoils)
        end
    end
    @testset "RF acquisition through public simulate" begin
        for precision in ("f32", "f64"), rf_limit in (1, Inf),
            scenario in ((0, 0, 11, true), (3, 2, 11, true), (5, 2, 0, true),
                         (0, 0, 11, false), (0, 0, 0, true))
            check_public_rf_buffers(precision, rf_limit, scenario)
        end
    end
    println("GPU_RF_BUFFER_TESTS_OK")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_gpu_rf_buffers()
end
