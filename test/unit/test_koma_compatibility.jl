using Test
using HighOrderMRISim
import KomaMRIBase, KomaMRICore

isdefined(@__MODULE__, :compatibility_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

@testset "KomaMRI compatibility" begin
    @testset "Parameterized Grad inputs and high-order-only sampling" begin
        gradients = reshape([Grad(Float64(i), 1e-3) for i in 1:4], :, 1)
        hoseq = HO_Sequence(gradients)
        @test [g.A for g in hoseq.SEQ.GR[:, 1]] == [2.0, 3.0, 4.0]
        @test size(hoseq.GR_dfc) == (16, 1)
        @test HO_Sequence(hoseq.SEQ, gradients) isa HO_Sequence

        # No nominal gradients: the field's own sample times must still be present.
        jump = HO_Sequence([Grad(1e-6, 1e-3) Grad(-2e-6, 1e-3)])
        original_durations = copy(jump.SEQ.DUR)
        sampled = discretize(jump; sampling_params=Dict("Δt" => 2e-4))
        @test jump.SEQ.DUR == original_durations
        @test last(sampled.seqd.t) ≈ 2e-3
        @test all(>=(0), sampled.seqd.Δt)
        @test any(iszero, sampled.seqd.Δt)
        @test trapz(sampled.seqd.Δt, sampled.h0) ≈ -1e-9 atol=1e-20
        @test length(sampled.seqd.t) == length(sampled.seqd.Δt) + 1
        @test length(sampled.seqd.excitation_bool) == length(sampled.seqd.Δt)
        for term in 0:15
            @test length(getfield(sampled, Symbol("h", term))) == length(sampled.seqd.t)
        end
        nodes = 1:3
        @test length(view(sampled, nodes)) == length(nodes) - 1
        @test length(view(sampled, nodes).h0) == length(nodes)
        @test sampled[nodes].h0 == sampled.h0[nodes]
    end

    @testset "ADC ownership and generalized encoding output" begin
        seq = compatibility_sequence()
        hoseq = HO_Sequence(seq)
        sampled = discretize(hoseq)
        parts, _ = KomaMRICore.get_sim_ranges(sampled.seqd; max_block_length=1)
        @test sum(count(view(sampled.seqd.ADC, first(p)+1:last(p))) for p in parts) == 5
        native_times = get_adc_sampling_times(seq)
        @test sampled.seqd.t[sampled.seqd.ADC] ≈ native_times
        nominal, nominal_adc, measured, measured_adc = get_kspace(hoseq)
        @test size(nominal, 2) == 3
        @test size(nominal_adc) == (5, 3)
        @test size(measured, 2) == 16
        @test size(measured_adc) == (5, 16)
        @test all(iszero, nominal_adc)
        @test all(iszero, measured_adc)
    end

    @testset "Legacy simulation API and state ownership" begin
        seq = compatibility_sequence()
        hoseq = HO_Sequence(seq)
        obj = compatibility_phantom()
        density = copy(obj.ρ)
        params = Dict{String,Any}(
            "gpu" => false, "precision" => "f64", "Nthreads" => 1,
            "sim_method" => BlochHighOrder("0000"), "return_type" => "mat",
            "Nblocks" => 1,
        )
        original_params = copy(params)
        signal = simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false)
        expected = im .* (transpose(density) * obj.csm)
        @test size(signal) == (5, 2, 1)
        @test signal[:, :, 1] ≈ repeat(expected, 5, 1) atol=1e-12
        @test obj.ρ == density
        @test params == original_params

        samples_seen = Int[]
        callback = (progress, _, _, _) -> push!(samples_seen, progress.samples - 1)
        split_params = merge(params, Dict("Nblocks" => 1000, "Nthreads" => 2))
        split_signal = simulate(
            obj, hoseq, Scanner(); sim_params=split_params, verbose=false,
            callbacks=(callback,),
        )
        @test split_signal ≈ signal atol=1e-12
        @test issorted(samples_seen)
        @test last(samples_seen) == 5
        single = simulate(
            obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(params, Dict("precision" => "f32")),
        )
        @test eltype(single) == ComplexF32
        @test single ≈ signal atol=2e-5 rtol=2e-5

        state = simulate(
            obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(params, Dict("return_type" => "state")),
        )
        @test state isa Mag
        @test obj.ρ == density
        @test state.z !== obj.ρ
        @test state.xy ≈ im .* density atol=1e-12

        raw = simulate(
            obj, hoseq, Scanner(); verbose=false,
            sim_params=merge(params, Dict("return_type" => "raw")),
        )
        @test raw isa RawAcquisitionData
        @test sum(size(profile.data, 1) for profile in raw.profiles) == 5
        @test all(size(profile.data, 2) == 2 for profile in raw.profiles)

        # The nominal-only path must continue to agree with native Koma Bloch.
        native = Phantom(x=copy(obj.x), ρ=copy(density), T1=fill(Inf, 2), T2=fill(Inf, 2))
        nominal_params = Dict{String,Any}(
            "gpu" => false, "precision" => "f64", "Nthreads" => 1,
            "return_type" => "mat", "sim_method" => Bloch(),
        )
        native_signal = simulate(native, seq, Scanner(); sim_params=nominal_params, verbose=false)
        legacy_signal = simulate(obj, hoseq, Scanner(); sim_params=nominal_params, verbose=false)
        @test legacy_signal ≈ native_signal atol=1e-12
    end

    @testset "Measured zeroth-order phase and legacy RF behavior" begin
        seq = compatibility_sequence()
        hoseq = HO_Sequence(seq)
        obj = compatibility_phantom()
        hoseq.GR_dfc[1, 2] = Grad(20 / γ, 1e-3)
        params = Dict{String,Any}(
            "gpu" => false, "precision" => "f64", "Nthreads" => 1,
            "sim_method" => BlochHighOrder("1000"), "return_type" => "mat",
        )
        signal = simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false)
        expected = im .* (transpose(obj.ρ) * obj.csm)
        readout_times = get_adc_sampling_times(seq) .- seq.DUR[1]
        @test signal[:, :, 1] ≈ cis.(-2π * 20 .* readout_times) .* expected atol=1e-12

        # Compatibility migration must not silently enable high-order fields during RF.
        hoseq.GR_dfc[1, 1] = Grad(30 / γ, seq.DUR[1])
        unchanged = simulate(obj, hoseq, Scanner(); sim_params=params, verbose=false)
        @test unchanged ≈ signal atol=1e-12
    end
end
