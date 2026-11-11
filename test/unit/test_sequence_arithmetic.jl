using Test
using HighOrderMRISim

isdefined(@__MODULE__, :compatibility_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

@testset "High-order sequence arithmetic" begin
    @testset "$T" for T in (Float32, Float64)
        native = compatibility_sequence()
        gradients = Grad[
            block == 1 ? Grad(zero(T), 0.0) : Grad(T(axis * 1e-3), 1e-3)
            for axis in 1:3, block in 1:2
        ]
        seq = Sequence(gradients, native.RF, native.ADC, native.DUR)
        fields = Grad[
            block == 1 ? Grad(T(term * 1e-6), seq.DUR[block]) :
            Grad(T[0, 1, -0.25, 0] .* T(term * 1e-5), 1e-3)
            for term in 1:16, block in 1:2
        ]
        x = HO_Sequence(seq, fields)
        y = T(1.5) * x
        original_x, original_y = deepcopy(x), deepcopy(y)

        @testset "Subtraction preserves native concatenation semantics" begin
            actual, expected = x - y, x + (-y)
            @test actual.SEQ ≈ x.SEQ - y.SEQ
            @test actual.SEQ ≈ expected.SEQ
            @test actual.DUR == expected.DUR == [x.DUR; y.DUR]
            @test all(isapprox.(actual.GR_dfc, expected.GR_dfc))
            @test all(isapprox.(actual.SEQ.RF, [x.SEQ.RF y.SEQ.RF]))
            @test all(isapprox.(actual.SEQ.ADC, [x.SEQ.ADC; y.SEQ.ADC]))
            # Sampling may promote amplitudes; compare at the source precision.
            sampled, reference = get_samples(actual), get_samples(expected)
            for name in propertynames(reference)
                @test sampled[name].t == reference[name].t
                @test sampled[name].A ≈ reference[name].A rtol=10eps(T)
            end
        end

        @testset "Division scales all field channels: $α" for α in T.((2, -3))
            actual, expected = x / α, inv(α) * x
            @test actual.SEQ ≈ x.SEQ / α
            @test actual.SEQ ≈ expected.SEQ
            @test actual.DUR == expected.DUR == x.DUR
            @test all(isapprox.(actual.SEQ.RF, x.SEQ.RF))
            @test all(isapprox.(actual.SEQ.ADC, x.SEQ.ADC))
            for i in eachindex(x.GR_dfc)
                @test actual.GR_dfc[i] ≈ expected.GR_dfc[i]
                @test actual.GR_dfc[i].A ≈ x.GR_dfc[i].A / α
                @test typeof(actual.GR_dfc[i].A) == typeof(expected.GR_dfc[i].A)
            end
            sampled, reference = get_samples(actual), get_samples(expected)
            for name in propertynames(reference)
                @test sampled[name].t == reference[name].t
                @test sampled[name].A ≈ reference[name].A rtol=10eps(T)
            end
        end

        # Real scaling and subtraction affect fields, not the RF preparation.
        @testset "CPU signal parity: $orders" for orders in ("0000", "1111")
            params = Dict{String,Any}(
                "gpu" => false, "precision" => T === Float32 ? "f32" : "f64",
                "Nthreads" => 1, "sim_method" => BlochHighOrder(orders),
                "return_type" => "mat",
            )
            obj = compatibility_phantom()
            for (actual, expected) in ((x - y, x + (-y)), (x / T(2), T(0.5) * x))
                signal = simulate(obj, actual, Scanner(); sim_params=params, verbose=false)
                reference = simulate(obj, expected, Scanner(); sim_params=params, verbose=false)
                @test eltype(signal) == Complex{T}
                @test signal ≈ reference rtol=10eps(T) atol=10eps(T)
            end
        end

        @test x.SEQ ≈ original_x.SEQ
        @test y.SEQ ≈ original_y.SEQ
        @test get_samples(x) == get_samples(original_x)
        @test get_samples(y) == get_samples(original_y)
        @test x.DUR == original_x.DUR
        @test y.DUR == original_y.DUR
    end
end
