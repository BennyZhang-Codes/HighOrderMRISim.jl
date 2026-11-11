using Test
using HighOrderMRISim

isdefined(@__MODULE__, :compatibility_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

@testset "Phantom construction and default receive coil" begin
    @testset "$T" for T in (Float32, Float64)
        @testset "Omitted CSM: $n spins" for n in (0, 1, 3)
            x = zeros(T, n)
            for constructor in (HO_Phantom, HO_Phantom{T})
                obj = constructor(; x)
                @test obj isa HO_Phantom{T}
                @test obj.x === x
                @test size(obj.csm) == (n, 1)
                @test eltype(obj.csm) == Complex{T}
                @test all(isone, obj.csm)
                for name in (:y, :z, :Δw, :Dλ1, :Dλ2, :Dθ)
                    @test getproperty(obj, name) == zeros(T, n)
                    @test eltype(getproperty(obj, name)) == T
                end
                @test obj.ρ == ones(T, n)
                @test eltype(obj.ρ) == T
                for name in (:T1, :T2, :T2s)
                    @test getproperty(obj, name) == fill(T(1_000_000), n)
                    @test eltype(getproperty(obj, name)) == T
                end
                other = constructor(; x)
                @test obj.csm !== other.csm
                if n > 0
                    obj.csm[1, 1] = zero(Complex{T})
                    @test all(isone, other.csm)
                end
            end
        end

        @testset "Explicit CSM values and storage" begin
            x = T[0, 0.01]
            supplied = Complex{T}[1+2im -3im; -0.5+0.25im 2]
            states = Complex{T}[1+im 0.5; -2im 3-im]
            for csm in (supplied, view(supplied, :, 2:-1:1), zeros(Complex{T}, 2, 2))
                original = copy(csm)
                for constructor in (HO_Phantom, HO_Phantom{T})
                    obj = constructor(; x, csm)
                    @test obj.csm === csm
                    @test obj.csm == original
                    @test size(obj.csm) == (2, 2)
                    signal = zeros(Complex{T}, 2, 2)
                    HighOrderMRISim.combine_array_signal!(signal, states, obj.csm)
                    expected = [sum(states[:, sample] .* original[:, coil])
                        for sample in 1:2, coil in 1:2]
                    @test signal ≈ expected
                    @test obj.csm == original
                end
            end
        end

        @testset "Explicit type keeps constructor conversion behavior" begin
            S = T === Float32 ? Float64 : Float32
            x = S[0, 0.01]
            csm = reshape(Complex{S}[1+2im, -0.5+0.25im], :, 1)
            original = copy(csm)
            obj = HO_Phantom{T}(; x, csm)
            @test obj isa HO_Phantom{T}
            @test obj.x == T.(x)
            @test eltype(obj.csm) == Complex{T}
            @test obj.csm == Complex{T}.(original)
            @test csm == original
            implicit = HO_Phantom{T}(; x)
            @test eltype(implicit.csm) == Complex{T}
            @test implicit.csm == ones(Complex{T}, 2, 1)

            # Narrow input coordinates must not overflow a wider phantom's
            # finite relaxation defaults before constructor conversion.
            narrow = HO_Phantom{T}(x=Float16[0], csm=ones(Complex{T}, 1, 1))
            for name in (:T1, :T2, :T2s)
                @test only(getproperty(narrow, name)) == T(1_000_000)
            end
        end

        @testset "Public simulation with an ideal receive coil" begin
            inputs = (; x=T[0, 0.01], ρ=T[1, 0.7], T1=fill(T(Inf), 2), T2=fill(T(Inf), 2))
            implicit = HO_Phantom(; inputs...)
            explicit = HO_Phantom(; inputs..., csm=ones(Complex{T}, 2, 1))
            params = Dict{String,Any}(
                "precision" => T === Float32 ? "f32" : "f64",
                "Nthreads" => 1, "sim_method" => BlochHighOrder("1111"),
                "return_type" => "mat",
            )
            hoseq = HO_Sequence(compatibility_sequence())
            signal = simulate(implicit, hoseq, Scanner(); sim_params=params, array_type=Array, verbose=false)
            reference = simulate(explicit, hoseq, Scanner(); sim_params=params, array_type=Array, verbose=false)
            @test eltype(signal) == Complex{T}
            @test size(signal) == (5, 1, 1)
            @test signal == reference
            @test all(isfinite, signal)
            @test all(isapprox.(abs.(signal), sum(inputs.ρ); rtol=20eps(T)))
            @test all(isone, implicit.csm)

            # Explicit zero/complex maps remain zero/complex; no normalization,
            # conjugation, or fallback to the default coil is applied.
            weights = Complex{T}[0 2+im; 0 2+im]
            weighted = HO_Phantom(; inputs..., csm=weights)
            result = simulate(weighted, hoseq, Scanner(); sim_params=params, array_type=Array, verbose=false)
            @test size(result) == (5, 2, 1)
            @test iszero(result[:, 1, 1])
            @test result[:, 2, 1] ≈ Complex{T}(2+im) .* signal[:, 1, 1] rtol=20eps(T)
            @test weighted.csm === weights
            @test weights == Complex{T}[0 2+im; 0 2+im]
        end
    end
end
