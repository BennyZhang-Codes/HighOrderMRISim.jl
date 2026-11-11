using Test
using HighOrderMRISim

isdefined(@__MODULE__, :compatibility_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

@testset "Phantom interfaces and simulation input validation" begin
    for T in (Float32, Float64)
        @testset "$T" begin
            empty = HO_Phantom(x=T[])
            @test iterate(empty) === nothing
            @test isempty(collect(empty))
            obj = HO_Phantom(x=T[0, 0.01, -0.02])
            @test length(collect(obj)) == 3
            @test [only(spin.x) for spin in obj] == obj.x
            @test iterate(obj, 4) === nothing

            params = Dict{String,Any}("precision" => T === Float32 ? "f32" : "f64",
                "Nthreads" => 1, "return_type" => "mat")
            hoseq = HO_Sequence(compatibility_sequence())
            for name in (:x, :y, :z, :ρ, :T1, :T2, :T2s, :Δw, :Dλ1, :Dλ2, :Dθ)
                malformed = deepcopy(obj)
                setproperty!(malformed, name, zeros(T, 2))
                for method in (Bloch(), BlochHighOrder("1111"))
                    settings = merge(params, Dict("sim_method" => method))
                    @test_throws DimensionMismatch simulate(malformed, hoseq, Scanner();
                        sim_params=settings, array_type=Array, verbose=false)
                end
                # Invalid shapes must fail before device selection or GPU allocation.
                @test_throws DimensionMismatch prepare_simulation(malformed, hoseq, Scanner();
                    sim_params=merge(params, Dict("sim_method" => BlochHighOrder("1111"))),
                    array_type=HighOrderMRISim.CUDA.CuArray, gpus=[0])
            end

            for axis in (:ux, :uy, :uz)
                moving = deepcopy(obj)
                displacement = (x, y, z, t) -> t
                setproperty!(moving, axis, displacement)
                @test_throws ArgumentError obj + moving
                @test_throws ArgumentError moving + obj
                joined = moving + moving
                @test getproperty(joined, axis) === displacement
                @test joined.ρ == vcat(moving.ρ, moving.ρ)
                @test getproperty(empty + moving, axis) === displacement
                @test getproperty(moving + empty, axis) === displacement
            end
        end
    end
end

@testset "Phantom metadata and receive-coil selection" begin
    mktempdir() do directory
        data = reshape(UInt8.(1:24), 2, 3, 4)
        source = joinpath(directory, "phantom.mat")
        MAT.matwrite(source, Dict("data" => data))
        phantom = BrainPhantom(x=1.0, y=2.0, z=3.0, matpath=source)
        for T in (Float32, Float64)
            result = load_phantom_mat(phantom, (T(1e-3), T(2e-3), T(3e-3)),
                                      (2, 3, 4), (zero(T), zero(T), zero(T)))
            @test result == data
            spacing = (T(1e-3), T(2e-3), T(3e-3))
            for sign in (-1, 1)
                edge = load_phantom_mat(phantom, spacing, (1, 1, 1),
                    (T(sign*1e-3), zero(T), zero(T)))
                outside = load_phantom_mat(phantom, spacing, (1, 1, 1),
                    (T(sign*1.1e-3), zero(T), zero(T)))
                @test only(edge) != 0
                @test only(outside) == 0
            end
        end
        invalid = BrainPhantom(x=0.0, matpath=source)
        @test_throws ArgumentError load_phantom_mat(invalid, (1e-3, 1e-3, 1e-3),
                                                  (2, 3, 4), (0.0, 0.0, 0.0))
    end
    for T in (Float32, Float64)
        spacing = (T(4e-3), T(4e-3), T(4e-3))
        location = (zero(T), zero(T), zero(T))
        full = load_csm(:real_32cha, 32, spacing, (2, 3, 1), location)
        subset = load_csm(:real_32cha, 4, spacing, (2, 3, 1), location)
        @test size(subset) == (2, 3, 1, 4)
        @test eltype(subset) == Complex{T}
        @test subset == full[:, :, :, 1:4]
        @test_throws ArgumentError load_csm(:real_32cha, 0, spacing, (2, 3, 1), location)
        @test_throws ArgumentError load_csm(:real_32cha, 33, spacing, (2, 3, 1), location)
    end
end

@testset "High-order mask consistency" begin
    for bits in 0:15
        name = string(bits; base=2, pad=4)
        flags = Tuple(==('1').(collect(name)))
        for method in (BlochHighOrder(name), BlochHighOrder(name, false, true))
            @test (method.ho0, method.ho1, method.ho2, method.ho3) == flags
            @test method.name == name
        end
    end
    for name in ("", "000", "00000", "10x1", "111₂")
        @test_throws ArgumentError BlochHighOrder(name)
        @test_throws ArgumentError BlochHighOrder(name, false, false)
    end
    @test_throws ArgumentError BlochHighOrder(ho0=false, name="1111")
    @test BlochHighOrder(ho0=false).name == "0111"
end

@testset "Sequence storage and transmit-channel validation" begin
    native = compatibility_sequence()
    obj = compatibility_phantom()
    for field in (:GR_dfc, :DUR)
        invalid = HO_Sequence(native)
        if field == :GR_dfc
            invalid.GR_dfc = invalid.GR_dfc[1:9, :]
        else
            invalid.DUR = invalid.DUR[1:1]
        end
        @test_throws DimensionMismatch discretize(invalid)
        @test_throws DimensionMismatch simulate(obj, invalid, Scanner(); array_type=Array,
            sim_params=Dict("sim_method" => BlochHighOrder("1111")), verbose=false)
    end
    multichannel = HO_Sequence(Sequence(native.GR, vcat(native.RF, native.RF),
                                       native.ADC, native.DUR))
    @test_throws ArgumentError discretize(multichannel)
    @test_throws ArgumentError simulate(obj, multichannel, Scanner(); array_type=Array,
        sim_params=Dict("sim_method" => BlochHighOrder("1111")), verbose=false)
end
