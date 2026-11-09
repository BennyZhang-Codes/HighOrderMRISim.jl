using Test
using HighOrderMRISim
import HighOrderMRISim: load_csm, _resample_csm

@testset "Coil-sensitivity maps" begin
    @testset "Bundled 2D/3D map shape" begin
        for dims in ((100, 100, 1), (100, 100, 10))
            csm = load_csm(:real_32cha, 32, (2e-3, 2e-3, 2e-3), dims, (0.0, 0.0, 0.0))
            @test csm isa Array{ComplexF64,4}
            @test size(csm) == (dims..., 32)
            @test all(isfinite, csm)
        end
    end

    @testset "Analytic cell boundaries: $T" for T in (Float32, Float64)
        spacing = T.((0.004, 0.006, 0.008))
        dims = (3, 4, 2)
        axes = ntuple(i -> ((0:dims[i]-1) .- (dims[i]-1)/2) .* spacing[i], 3)
        field(x, y, z) = Complex{T}[
            complex(2+100x-20y+30z, 0.5-80x+60y+25z),
            complex(0.7-40x+10y-90z, 1.2+35x-45y+70z)]
        source = Array{Complex{T}}(undef, dims..., 2)
        for index in CartesianIndices(dims)
            source[Tuple(index)..., :] .= field((axes[i][index[i]] for i in 1:3)...)
        end
        sample(point) = vec(_resample_csm(source, spacing, T.((0.001, 0.001, 0.001)),
            (1, 1, 1), point))
        faces = ntuple(i -> T(dims[i]*spacing[i]/2), 3)
        tolerance = 64eps(T)

        # A linear complex field has an independent exact interpolation oracle.
        for point in (T.((0.0, 0.0, 0.0)), T.((0.0011, -0.0023, 0.0009)))
            expected = field(point...)
            expected ./= sqrt(sum(abs.(expected).^2))
            @test sample(point) ≈ expected rtol=tolerance atol=tolerance
        end

        # Face/edge/corner cells are valid, with no continuation beyond a cell.
        for signs in Iterators.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1))
            point = ntuple(i -> T(signs[i])*faces[i], 3)
            clamped = ntuple(i -> clamp(point[i], first(axes[i]), last(axes[i])), 3)
            expected = field(clamped...)
            expected ./= sqrt(sum(abs.(expected).^2))
            actual = sample(point)
            @test actual ≈ expected rtol=tolerance atol=tolerance
            @test sum(abs2, actual) ≈ one(T) rtol=tolerance
            @test all(isfinite, actual)
        end
        for axis in 1:3, sign in (-1, 1)
            point = ntuple(i -> i == axis ? T(sign)*nextfloat(faces[i]) : zero(T), 3)
            @test all(iszero, sample(point))
        end

        zeros_map = _resample_csm(zeros(Complex{T}, dims..., 3), spacing,
            T.((0.002, 0.002, 0.002)), (5, 5, 5), T.((0.0, 0.0, 0.0)))
        @test eltype(zeros_map) === Complex{T}
        @test all(isfinite, zeros_map)
        @test all(iszero, zeros_map)
    end

    @testset "Previously blind real-map border" begin
        spacing = (1e-3, 1e-3, 1e-3)
        dims = (1, 9, 6)
        # Includes all 13 blind spins found at x=-88.5 mm in the old 3D grid.
        boundary = load_csm(:real_32cha, 32, spacing, dims, (-0.0885, -0.0045, -0.017))
        centre = load_csm(:real_32cha, 32, spacing, dims, (-0.088, -0.0045, -0.017))
        @test boundary == centre
        @test all(isfinite, boundary)
        @test all(isapprox.(sum(abs2, boundary; dims=4), 1; rtol=1e-12))
        outside = load_csm(:real_32cha, 32, spacing, (1, 1, 1), (-0.0901, 0.0, 0.0))
        @test all(iszero, outside)
        @test all(isfinite, outside)
        for T in (Float32, Float64)
            # The physical face is represented in the target coordinate precision.
            face = load_csm(:real_32cha, 32, T.(spacing), (1, 1, 1),
                T.((-0.09, -0.0045, -0.017)))
            beyond = load_csm(:real_32cha, 32, T.(spacing), (1, 1, 1),
                (-nextfloat(T(0.09)), T(-0.0045), T(-0.017)))
            @test eltype(face) === Complex{T}
            @test all(isfinite, face)
            @test sum(abs2, face) ≈ one(T) rtol=64eps(T)
            @test all(iszero, beyond)
        end
    end
end
