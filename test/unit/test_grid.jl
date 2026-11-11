using Test
using HighOrderMRISim

@testset "Voxel-center grid geometry" begin
    for T in (Float32, Float64)
        fov = T[0.2, 0.3, 0.004]
        dims = [2, 3, 1]
        xyz = generate_grid(fov, dims)
        @test size(xyz) == (2, 3, 1, 3)
        @test eltype(xyz) == Float64
        @test xyz[:, 1, 1, 1] ≈ [-1, 1] .* Float64(fov[1]) ./ 4
        @test xyz[1, :, 1, 2] ≈ [-1, 0, 1] .* Float64(fov[2]) ./ 3
        @test iszero(xyz[:, :, :, 3])
        @test generate_grid(view(fov, :), view(dims, :)) == xyz
        spacing = T.(fov ./ dims)
        grid = Grid(dims..., spacing...)
        for (axis, values) in enumerate((grid.x, grid.y, grid.z))
            @test eltype(values) == T
            @test values ≈ vec(xyz[:, :, :, axis]) atol=eps(T)
        end
        @test_throws ArgumentError Grid(0, 3, 1, spacing...)
        @test_throws ArgumentError Grid(2, 3, 1, zero(T), spacing[2], spacing[3])
        @test_throws ArgumentError Grid(2, 3, 1, T(Inf), spacing[2], spacing[3])
    end
    for fov in ([0.0, 0.2, 0.2], [-1.0, 0.2, 0.2], [Inf, 0.2, 0.2], [NaN, 0.2, 0.2])
        @test_throws ArgumentError generate_grid(fov, [2, 3, 1])
    end
    for dims in ([0, 3, 1], [-1, 3, 1], [1.5, 3, 1], [Inf, 3, 1], [NaN, 3, 1])
        @test_throws ArgumentError generate_grid([0.2, 0.3, 0.004], dims)
    end
    @test_throws ArgumentError generate_grid([0.2, 0.3], [2, 3, 1])
end
