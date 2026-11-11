using Test
using HighOrderMRISim

@testset "Trajectory integration, differentiation and interpolation" begin
    for T in (Float32, Float64)
        dt = T(0.25)
        values = reshape(T.(0:11), 4, 3)
        @test traj2grad(values, dt) == diff(values; dims=1) ./ dt
        expected = vcat(zeros(T, 1, 3), cumsum((values[1:3, :]+values[2:4, :]) .* (dt/2); dims=1))
        @test grad2traj(values, dt) == expected
        @test grad2traj(transpose(values), dt; dim=2) == transpose(expected)
        @test size(grad2traj(zeros(T, 0, 3), dt)) == (0, 3)
        @test size(traj2grad(zeros(T, 0, 3), dt)) == (0, 3)
        @test size(traj2grad(ones(T, 1, 3), dt)) == (0, 3)
        for operation in (grad2traj, traj2grad)
            for invalid_dt in (0, -1, Inf, NaN)
                @test_throws ArgumentError operation(values, invalid_dt)
            end
            for dim in (0, 3)
                @test_throws ArgumentError operation(values, dt; dim)
            end
        end
        # Coordinate precision need not match the precision of sample times.
        times = Float64.(0:3) .* dt
        interpolated = InterpTrajTime(values, dt, 0.0, times)
        @test interpolated ≈ values
        @test eltype(interpolated) == T
        @test InterpTrajTime(values, dt, 0.0) ≈ values
        @test size(InterpTrajTime(zeros(T, 4, 0), dt, 0.0, times)) == (4, 0)
        @test size(InterpTrajTime(values, dt, 0.0, Float64[])) == (0, 3)
        for invalid_dt in (0, -1, Inf, NaN)
            @test_throws ArgumentError InterpTrajTime(values, invalid_dt, 0.0, times)
        end
        @test_throws ArgumentError InterpTrajTime(values, dt, Inf, times)
        @test_throws ArgumentError InterpTrajTime(values, dt, 0.0, [0.0, NaN])
        shots = cat(reshape(values, 1, 4, 3), reshape(2values, 1, 4, 3); dims=1)
        sampled = InterpTrajTime(shots, dt, [0.0, 0.0], repeat(times', 2, 1))
        @test sampled ≈ shots
        @test eltype(sampled) == T
    end
end
