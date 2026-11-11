using Test
using HighOrderMRISim
using LinearAlgebra

@testset "Image normalization and legacy metrics" begin
    for T in (Float32, Float64)
        constant = fill(T(3), 3, 3)
        @test normalization(constant) == zeros(T, 3, 3)
        @test standardization(constant) == zeros(T, 3, 3)
        @test eltype(normalization(constant)) == T
        @test eltype(standardization(constant)) == T
        @test HO_MSE(constant, constant) == 0
        @test HO_RMSE(constant, constant) == 0
        @test_throws DomainError HO_NRMSE(constant, constant)
        @test_throws DomainError HO_NRMSE(zeros(T, 3, 3), zeros(T, 3, 3))
        reference = reshape(T.(0:8), 3, 3)
        image = reference .+ T(0.5)
        @test normalization(reference) ≈ reference ./ T(8)
        @test HO_NRMSE(reference, image; scale=false) ≈
            norm(reference-image) / sqrt(length(reference)) / T(8)
        @test HO_NRMSE(reference, 2reference) ≈ 0 atol=10eps(T)
        @test HO_MSE(reference, image; scale=false) ≈ T(0.25)
        @test HO_RMSE(reference, image; scale=false) ≈ T(0.5)
        @test HO_img_scale(reference, zeros(T, 3, 3)) == zeros(T, 3, 3)
        for metric in (HO_img_scale, HO_NRMSE, HO_MSE, HO_RMSE, HO_SSIM)
            @test_throws DimensionMismatch metric(reference, zeros(T, 9))
            @test_throws ArgumentError metric(T[], T[])
            @test_throws DomainError metric(reference, fill(T(NaN), 3, 3))
        end
        for transform in (normalization, standardization)
            @test_throws ArgumentError transform(zeros(T, 0, 0))
            @test_throws DomainError transform(fill(T(Inf), 2, 2))
        end
    end
end
