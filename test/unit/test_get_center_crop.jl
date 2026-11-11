using Test
using HighOrderMRISim

@testset "get_center_crop tests" begin
    # Test cropping a 2D image
    img2d = rand(100, 100)
    cropped2d = get_center_crop(img2d, 50, 50)
    @test size(cropped2d) == (50, 50)

    # Test padding when output size is larger
    padded2d = get_center_crop(img2d, 120, 120)
    @test size(padded2d) == (120, 120)
    @test sum(padded2d .== 0) > 0   # check padding occurred

    # Test cropping a 3D image
    img3d = rand(100, 100, 10)
    cropped3d = get_center_crop(img3d, 80, 80)
    @test size(cropped3d) == (80, 80, 10)

    # Test padding in 3D
    padded3d = get_center_crop(img3d, 120, 120)
    @test size(padded3d) == (120, 120, 10)
end

@testset "Center crop preserves values on equal axes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        image = reshape(T.(1:20), 4, 5)
        @test get_center_crop(image, 4, 5) == image
        @test get_center_crop(image, 2, 5) == image[2:3, :]
        @test get_center_crop(image, 4, 3) == image[:, 2:4]
        padded = get_center_crop(image, 6, 5)
        @test padded[2:5, :] == image
        @test iszero(padded[[1, 6], :])
        padded = get_center_crop(image, 4, 7)
        @test padded[:, 2:6] == image
        @test iszero(padded[:, [1, 7]])
        volume = cat(image, 2image; dims=3)
        @test get_center_crop(volume, 4, 5) == volume
        @test get_center_crop(volume, 2, 5) == volume[2:3, :, :]
        @test get_center_crop(volume, 4, 3) == volume[:, 2:4, :]
        @test get_center_crop(volume, 4, 5, 2) == volume
        @test get_center_crop(image, 4, 5, 1) == reshape(image, 4, 5, 1)
    end
end

