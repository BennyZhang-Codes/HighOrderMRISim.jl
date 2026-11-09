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

