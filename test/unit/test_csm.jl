using Test
using HighOrderMRISim
import HighOrderMRISim: load_csm

type            = nothing
nCoil           = 32          
target_res      = ( 2.,  2., 2.).*1e-3;     # target resolution
target_matrix   = (100, 100, 10)      ;     # target matrix size
location        = ( 0.,  0., 0.).*1e-3;     # target location of the sampling center
verbose         = false 


@testset "csm" begin
    @testset "csm_Real_32cha 2D" begin
        target_matrix   = (100, 100, 1)
        csm = load_csm(:real_32cha, nCoil, target_res, target_matrix, location; verbose=verbose);
        @test typeof(csm) <: AbstractArray{<:Number,4}
        @test size(csm) == (target_matrix..., nCoil)
    end
    @testset "csm_Real_32cha 3D" begin
        target_matrix   = (100, 100, 10)
        csm = load_csm(:real_32cha, nCoil, target_res, target_matrix, location; verbose=verbose);
        @test typeof(csm) <: AbstractArray{<:Number,4}
        @test size(csm) == (target_matrix..., nCoil)
    end
end

