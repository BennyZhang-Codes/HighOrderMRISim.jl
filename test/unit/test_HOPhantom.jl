using Test
using HighOrderMRISim

brainphantom = BrainPhantom(prefix="brain3D", x=0.5, y=0.5, z=0.5)

objbrain     = brainphantom
target_res   = ( 2.,  2., 2.).*1e-3;     # target resolution
target_matrix= (100, 100, 10)      ;     # target matrix size
location     = ( 0.,  0., 0.).*1e-3;     # target location of the sampling center
offcenter    = ( 0.,  0., 0.).*1e-3;     # offcenter shift
b0_type      = :real
csm_type     = :real_32cha 
csm_nCoil    = 32
verbose      = false    


@testset "HighOrder Phantom" begin
    @testset "HOPhantom Brain" begin
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

        nSpin, nCoil = size(obj.csm)
        @test nSpin == length(obj.x)
        @test nCoil == csm_nCoil
        # plt_phantom(obj, :ρ; view_2d=false)
    end

    @testset "HOPhantom Brain: Reference" begin
        target_matrix= (100, 100, 10)

        xyz_grid = hophantom_brain(brainphantom, :grid, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        @test size(xyz_grid) == (target_matrix..., 3)
        # plt_grid(xyz_grid)

        ρ = hophantom_brain(brainphantom, :ρ, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

        @test size(ρ) == target_matrix
        # plt_image(mosaic(ρ; dim=3, nRow=2, nCol=5))

        Δw = hophantom_brain(brainphantom, :Δw, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

        @test size(Δw) == target_matrix
        # plt_image(mosaic(Δw; dim=3, nRow=2, nCol=5))

        csm = hophantom_brain(brainphantom, :csm, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

        @test size(csm) == (target_matrix..., csm_nCoil)
        # plt_image(mosaic(abs.(csm[:,:,:,32]); dim=3, nRow=2, nCol=5))
    end

    @testset "HOPhantom Brain: 2D & Orientation" begin
        target_matrix= (1, 100, 100)        
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        # plt_phantom(obj, :ρ; view_2d=false)

        target_matrix= (100, 1, 100)  
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        # plt_phantom(obj, :ρ; view_2d=false)

        target_matrix= (100, 100, 1)    
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        # plt_phantom(obj, :ρ; view_2d=false)
    end

    @testset "HOPhantom Brain: 3D & Orientation" begin
        target_matrix= (10, 100, 100)  
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        # plt_phantom(obj, :ρ; view_2d=false)

        target_matrix= (100, 10, 100)     
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        # plt_phantom(obj, :ρ; view_2d=false)

        target_matrix= (100, 100, 10)   
        obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
                    b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
        # plt_phantom(obj, :ρ; view_2d=false)
    end
end
