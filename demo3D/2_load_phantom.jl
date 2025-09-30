brainphantom = BrainPhantom(prefix="brain3D", x=0.5, y=0.5, z=0.5)
target_res   = ( 1.,  1., 1.).*1e-3     # target resolution
target_matrix= (180, 180, 80)           # target matrix size
location     = ( 0.,  0., 20.).*1e-3     # target location of the sampling center
offcenter    = ( 0.,  0., 0.).*1e-3     # offcenter shift

b0_type      = :real
csm_type     = :real_32cha 
csm_nCoil    = 32
verbose      = false    

obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter;
            b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

# plt_phantom(obj, :ρ;)
# plt_phantom(obj, :csm_mag)
# plt_phantom(obj, :Δw)

B0 = true    # turn on B0
T2 = true    # turn on T2

obj.Δw .= B0 ? obj.Δw : obj.Δw * 0;     # set Δw to 0 if B0=false
obj.T2 .= T2 ? obj.T2 : obj.T2 * Inf;   # cancel T2 relaxiation

