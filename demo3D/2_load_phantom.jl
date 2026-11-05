# ------------------------------------------------------------
# 1. Phantom generation and coil/B0 configuration
# ------------------------------------------------------------
brainphantom = BrainPhantom(prefix="brain3D", x=0.5, y=0.5, z=0.5)

# Reuse these sequence-defined values when exporting reconstruction inputs.
target_matrix = Tuple(Int.(seq.DEF["MatrixSize"]))
target_fov = Tuple(Float64.(seq.DEF["FOV"]))
target_res = target_fov ./ target_matrix
location      = (0., 0., 20.) .* 1e-3       # center location of the sampled volume [m]
offcenter     = (0., 0., 0.) .* 1e-3        # off-center shift [m]

fieldstrength= "1.5T"          # to determine the tissue properties
b0_type   = :real              # static field (B0) type
csm_type  = :real_32cha        # coil sensitivity map type
csm_nCoil = 32                 # number of coils
verbose   = false              # suppress console messages

# Generate high-order brain phantom object
obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter;
            fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

# Optional visualization:
# plt_phantom(obj, :ρ)         # proton density
# plt_phantom(obj, :csm_mag)   # coil sensitivity magnitude
# plt_phantom(obj, :Δw)        # B0 inhomogeneity map

# ------------------------------------------------------------
# 2. Enable or disable specific physical effects (B0, T2)
# ------------------------------------------------------------
B0 = true    # turn on B0
T2 = true    # turn on T2

# Apply toggles: zero Δw if B0 is off, or make T2 infinite (no decay) if T2 is off
obj.Δw .= B0 ? obj.Δw : obj.Δw * 0;     
obj.T2 .= T2 ? obj.T2 : obj.T2 * Inf;   

