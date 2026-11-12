using HighOrderMRISim, MAT

# CPU-only inputs shared by the demo and documentation.
# Set demo_plots before including; this file never simulates or reconstructs.

#########################################################################################
# 1. Load sequence and dynamic field data
#     a. Load sequence file (*.seq format from pulseq)
#     b. Load dynamic field data (*.mat format)
#     c. Create a HO_Sequence object as defined in HighOrderMRISim.jl by combining the
#        sequence and dynamic field data.
#########################################################################################
path     = joinpath(@__DIR__, "")
seq_file = "$(path)/7T_1p0_200_r4.seq"   # *.seq file is the pulseq's sequence file
# under-sampling factor: R = 4
# inplane resolution: 1 mm x 1 mm
# FOV: 200 mm x 200 mm
# readout duration: ~29 ms

dfc_file = "$(path)/7T_1p0_200_r4.mat"   # *.mat file contains the nominal and measured trajectories.

dfc_data = MAT.matread(dfc_file);
dt_Measured   = dfc_data["dt_Measured"];

ksphaMeasured = dfc_data["ksphaMeasured"];   # coefficients of dynamic measurment (up to 3rd order) of the data stitching method
ksphaMeasured[:, 2] *= -1;
startMeasured = dfc_data["startMeasured"];
tauMeasured   = dfc_data["tauMeasured"];      # synchronization delay

dt_adc        = dfc_data["dt_adc"];
datatime      = vec(dfc_data["datatime"]);
nSample = size(datatime, 1);
nTerm       = 16;

ksphaMeasured = InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured + tauMeasured * dt_adc, datatime);
bfieldMeasured = traj2grad(ksphaMeasured, dt_adc; dim=1) / γ * 1e3;
bfieldMeasured = bfieldMeasured' * 1e-3;

# 1. read the *.seq file
seq = read_seq(seq_file)[3:end-3]; # read_seq function from KomaMRI.jl, return a struct of Sequence
demo_plots && HighOrderMRISim.KomaMRI.plot_seq(seq)

t = dt_adc * (nSample-1);
GR_dfcMeasured = reshape([KomaMRIBase.Grad(bfieldMeasured[idx,:], t, dt_adc/2, dt_adc/2, 0) for idx=1:nTerm], :, 1);

hoseqMeasured = HO_Sequence(seq);
# copy the 1st-order nominal gradient data from the seq object to the hoseq object
hoseqMeasured.GR_dfc[2:4, :] = hoseqMeasured.SEQ.GR;
hoseqMeasured.GR_dfc[:,5] = GR_dfcMeasured;   # "5" is the index of the readout block in the spiral sequence
# Explicit styling also applies in the release-pinned demo environment.
demo_plots && plt_seq(hoseqMeasured; color_facecolor="#ffffff", color_label="#000000")

#########################################################################################
# 2. Simulation with dynamic field changes
#     ΔB₀ map and coil-sensitivity maps (acquired from a real GRE acquisition)
#########################################################################################
T = Float32;
# One sequence-defined grid for simulation, receive maps, and reconstruction.
target_matrix = Tuple(Int.(seq.DEF["matrixSize"]))
target_fov = Tuple(Float64.(seq.DEF["FOV"]))
target_res = target_fov ./ target_matrix
nX, nY, nZ = target_matrix
Δx, Δy, Δz = target_res

# settings for phantom
B0 = false    # turn on B0
T2 = false    # turn off T2

BHO          = BlochHighOrder("1111", true, true)
brainphantom = BrainPhantom(prefix="brain3D", x=0.5, y=0.5, z=0.5) # setting for Phantom: decide which phantom file to use, loading phantom from src/phantom/mat folder
location     = ( 0.,  0., 0.).*1e-3     # target location of the sampling center
offcenter    = ( 0.,  0., 0.).*1e-3     # offcenter shift
fieldstrength= "1.5T"
b0_type      = :real
csm_type     = :real_32cha
csm_nCoil    = 32
verbose      = false

# 1. phantom
obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter;
            fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
obj.Δw .= B0 ? obj.Δw : obj.Δw * 0;     # set Δw to 0 if B0=false
obj.T2 .= T2 ? obj.T2 : obj.T2 * Inf;   # cancel T2 relaxiation
demo_plots && plt_phantom(obj, :ρ; color_facecolor="#ffffff", color_label="#000000")
