using HighOrderMRISim, MRIReco, MAT
import RegularizedLeastSquares: SolverInfo

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
HighOrderMRISim.KomaMRI.plot_seq(seq) # plot_seq function from KomaMRI.jl, plot the Sequence

t = dt_adc * (nSample-1);
GR_dfcMeasured = reshape([KomaMRIBase.Grad(bfieldMeasured[idx,:], t, dt_adc/2, dt_adc/2, 0) for idx=1:nTerm], :, 1);

hoseqMeasured = HO_Sequence(seq); 
# copy the 1st-order nominal gradient data from the seq object to the hoseq object
hoseqMeasured.GR_dfc[2:4, :] = hoseqMeasured.SEQ.GR;    
hoseqMeasured.GR_dfc[:,5] = GR_dfcMeasured;   # "5" is the index of the readout block in the spiral sequence     
plt_seq(hoseqMeasured)

#########################################################################################
# 2. Simulation with dynamic field changes
#     ΔB₀ map and coil-sensitivity maps (acquired from a real GRE acquisition)
#########################################################################################
T = Float64;
nX = nY = 200; nZ = 1;  # matrix size for recon
Δx = Δy = 1e-3; Δz = 2e-3;

# settings for phantom
B0 = false    # turn on B0
T2 = false    # turn off T2

BHO          = BlochHighOrder("0000", true, true)  # ["0000"] just using nominal gradients. turn on Δw_excitation, Δw_precession
brainphantom = BrainPhantom(prefix="brain3D", x=0.5, y=0.5, z=0.5) # setting for Phantom: decide which phantom file to use, loading phantom from src/phantom/mat folder                                
target_res   = (0.5, 0.5, 2.).*1e-3     # target resolution
target_matrix= (400, 400,  1)           # target matrix size
location     = ( 0.,  0., 0.).*1e-3     # target location of the sampling center
offcenter    = ( 0.,  0., 0.).*1e-3     # offcenter shift
b0_type      = :real
csm_type     = :real_32cha 
csm_nCoil    = 32
verbose      = false    

# 1. phantom
obj = hophantom_brain(brainphantom, target_res, target_matrix, location, offcenter; 
            b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
obj.Δw .= B0 ? obj.Δw : obj.Δw * 0;     # set Δw to 0 if B0=false
obj.T2 .= T2 ? obj.T2 : obj.T2 * Inf;   # cancel T2 relaxiation
plt_phantom(obj, :ρ)

# 2. scanner & sim_params
sys = Scanner();
sim_params = KomaMRICore.default_sim_params();
sim_params["sim_method"]  = BHO;      # using "BlochHighOrder" for simulation with high-order terms
sim_params["return_type"] = "mat";    # setting with "mat", return the signal data for all channel
sim_params["precision"]   = "f64";
sim_params["gpu"]         = true;     # using GPU for simulation
sim_params["gpu_device"]  = 0;        # set the GPU device number, if using GPU for simulation
sim_params["Nblocks"]     = 2000;     # the number of blocks for GPU simulation, set according to the GPU memory

# 3. simulate
signal    = simulate(obj, hoseqMeasured, sys; sim_params);          

# 4. Adding noise to signal data
snr       = 10;
data      = AddNoise(signal[:,:,1], snr);


#########################################################################################
# 3. preparation for reconstruction
#########################################################################################
# get your ΔB₀ map and coil-sensitivity maps
target_fov    = (200e-3, 200e-3, 2e-3)
target_matrix = (200, 200, 1)
target_res    = target_fov ./ target_matrix

b0 = hophantom_brain(brainphantom, :Δw, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
plt_image(rotl90(b0[:,:]))

csm = hophantom_brain(brainphantom, :csm, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
plt_image(rotl90(abs.(csm)[:,:, 1,1]))

# get spin-density map
ρ = hophantom_brain(brainphantom, :ρ, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
plt_image(rotl90(ρ[:,:]))

# get nominal k-space trajectory
_, kspha_xyz = get_kspace(hoseqMeasured.SEQ; Δt=1);
C = maximum(2*abs.(kspha_xyz[:]));  #Normalize k-space to -.5 to .5 for NUFFT

# preparation for NUFFT reconstruction
tr = Trajectory("2dspiral", T.(kspha_xyz[:,1:2]' ./ C),  T.(datatime), T.(5e-3), T.(nSample*dt_adc), 1, nSample, 1, false, false);
dat         = Array{Array{Complex{T},2},3}(undef,1,1,1);
dat[1,1,1]  = signal[:,:,1];
acqData     = AcquisitionData(tr, dat, encodingSize=(nX, nY));

###########################################################################################
# 4. Reconstruction (NUFFT)
###########################################################################################
params = Dict{Symbol, Any}()
params[:reco] = "multiCoil"
params[:reconSize] = (nX, nY)
params[:regularization] = "L2"
params[:λ] = 1e-3
params[:iterations] = 20
params[:solver] = "cgnr"
params[:senseMaps] = Complex{T}.(reshape(csm, nX, nY, nZ, 32));

@time Ireco = reconstruction(acqData, params);
img = Ireco[:,:,1,1,1,1];
fig = plt_image(rotl90(abs.(img)); vminp=0, vmaxp=99.0)




