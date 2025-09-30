BHO = BlochHighOrder("1111", true, true)            
# BHO = BlochHighOrder("0000", true, true)   # only nominal Gx, Gy and Gz are used
# 2. scanner & sim_params
sys = Scanner();
sim_params = KomaMRICore.default_sim_params();
sim_params["sim_method"]  = BHO;      # using "BlochHighOrder" for simulation with high-order terms
sim_params["return_type"] = "mat";    # setting with "mat", return the signal data for all channel
sim_params["precision"]   = "f32";
sim_params["gpu"]         = true;     # using GPU for simulation
sim_params["gpu_device"]  = 0;        # set the GPU device number, if using GPU for simulation
sim_params["Nblocks"]     = 2000;     # the number of blocks for GPU simulation, set according to the GPU memory

# 3. simulate
signal    = simulate(obj, hoseq, sys; sim_params);          


s = reshape(signal, nSample_adc, :, csm_nCoil); # reshape to [nSample_adc, nADC, nCha]
plt_plot([abs.(s[:,1,8])])


fig = plt_plot([abs.(signal[:,1])]; width=30, height=5, fontsize_label=6, ylabel="Signal Amplitude [a.u.]", xlabel="ADC Sample")
# fig.savefig("$(path)/out/signal.png", dpi=300, transparent=true, bbox_inches="tight", pad_inches=0.0)


##########################################
# preparation for reconstruction
##########################################
_, _, _, kspha = get_kspace(hoseq; Δt=1);  # get kspha, with high-order terms
# _, kspha = get_kspace(hoseq.SEQ; Δt=1);  # get nominal k-space trajectory
# a = zeros(size(kspha, 1), 16);
# a[:, 2:4] = kspha;
# kspha = a;

# fig = plt_kspha(kspha*2π, dt_adc; width=30, height=9, fontsize_label=6)
# fig.savefig("$(path)/out/kspha.png", dpi=300, transparent=true, bbox_inches="tight", pad_inches=0.0)


target_fov    = Tuple(params["params"]["FOV"])
target_matrix = Tuple(Int64.(params["params"]["MatrixSize"]))
target_res    = target_fov ./ target_matrix

b0 = hophantom_brain(brainphantom, :Δw, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
# plt_image(rotl90(mosaic(b0; dim=3, nRow=5, nCol=7)))

csm = hophantom_brain(brainphantom, :csm, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
# plt_image(rotl90(mosaic(abs.(csm)[:,:,:,1]; dim=3, nRow=5, nCol=7)))

ρ = hophantom_brain(brainphantom, :ρ, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)
# plt_image(rotl90(mosaic(ρ; dim=3, nRow=5, nCol=7)))

xyz_grid = hophantom_brain(brainphantom, :grid, target_res, target_matrix, location, offcenter; b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)


dt_adc = params["skope"]["dt_adc"]
nSample_adc = Int64(params["skope"]["nSample_adc"]);
nDynamic = Int64(params["skope"]["nSample_adc"])
t_adc = (collect(1:nSample_adc).-0.5)*dt_adc;
t_adc = repeat(t_adc', nDynamic,1);

data = Dict(
    "signal" => signal,
    "kspha" => kspha,
    "t_adc" => t_adc,
    "dt_adc" => dt_adc,
    "nSample_adc" => nSample_adc,
    "ref"   => ρ,
    "b0"  => b0,
    "csm" => csm,
    "xyz_grid" => xyz_grid,
    "target_fov" => collect(target_fov),
    "target_res" => collect(target_res),
    "matrixsize" => Int64.(collect(target_matrix)),
)
matwrite("$(path)/$(name)_1111_wB0.mat", data)
