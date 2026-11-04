# ------------------------------------------------------------
# 1. Define Bloch simulation method (BlochHighOrder)
# ------------------------------------------------------------
BHO = BlochHighOrder(get(ENV, "HIGHORDERMRISIM_DEMO_ORDERS", "1111"), true, true)
# Alternative: BHO = BlochHighOrder("0000", true, true), use only nominal Gx, Gy, Gz

# ------------------------------------------------------------
# 2. Initialize scanner system and simulation parameters
# ------------------------------------------------------------
sys = Scanner();
sim_params = KomaMRICore.default_sim_params()

sim_params["sim_method"]  = BHO      # using "BlochHighOrder" for simulation with high-order terms
sim_params["return_type"] = "mat"    # return full complex signal for all channels
sim_params["precision"]   = "f32"    # single-precision computation
sim_params["gpu"]         = true     # enable GPU acceleration
sim_params["gpu_device"]  = 0        # select GPU device ID
sim_params["Nblocks"]     = 2000     # number of blocks for GPU simulation (memory dependent)
if haskey(ENV, "HIGHORDERMRISIM_DEMO_MAX_BLOCK_LENGTH")
    sim_params["max_block_length"] = parse(Int, ENV["HIGHORDERMRISIM_DEMO_MAX_BLOCK_LENGTH"])
end

# ------------------------------------------------------------
# 3. Run the signal simulation
# ------------------------------------------------------------
signal    = simulate(obj, hoseq, sys; sim_params);          

if get(ENV, "HIGHORDERMRISIM_DEMO_SIMULATION_ONLY", "false") != "true"
# Reshape
reshape(signal, nSample_adc, :, csm_nCoil);  # [nSample_adc, nADC, nCoil]

demo_plots && plt_plot([abs.(signal[:, 1])]; width=30, height=5, fontsize_label=6, ylabel="Signal Amplitude [a.u.]", xlabel="ADC Sample")
# fig.savefig("$(path)/out/signal.png", dpi=300, transparent=true, bbox_inches="tight", pad_inches=0.0)


# ------------------------------------------------------------
# 4. Prepare k-space trajectory for reconstruction
# ------------------------------------------------------------
_, _, _, kspha = get_kspace(hoseq; Δt=1);  # get kspha, with high-order terms

# Optional visualization
# fig = plt_kspha(kspha*2π, dt_adc; width=30, height=9, fontsize_label=6)
# fig.savefig("$(path)/out/kspha.png", dpi=300, transparent=true, bbox_inches="tight", pad_inches=0.0)

# ------------------------------------------------------------
# 5. Generate field maps, coil sensitivities, and reference images
# ------------------------------------------------------------
target_fov    = Tuple(params["params"]["FOV"])
target_matrix = Tuple(Int64.(params["params"]["MatrixSize"]))
target_res    = target_fov ./ target_matrix

b0  = hophantom_brain(brainphantom, :Δw , target_res, target_matrix, location, offcenter; 
        fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

csm = hophantom_brain(brainphantom, :csm, target_res, target_matrix, location, offcenter; 
        fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

ρ   = hophantom_brain(brainphantom, :ρ  , target_res, target_matrix, location, offcenter; 
        fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

mask = hophantom_brain(brainphantom, :headmask, target_res, target_matrix, location, offcenter; 
        fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

xyz_grid = hophantom_brain(brainphantom, :grid, target_res, target_matrix, location, offcenter; 
        fieldstrength=fieldstrength, b0_type=b0_type, csm_type=csm_type, csm_nCoil=csm_nCoil, verbose=verbose)

# Optional plotting for sanity check:
# plt_image(rotl90(mosaic(ρ; dim=3, nRow=5, nCol=7)))
# plt_image(rotl90(mosaic(abs.(csm)[:,:,:,1]; dim=3, nRow=5, nCol=7)))

# ------------------------------------------------------------
# 6. Build timing vectors and export data for reconstruction
# ------------------------------------------------------------
dt_adc       = params["skope"]["dt_adc"]
nSample_adc  = Int64(params["skope"]["nSample_adc"])
nDynamic     = Int64(params["skope"]["nDynamic"])
t_adc = (collect(1:nSample_adc) .- 0.5) * dt_adc
t_adc = repeat(t_adc', nDynamic, 1)

data = Dict(
    "signal"       => signal,
    "kspha"        => kspha,
    "t_adc"        => t_adc,
    "dt_adc"       => dt_adc,
    "nSample_adc"  => nSample_adc,
    "ref"          => ρ,
    "mask"         => mask,
    "b0"           => b0,
    "csm"          => csm,
    "xyz_grid"     => xyz_grid,
    "target_fov"   => collect(target_fov),
    "target_res"   => collect(target_res),
    "matrixsize"   => Int64.(collect(target_matrix)),
)

# Save results for later reconstruction
matwrite("$(path)/$(name)_1111_wB0.mat", data)
end # optional reconstruction-data preparation
