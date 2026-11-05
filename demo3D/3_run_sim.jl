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
sim_params["implementation"] = get(ENV, "HIGHORDERMRISIM_DEMO_IMPLEMENTATION", "auto")
sim_params["coil_combination"] = get(ENV, "HIGHORDERMRISIM_DEMO_COIL_COMBINATION", "gemm")
sim_params["precision"]   = get(ENV, "HIGHORDERMRISIM_DEMO_PRECISION", "f32")
sim_params["gpu"]         = true     # enable GPU acceleration
sim_params["gpu_device"]  = 0        # select GPU device ID

# ------------------------------------------------------------
# 3. Run the signal simulation
# ------------------------------------------------------------
include(joinpath(@__DIR__, "..", "demo", "SimulationDemoSupport.jl"))
run = SimulationDemoSupport.prepare_run!(obj, hoseq, sim_params)
signal = simulate(obj, hoseq, sys; sim_params,
    SimulationDemoSupport.device_options()..., callbacks=run.callbacks)
if get(ENV, "HIGHORDERMRISIM_DEMO_SIMULATION_ONLY", "false") != "true"
    rho = hophantom_brain(brainphantom, :ρ, target_res, target_matrix, location, offcenter;
        fieldstrength, b0_type, csm_type, csm_nCoil, verbose=false)
    dataset = SimulationDemoSupport.export_run("3d", signal, obj, hoseq,
        sim_params, run, rho, target_res, offcenter)
    println("Reconstruct separately with demo3D/4_recon.jl and the saved dataset.")
end
