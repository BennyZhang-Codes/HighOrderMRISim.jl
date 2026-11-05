using CUDA

# Demo settings
simulation_only = false
mode = "physical"  # use "density" for the ideal spin-density control
array_type = CuArray
gpus = [0]         # zero-based IDs among visible devices; use [] with Array

# ------------------------------------------------------------
# 1. Define Bloch simulation method (BlochHighOrder)
# ------------------------------------------------------------
BHO = BlochHighOrder("1111", true, true)
# Alternative: BHO = BlochHighOrder("0000", true, true), use only nominal Gx, Gy, Gz

# ------------------------------------------------------------
# 2. Initialize scanner system and simulation parameters
# ------------------------------------------------------------
sys = Scanner();
sim_params = KomaMRICore.default_sim_params()

sim_params["sim_method"]  = BHO      # using "BlochHighOrder" for simulation with high-order terms
sim_params["return_type"] = "mat"    # return full complex signal for all channels
sim_params["implementation"] = "auto"
sim_params["coil_combination"] = "gemm"
sim_params["precision"]   = "f32"

# ------------------------------------------------------------
# 3. Run the signal simulation
# ------------------------------------------------------------
include(joinpath(@__DIR__, "..", "demo", "SimulationDemoSupport.jl"))
run = SimulationDemoSupport.prepare_run!(obj, hoseq, sim_params; mode)
signal = simulate(obj, hoseq, sys; sim_params, array_type, gpus, callbacks=run.callbacks)
if !simulation_only
    rho = hophantom_brain(brainphantom, :ρ, target_res, target_matrix, location, offcenter;
        fieldstrength, b0_type, csm_type, csm_nCoil, verbose=false)
    dataset = SimulationDemoSupport.export_run("3d", signal, obj, hoseq,
        sim_params, run, rho, target_res, offcenter)
    println("Reconstruct separately with demo3D/4_recon.jl and the saved dataset.")
end
