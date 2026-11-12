using HighOrderMRISim, CUDA, MAT

# Demo settings
demo_plots = true
simulation_only = false
mode = "physical"  # use "density" for the ideal spin-density control
array_type = CuArray
gpus = [0]         # zero-based IDs among visible devices; use [] with Array

include(joinpath(@__DIR__, "1_load_inputs.jl"))

# 2. scanner & sim_params
sys = Scanner();
sim_params = KomaMRICore.default_sim_params();
sim_params["sim_method"]  = BHO;      # using "BlochHighOrder" for simulation with high-order terms
sim_params["return_type"] = "mat";    # setting with "mat", return the signal data for all channel
sim_params["implementation"] = "auto";
sim_params["coil_combination"] = "gemm";
sim_params["precision"]   = "f32";

# 3. Simulate and export the exact grid/encoding used by the simulation.
include(joinpath(@__DIR__, "..", "demo", "SimulationDemoSupport.jl"))
run = SimulationDemoSupport.prepare_run!(obj, hoseqMeasured, sim_params; mode)
signal = simulate(obj, hoseqMeasured, sys; sim_params, array_type, gpus, callbacks=run.callbacks)
if !simulation_only
    rho = hophantom_brain(brainphantom, :ρ, target_res, target_matrix, location, offcenter;
        fieldstrength, b0_type, csm_type, csm_nCoil, verbose=false)
    dataset = SimulationDemoSupport.export_run("2d", signal, obj, hoseqMeasured,
        sim_params, run, rho, target_res, offcenter)
    println("Reconstruct separately with demo2D/Recon2D.jl and the saved dataset.")
end

