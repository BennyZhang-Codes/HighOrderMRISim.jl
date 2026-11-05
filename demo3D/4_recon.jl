# Run after Sim3D.jl, in a process exposing only the reconstruction GPUs.
include(joinpath(@__DIR__, "..", "demo", "ReconstructionDemoSupport.jl"))
length(ARGS) == 1 || error("Usage: julia demo3D/4_recon.jl DATASET.mat")
operator = get(ENV, "HIGHORDERMRISIM_RECON_OPERATOR", "kernel")
image, metrics = ReconstructionDemoSupport.reconstruct(abspath(only(ARGS)); operator)
