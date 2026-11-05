# Run after Sim2D.jl, in a process exposing only the reconstruction GPUs.
include(joinpath(@__DIR__, "..", "demo", "ReconstructionDemoSupport.jl"))
length(ARGS) == 1 || error("Usage: julia demo2D/Recon2D.jl DATASET.mat")
image, metrics = ReconstructionDemoSupport.reconstruct(abspath(only(ARGS)))
