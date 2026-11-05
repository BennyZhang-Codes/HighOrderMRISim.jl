# Run after Sim2D.jl, in a process exposing only the reconstruction GPUs.
operator = "lowrank"
gpus = [0]

length(ARGS) <= 1 || error("Usage: julia demo2D/Recon2D.jl [DATASET.mat]")
dataset = isempty(ARGS) ?
    joinpath(@__DIR__, "..", "output", "2d", "2d-0000-physical.mat") : only(ARGS)
include(joinpath(@__DIR__, "..", "demo", "ReconstructionDemoSupport.jl"))
image, metrics = ReconstructionDemoSupport.reconstruct(abspath(dataset); operator, gpus)
