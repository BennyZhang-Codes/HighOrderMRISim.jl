# Run after Sim3D.jl, in a process exposing only the reconstruction GPUs.
operator = "kernel"
gpus = [0]

length(ARGS) <= 1 || error("Usage: julia demo3D/4_recon.jl [DATASET.mat]")
dataset = isempty(ARGS) ?
    joinpath(@__DIR__, "..", "output", "3d", "3d-1111-physical.mat") : only(ARGS)
include(joinpath(@__DIR__, "..", "demo", "ReconstructionDemoSupport.jl"))
image, metrics = ReconstructionDemoSupport.reconstruct(abspath(dataset); operator, gpus)
