# Run after Sim2D.jl, in a process exposing only the reconstruction GPUs.
operator = "kernel"
gpus = [0]
iterations = 100  # Stop before the observed Float32 CGNR semiconvergence.

length(ARGS) <= 1 || error("Usage: julia demo2D/Recon2D.jl [DATASET.mat]")
dataset = isempty(ARGS) ?
    joinpath(@__DIR__, "..", "output", "2d", "2d-1111-physical.mat") : only(ARGS)
include(joinpath(@__DIR__, "..", "demo", "ReconstructionDemoSupport.jl"))
image, metrics = ReconstructionDemoSupport.reconstruct(abspath(dataset); operator, gpus, iterations)
