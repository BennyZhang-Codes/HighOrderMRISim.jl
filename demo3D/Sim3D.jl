# Run from any directory; sequence and phantom files are resolved relative to this script.
include("1_load_seq.jl")
include("2_load_phantom.jl")
include("3_run_sim.jl")
