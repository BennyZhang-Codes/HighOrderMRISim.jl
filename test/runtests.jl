using HighOrderMRISim
using Test

@info "testing... csm"
include("test_csm.jl")

@info "testing... HOPhantom"
include("test_HOPhantom.jl")

@info "testing... get_center_crop"
include("test_get_center_crop.jl")