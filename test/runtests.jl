using HighOrderMRISim
using Test

@info "testing... KomaMRI compatibility"
include("unit/test_koma_compatibility.jl")
include("unit/test_execution_options.jl")
include("unit/test_sampled_storage.jl")

@info "testing... csm"
include("unit/test_csm.jl")

@info "testing... HOPhantom"
include("unit/test_HOPhantom.jl")

@info "testing... get_center_crop"
include("unit/test_get_center_crop.jl")
