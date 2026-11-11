ENV["MPLBACKEND"] = "Agg"

using HighOrderMRISim
using Test

@info "testing... KomaMRI compatibility"
include("unit/test_koma_compatibility.jl")
include("unit/test_sequence_arithmetic.jl")
include("unit/test_execution_options.jl")
include("unit/test_sampled_storage.jl")
include("unit/test_grid.jl")

@info "testing... csm"
include("unit/test_csm.jl")

@info "testing... HOPhantom"
include("unit/test_phantom_construction.jl")
include("unit/test_phantom_interfaces.jl")
include("unit/test_HOPhantom.jl")

@info "testing... get_center_crop"
include("unit/test_get_center_crop.jl")
include("unit/test_image_metrics.jl")
include("unit/test_trajectory_utils.jl")
include("unit/test_readme_example.jl")

@info "testing... plotting"
include("unit/test_field_plots.jl")
include("unit/test_spatial_plots.jl")
