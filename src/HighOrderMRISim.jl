"""
    HighOrderMRISim.jl

A Julia package for high-order MRI signal simulation.

1. For simulation: we extend the KomaMRI.jl to realize high-order and parallel MRI simulation.
"""
module HighOrderMRISim

import Base.*, Base.+, Base.-, Base./, Base.vcat, Base.size, Base.abs, Base.getproperty, Base.copy, Base.show
# IMPORT PACKAGES


using PyPlot#, PlotlyJS  

using Parameters
using ThreadsX

using ProgressMeter
using Reexport

using CUDA, Interpolations, Statistics

import ImageTransformations: imresize

@reexport import MAT

@reexport using KomaMRI
import KomaMRI.KomaMRICore: SimulationMethod, SpinStateRepresentation
import Functors: @functor


include("utils/utils.jl")
include("sphericalharmonics/SphericalHarmonics.jl")
include("datatypes/datatypes.jl")
include("grid/Grid.jl")             # grid for simulation and reconstruction
include("phantom/phantom.jl")
include("simulation/simulation.jl")

include("plot/plot.jl")

#Package version
using Pkg
__VERSION__ = VersionNumber(Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

end # module HighOrderMRISim
