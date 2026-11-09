import KomaMRICore: run_spin_precession!, run_spin_excitation!
import KomaMRICore: initialize_spins_state, Bloch, sim_output_dim

import KomaMRICore: mul!, Q
import KomaMRICore: run_spin_excitation_parallel!, run_spin_precession_parallel!, run_sim_time_iter!
import KomaMRICore: simulate

# Simulation methods and the shared time loop.
include("BlochHighOrder.jl")
export BlochHighOrder

include("Simulate.jl")
include("SimulatorCore.jl")
include("Bloch/BlochSimulatitonMethod.jl")
include("Bloch/BlochHighOrderSimulatitonMethod.jl")

# CUDA kernels and coil workspaces.
include("kernels/HighOrderGPU.jl")
include("kernels/CoilReductionGPU.jl")
include("kernels/FusedCoilGPU.jl")

# Multi-device execution and optional resource reuse.
include("DistributedSimulation.jl")
include("PreparedSimulation.jl")
export PreparedSimulation, prepare_simulation

include("SimType.jl")
export SimType
