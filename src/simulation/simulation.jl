import KomaMRICore: run_spin_precession!, run_spin_excitation!
import KomaMRICore: initialize_spins_state, Bloch, sim_output_dim

import KomaMRICore: mul!, Q
import KomaMRICore: run_spin_excitation_parallel!, run_spin_precession_parallel!, run_sim_time_iter!
import KomaMRICore: simulate

# define BlochHighOrder struct first, for workflow control in the run_spin_excitation! & run_spin_precession! functions
include("BlochHighOrder.jl")
export BlochHighOrder

# define simulation functions for different phantoms
include("Simulate.jl")
include("SimulatorCore.jl")
include("Bloch/BlochSimulatitonMethod.jl")
include("Bloch/BlochHighOrderSimulatitonMethod.jl")
include("kernels/HighOrderGPU.jl")
include("kernels/CoilReductionGPU.jl")
include("kernels/FusedCoilGPU.jl")

include("SimType.jl")
export SimType
