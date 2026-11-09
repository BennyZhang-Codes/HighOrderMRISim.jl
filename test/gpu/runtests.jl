# Explicit CUDA entry point; CPU-only test/runtests.jl never includes this file.
using Test
using CUDA
using HighOrderMRISim

CUDA.functional() || error("A functional CUDA device is required for GPU tests.")
length(collect(CUDA.devices())) == 1 || error("Expose one physical GPU by UUID.")
CUDA.device!(0)
CUDA.allowscalar(false)
println("GPU regression: Julia ", VERSION, " / ", CUDA.name(CUDA.device()),
        " / ", CUDA.uuid(CUDA.device()))
flush(stdout)

@info "Testing GPU compatibility"
include("test_gpu_compatibility.jl")

@info "Testing GPU kernels"
include("test_gpu_kernels.jl")

@info "Testing independent coil reduction"
include("test_gpu_reduction.jl")
test_gpu_reduction()

@info "Testing fused coil combination"
include("test_gpu_fused_coils.jl")
test_fused_coils()

@info "Testing RF-aware workspaces and acquisition"
include("test_gpu_rf_buffers.jl")
test_gpu_rf_buffers()

@info "Testing reusable simulation and synchronization"
include("test_prepared_simulation.jl")
test_prepared_simulation([0])

println("GPU_REGRESSION_TESTS_OK")
