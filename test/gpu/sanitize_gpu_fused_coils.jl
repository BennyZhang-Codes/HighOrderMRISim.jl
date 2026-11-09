# Small padded/tail cases for compute-sanitizer memcheck/racecheck/synccheck.
include("test_gpu_fused_coils.jl")
include("test_gpu_rf_buffers.jl")
length(collect(CUDA.devices())) == 1 || error("Expose one GPU by UUID.")
CUDA.device!(0)
CUDA.allowscalar(false)
@testset "Fused CUDA sanitizer workload" begin
    for mode in ("fused_warp", "fused_tiled"), precision in ("f32", "f64"),
        (nspins, nadc, ncoils) in ((1, 1, 1), (259, 7, 33), (257, 32, 3))
        check_fused_coils(precision, mode; nspins, nadc, ncoils)
    end
    for precision in ("f32", "f64"), mode in ("fused_warp", "fused_tiled"), max_rf_adc in (0, 3)
        check_rf_workspace_reuse(precision, mode, max_rf_adc, 3)
    end
end
println("FUSED_COIL_SANITIZER_WORKLOAD_OK")
