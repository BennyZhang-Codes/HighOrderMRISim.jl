module ReconstructionDemoSupport
using CUDA, MAT, LinearAlgebra
import HighOrderMRI as Recon
import RegularizedLeastSquares as RLS
import HighOrderMRI: AbstractNFFTs
import NFFT, NonuniformFFTs

function reconstruct(dataset; operator=get(ENV, "HIGHORDERMRISIM_RECON_OPERATOR", "lowrank"))
    d = MAT.matread(dataset)
    d["schema_version"] == 1 || error("Unsupported demo data schema.")
    dims = Tuple(Int.(vec(d["dimensions"])))
    T = eltype(d["rho"])
    spacing = T.(vec(d["resolution_m"]))
    xyz = d["xyz"]
    grid = Recon.Grid(; nX=dims[1], nY=dims[2], nZ=dims[3],
        Δx=spacing[1], Δy=spacing[2], Δz=spacing[3],
        x=vec(xyz[:, :, :, 1]), y=vec(xyz[:, :, :, 2]), z=vec(xyz[:, :, :, 3]))
    # NFFT first-order encoding supports centred Cartesian active axes only.
    centred = Recon.Grid(dims..., spacing...)
    for axis in (operator != "lowrank" ? () : dims[3] == 1 ? (:x, :y) : (:x, :y, :z))
        actual, expected = getproperty(grid, axis), getproperty(centred, axis)
        maximum(abs, actual .- expected) <= 8eps(T)*maximum(abs, expected) ||
            error("The LowRank NFFT grid must be centred; axis $axis differs.")
    end
    gpus = parse.(Int, split(get(ENV, "HIGHORDERMRISIM_RECON_GPUS", "0"), ','))
    CUDA.device!(first(gpus))
    CUDA.allowscalar(false)
    CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    rho = reshape(d["rho"], dims)
    mask = reshape(Bool.(d["mask"]), dims)
    csm = reshape(d["csm"], dims..., size(d["data"], 2))
    common = (; fieldmap=reshape(d["fieldmap_hz"], dims), csm, mask,
              recon_terms=d["orders"], gpus)
    coefficients, times = d["coefficients"], vec(d["times"])
    previous_backend = AbstractNFFTs.active_backend()
    op = nothing
    try
        # NFFTTools.sdc currently accesses NFFT.jl-specific plan storage. It
        # must not inherit the GPU operator's NonuniformFFTs backend.
        AbstractNFFTs.set_active_backend!(NFFT.backend())
        active = dims[3] == 1 ? 2 : 3
        weights = Recon.samplingDensity(coefficients[2:active+1, :], dims[1:active]; normalize=true)
        if operator == "lowrank"
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            preparation = vec(Int.(d["rf_preparation_index"]))
            events = unique(preparation)
            ns = count(==(first(events)), preparation)
            length(preparation) == ns*length(events) &&
                all(count(==(event), preparation) == ns for event in events) ||
                error("LowRank demo expects equal samples per acquired readout.")
            op = Recon.HighOrderLowRankOp(grid, reshape(coefficients, 16, ns, :),
                reshape(times, ns, :); common..., arrayType=CuArray,
                L_rank=parse(Int, get(ENV, "HIGHORDERMRISIM_RECON_RANK", "30")),
                rsvd_finalize=:gram, rsvd_backend=:kernel,
                rsvd_distribution=length(gpus) > 1 ? :voxel : :single,
                normal_distribution=length(gpus) > 1 ? :rank : :single,
                shared_rank_max=256, shared_basis_tol=T(1e-4))
        elseif operator == "kernel"
            op = Recon.HighOrderKernelOp(grid, coefficients, times; common...,
                backend=dims[3] == 1 ? :cached_full : :blocked,
                phase_backend=:gemm, gemm_backend=:cublas, block_size=320)
        else
            error("RECON_OPERATOR must be lowrank or kernel.")
        end
        rec_params = Dict{Symbol,Any}(:reconSize => dims, :solver => RLS.CGNR,
            :reg => RLS.L2Regularization(zero(T)),
            :iterations => parse(Int, get(ENV, "HIGHORDERMRISIM_RECON_ITERATIONS", "200")),
            :relTol => T === Float32 ? T(1e-9) : T(1e-12))
        image = Recon.recon_HOOp(op, CuArray(d["data"]), CuArray(weights), rec_params)
        # These are reference diagnostics, not a guarantee of solver convergence.
        # Physical RF/relaxation data need not reconstruct to pure spin density.
        metrics = Dict("magnitude_nrmse" => Recon.magnitude_nrmse(image, rho),
            "raw_complex_nrmse" => Recon.raw_complex_nrmse(image, rho),
            "magnitude_ssim" => Recon.magnitude_ssim(image, rho))
        destination = splitext(dataset)[1]*"-$operator-reconstruction.mat"
        MAT.matwrite(destination, Dict("image" => image, "rho" => d["rho"],
            "mode" => d["mode"], "metrics" => metrics); compress=true)
        println("RECONSTRUCTION_SAVED: ", destination, " / ", metrics)
        return image, metrics
    finally
        isnothing(op) || close(op)
        AbstractNFFTs.set_active_backend!(previous_backend)
    end
end
end # module
