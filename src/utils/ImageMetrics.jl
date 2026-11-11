import ImageQualityIndexes: assess_ssim
import LinearAlgebra: norm, dot

function check_metric_inputs(x_ref, x)
    size(x_ref) == size(x) || throw(DimensionMismatch("Reference and image shapes must match."))
    check_image_values(x_ref)
    check_image_values(x)
    return nothing
end

function image_scale_unchecked(x_ref, x)
    alpha = norm(x)>0 ? (dot(vec(x_ref),vec(x))+dot(vec(x),vec(x_ref))) /
            (2*dot(vec(x),vec(x))) : 1.0
    x2 = x.*alpha
    return x2
end

"""
    HO_img_scale(reference, image)

Fit one real intensity scale to a finite image with the same nonempty shape as
the reference. A zero image remains zero. This removes global amplitude error;
do not use the fitted image to diagnose unscaled simulation/reconstruction error.
"""
function HO_img_scale(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real})
    check_metric_inputs(x_ref, x)
    return image_scale_unchecked(x_ref, x)
end

"""
    HO_NRMSE(reference, image; scale=true)

Legacy real-image RMSE divided by `maximum(abs.(reference)) - minimum(abs.(reference))`.
The default fits one real intensity scale before comparison. This is not the
norm-normalized `magnitude_nrmse` used by the HighOrderMRI reconstruction demos.
Inputs must have the same finite, nonempty shape; zero reference dynamic range
raises `DomainError` instead of returning NaN or Inf.
"""
function HO_NRMSE(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    check_metric_inputs(x_ref, x)
    dynamic_range = maximum(abs, x_ref) - minimum(abs, x_ref)
    dynamic_range > 0 || throw(DomainError(dynamic_range, "Reference magnitude dynamic range must be positive."))
    x     = vec(x)
    x_ref = vec(x_ref)
    N = length(x)
    x = scale ? image_scale_unchecked(x_ref, x) : x
    
    RMS =  1.0/sqrt(N)*norm(vec(x_ref)-vec(x))
    NRMS = RMS/dynamic_range
    return NRMS
end

"""Real-image mean squared error, with optional fitted intensity scaling (default: true)."""
function HO_MSE(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    check_metric_inputs(x_ref, x)
    N = length(vec(x))
    x = scale ? image_scale_unchecked(x_ref, x) : x
    mse = 1 / N * norm(x_ref-x)^2
    return mse
end

"""Real-image root mean squared error, with optional fitted intensity scaling (default: true)."""
function HO_RMSE(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    mse = HO_MSE(x_ref, x; scale=scale)
    rmse = sqrt(mse)
    return rmse
end

"""Real-image SSIM, with optional fitted intensity scaling (default: true)."""
function HO_SSIM(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    check_metric_inputs(x_ref, x)
    x = scale ? image_scale_unchecked(x_ref, x) : x
    ssim = assess_ssim(x_ref, x)
    return ssim
end



