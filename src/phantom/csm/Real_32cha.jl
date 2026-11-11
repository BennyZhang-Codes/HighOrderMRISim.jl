"""
    csm = csm_Real_32cha(nX, nY; verbose=false)

# Description
    This function loads the Real Coil-Sensitivity Map (32 channels) and resizes it to the desired size (nX, nY, 32).

# Arguments
- `nX::Int64`: number of voxels in the x-direction
- `nY::Int64`: number of voxels in the y-direction
- `verbose::Bool=false`: whether to print progress messages

# Returns
- `smap::Array{ComplexF64, 3}`: the coil-sensitivity map

```julia
>>> smap = csm_Real_32cha(200, 200)
```
"""
function csm_Real_32cha(nX::Int64, nY::Int64; verbose::Bool=false)
    if verbose
        @info "loading Real Coil-Sensitivity Map (32 channels)..."
    end
    sensitivity = MAT.matread("$(@__DIR__)/coilsensmap_32cha.mat")["coilsensmap_32cha"]
    out = imresize(sensitivity, (nX, nY, 32))
    norm = sqrt.(sum(abs.(out) .^ 2, dims=3))
    out = out./ norm
    out[isnan.(out)] .= 0 + 0im
    return out
end

"""
    csm = csm_Real_32cha(target_res, target_matrix, location; verbose=false)

# Description
    Loads the Real Coil-Sensitivity Map (32 channels) of the phantom and resamples it 
    to the specified target resolution and matrix size at a given physical location.
    Linear B-spline interpolation is applied to the spatial dimensions, while the coil 
    channel dimension is preserved. Edge samples extend to the outer faces of their
    source voxels (half a source voxel beyond the outermost centres). Coordinates
    beyond that physical FOV are set to zero. Nonzero receive vectors are normalized
    to unit summed squared magnitude; source zeros remain zero.

# Arguments
- `target_res::Tuple{Float64,Float64,Float64}`: target voxel size in meters `(dx, dy, dz)`  
- `target_matrix::Tuple{Int64,Int64,Int64}`: target number of voxels `(nx, ny, nz)`  
- `location::Tuple{Float64,Float64,Float64}`: physical coordinates of the sampling center `(x0, y0, z0)`  
- `verbose::Bool=false`: whether to print progress messages  

# Returns
- `smap::Array{ComplexF64,4}`: the resampled and normalized coil-sensitivity map with shape 
  `(nx, ny, nz, 32)`  

# Example
```julia
>>> target_res = (1e-3, 1e-3, 1e-3)
>>> target_matrix = (200, 200, 70)
>>> location = (0.0, 0.0, 0.0)
>>> smap = csm_Real_32cha(target_res, target_matrix, location; verbose=true)
```
"""
function csm_Real_32cha(
    target_res    :: Tuple{T,T,T} ,  # target resolution
    target_matrix :: Tuple{D,D,D} ,  # target matrix size
    location      :: Tuple{T,T,T} ;  # target location of the sampling center
    verbose       :: Bool = false ,
) where {T <: AbstractFloat, D <: Integer}
    if verbose
        @info "loading Real Coil-Sensitivity Map (32 channels)..."
    end
    data = MAT.matread("$(@__DIR__)/csm_4p0.mat")["csm"]
    return _resample_csm(data, (4e-3, 4e-3, 4e-3), target_res, target_matrix, location)
end

# Clamp only within the outer source voxel cells, not arbitrarily far outside
# the measured map. Keeping resampling separate from file loading permits small
# analytic boundary tests without replacing the bundled receive-map data.
function _resample_csm(
    data::AbstractArray{<:Number,4}, source_res::NTuple{3,<:Real},
    target_res::NTuple{3,T}, target_matrix::NTuple{3,<:Integer}, location::NTuple{3,T},
) where {T<:AbstractFloat}
    source_axes = ntuple(3) do axis
        n = size(data, axis)
        ((0:n-1) .- (n-1)/2) .* source_res[axis]
    end
    target_axes = ntuple(3) do axis
        n = target_matrix[axis]
        ((0:n-1) .- (n-1)/2) .* target_res[axis] .+ location[axis]
    end
    valid = ntuple(axis -> abs.(target_axes[axis]) .<= T(size(data, axis)*source_res[axis]/2), 3)
    coordinates = ntuple(axis -> clamp.(target_axes[axis],
        first(source_axes[axis]), last(source_axes[axis])), 3)
    out = zeros(Complex{T}, target_matrix..., size(data, 4))

    for c in axes(data, 4)
        real_itp = interpolate(real(data[:,:,:,c]), BSpline(Linear()))
        imag_itp = interpolate(imag(data[:,:,:,c]), BSpline(Linear()))
        real_itp = scale(real_itp, source_axes...)
        imag_itp = scale(imag_itp, source_axes...)

        for zi in eachindex(target_axes[3]), yi in eachindex(target_axes[2]), xi in eachindex(target_axes[1])
            if valid[1][xi] && valid[2][yi] && valid[3][zi]
                point = (coordinates[1][xi], coordinates[2][yi], coordinates[3][zi])
                out[xi, yi, zi, c] = real_itp(point...) + 1im*imag_itp(point...)
            end
        end
    end

    coil_norm = sqrt.(sum(abs.(out) .^ 2; dims=4))
    out ./= ifelse.(iszero.(coil_norm), one(T), coil_norm)
    return out
end
