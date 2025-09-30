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
    channel dimension is preserved. Out-of-bounds values are set to 0, and the resulting 
    coil-sensitivity map is normalized at each voxel such that the sum of squared magnitudes 
    across channels equals 1.  

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
    x, y, z, nCha = size(data)

    matrix_size = (x, y, z)
    phantom_res = (4., 4., 4.) .* 1e-3 # in meters

    x_grid = ((0:matrix_size[1]-1) .- (matrix_size[1]-1)/2) .* phantom_res[1]
    y_grid = ((0:matrix_size[2]-1) .- (matrix_size[2]-1)/2) .* phantom_res[2]
    z_grid = ((0:matrix_size[3]-1) .- (matrix_size[3]-1)/2) .* phantom_res[3]

    x_grid_new = ((0:target_matrix[1]-1) .- (target_matrix[1]-1)/2) .* target_res[1] .+ location[1]
    y_grid_new = ((0:target_matrix[2]-1) .- (target_matrix[2]-1)/2) .* target_res[2] .+ location[2]
    z_grid_new = ((0:target_matrix[3]-1) .- (target_matrix[3]-1)/2) .* target_res[3] .+ location[3]

    out = Array{Complex{T}}(undef, target_matrix..., nCha)

    for c in 1:nCha
        real_itp = interpolate(real(data[:,:,:,c]), BSpline(Linear()), OnGrid())
        imag_itp = interpolate(imag(data[:,:,:,c]), BSpline(Linear()), OnGrid())
        real_itp = scale(real_itp, x_grid, y_grid, z_grid)
        imag_itp = scale(imag_itp, x_grid, y_grid, z_grid)
        real_itp = extrapolate(real_itp, 0)
        imag_itp = extrapolate(imag_itp, 0)

        for xi in eachindex(x_grid_new), yi in eachindex(y_grid_new), zi in eachindex(z_grid_new)
            out[xi, yi, zi, c] = real_itp(x_grid_new[xi], y_grid_new[yi], z_grid_new[zi]) +
                                1im * imag_itp(x_grid_new[xi], y_grid_new[yi], z_grid_new[zi])
        end
    end

    norm = sqrt.(sum(abs.(out) .^ 2, dims=4))
    out .= out./ norm
    out[isnan.(out)] .= 0 + 0im
    return out
end