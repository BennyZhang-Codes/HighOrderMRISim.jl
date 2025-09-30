"""
    b0 = b0_real(target_res, target_matrix, location; verbose=false)

# Description
    This function loads a B0 map from the phantom dataset and resamples it 
    to the specified target resolution and matrix size using linear B-spline interpolation.
    The output can be sampled at an arbitrary physical location. Out-of-bounds values 
    are automatically set to 0.  

# Arguments
- `target_res::Tuple{Float64, Float64, Float64}`: target voxel size in meters `(dx, dy, dz)`  
- `target_matrix::Tuple{Int64, Int64, Int64}`: target number of voxels in each dimension `(nx, ny, nz)`  
- `location::Tuple{Float64, Float64, Float64}`: physical coordinates of the sampling center in meters `(x0, y0, z0)`  
- `verbose::Bool=false`: whether to print progress messages  

# Returns
- `b0::Array{Float64, 3}`: the resampled B0 map of shape `target_matrix` with linear interpolation.
  Values outside the original phantom bounds are set to 0.  

# Example
```julia
>>> target_res = (1e-3, 1e-3, 1e-3)
>>> target_matrix = (200, 200, 1)
>>> location = (0.0, 0.0, 0.0)
>>> b0 = b0_real(target_res, target_matrix, location; verbose=true)
````
"""
function b0_real(
    target_res    :: Tuple{T,T,T} ,  # target resolution
    target_matrix :: Tuple{D,D,D} ,  # target matrix size
    location      :: Tuple{T,T,T} ;  # target location of the sampling center
    verbose       :: Bool = false ,
) where {T <: AbstractFloat, D <: Integer}
    if verbose
        @info "loading Real ΔB₀ Map ..."
    end
    data = MAT.matread("$(@__DIR__)/b0_4p0.mat")["b0"]
    x, y, z = size(data)
    matrix_size = (x, y, z)
    phantom_res = (4., 4., 4.) .* 1e-3 # in meters

    x_grid = ((0:matrix_size[1]-1) .- (matrix_size[1]-1)/2) .* phantom_res[1]
    y_grid = ((0:matrix_size[2]-1) .- (matrix_size[2]-1)/2) .* phantom_res[2]
    z_grid = ((0:matrix_size[3]-1) .- (matrix_size[3]-1)/2) .* phantom_res[3]

    x_grid_new = ((0:target_matrix[1]-1) .- (target_matrix[1]-1)/2) .* target_res[1] .+ location[1]
    y_grid_new = ((0:target_matrix[2]-1) .- (target_matrix[2]-1)/2) .* target_res[2] .+ location[2]
    z_grid_new = ((0:target_matrix[3]-1) .- (target_matrix[3]-1)/2) .* target_res[3] .+ location[3]

    itp = interpolate(data, BSpline(Linear()))
    itp = Interpolations.scale(itp, x_grid, y_grid, z_grid)
    eitp = extrapolate(itp, 0)
    out = [eitp(x,y,z) for x in x_grid_new, y in y_grid_new, z in z_grid_new]
    out[isnan.(out)] .= 0
    return out
end