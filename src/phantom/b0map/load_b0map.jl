b0_list = [
    :real,    
]
export b0_list

"""
    b0map = load_b0map(type::Symbol, target_res::Tuple{Float64,Float64,Float64}, 
                        target_matrix::Tuple{Int64,Int64,Int64}, 
                        location::Tuple{Float64,Float64,Float64}; verbose=false)

# Description
    Loads the B0 map of the phantom and resamples it to the specified target resolution 
    and matrix size at a given physical location. Linear B-spline interpolation is used, 
    and values outside the original phantom bounds are set to 0.  

# Arguments
- `type::Symbol`: type of B0 map to load; currently only `:real` is supported
- `target_res::Tuple{Float64,Float64,Float64}`: target voxel size in meters `(dx, dy, dz)`  
- `target_matrix::Tuple{Int64,Int64,Int64}`: target number of voxels `(nx, ny, nz)`  
- `location::Tuple{Float64,Float64,Float64}`: physical coordinates of the sampling center `(x0, y0, z0)`  
- `verbose::Bool=false`: whether to print progress messages  

# Returns
- `b0map::Array{Float64,3}`: the resampled B0 map with shape `target_matrix`. Values outside
  the phantom bounds are set to 0.  

# Example
```julia
>>> target_res = (1e-3, 1e-3, 1e-3)
>>> target_matrix = (200, 200, 1)
>>> location = (0.0, 0.0, 0.0)
>>> b0map = load_b0map(:real, target_res, target_matrix, location; verbose=true)
````
"""
function load_b0map(
    type          :: Symbol       , 
    target_res    :: Tuple{T,T,T} , # target resolution
    target_matrix :: Tuple{D,D,D} , # target matrix size
    location      :: Tuple{T,T,T} ; # target location of the sampling center
    verbose       :: Bool = false ,
) where {T <: AbstractFloat, D <: Integer}
    @assert type in b0_list "type must be one of the following: :real"
    if type == :real
        b0 = b0_real(target_res, target_matrix, location; verbose=verbose)
    end
    return b0
end
