function load_phantom_mat(
    objbrain      :: BrainPhantom ,  # PhantomType
    target_res    :: Tuple{T,T,T} ,  # target resolution
    target_matrix :: Tuple{D,D,D} ,  # target matrix size
    location      :: Tuple{T,T,T} ,  # target location of the sampling center
) where {T <: AbstractFloat, D <: Integer}
    all(>(0), target_matrix) || throw(ArgumentError("Target matrix dimensions must be positive."))
    all(value -> isfinite(value) && value > 0, target_res) ||
        throw(ArgumentError("Target voxel sizes must be finite and positive."))
    all(isfinite, location) || throw(ArgumentError("The sampling location must be finite."))
    data = MAT.matread(objbrain.matpath)["data"]
    x, y, z = size(data)
    matrix_size = (x, y, z)
    phantom_res = (objbrain.x, objbrain.y, objbrain.z) .* 1e-3 # millimeters to meters
    all(value -> isfinite(value) && value > 0, phantom_res) ||
        throw(ArgumentError("Source phantom voxel sizes must be finite and positive."))

    x_grid = ((0:matrix_size[1]-1) .- (matrix_size[1]-1)/2) .* phantom_res[1]
    y_grid = ((0:matrix_size[2]-1) .- (matrix_size[2]-1)/2) .* phantom_res[2]
    z_grid = ((0:matrix_size[3]-1) .- (matrix_size[3]-1)/2) .* phantom_res[3]

    x_grid_new = ((0:target_matrix[1]-1) .- (target_matrix[1]-1)/2) .* target_res[1] .+ location[1]
    y_grid_new = ((0:target_matrix[2]-1) .- (target_matrix[2]-1)/2) .* target_res[2] .+ location[2]
    z_grid_new = ((0:target_matrix[3]-1) .- (target_matrix[3]-1)/2) .* target_res[3] .+ location[3]

    itp = interpolate(data, BSpline(Constant()))
    itp = scale(itp, x_grid, y_grid, z_grid)                      
    source_axes = (x_grid, y_grid, z_grid)
    target_axes = (x_grid_new, y_grid_new, z_grid_new)
    # Nearest-neighbor labels occupy full voxel cells, not just their centers.
    # Clamp within the physical half-voxel boundary and keep exterior samples zero.
    valid = ntuple(axis -> abs.(target_axes[axis]) .<= T(matrix_size[axis]*phantom_res[axis]/2), 3)
    coordinates = ntuple(axis -> clamp.(target_axes[axis],
        first(source_axes[axis]), last(source_axes[axis])), 3)
    out = [valid[1][xi] && valid[2][yi] && valid[3][zi] ?
        itp(coordinates[1][xi], coordinates[2][yi], coordinates[3][zi]) : zero(eltype(data))
        for xi in eachindex(x_grid_new), yi in eachindex(y_grid_new), zi in eachindex(z_grid_new)]
    out = UInt8.(out)
end
