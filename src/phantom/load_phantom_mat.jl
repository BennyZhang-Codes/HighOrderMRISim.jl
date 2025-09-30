function load_phantom_mat(
    objbrain      :: BrainPhantom ,  # PhantomType
    target_res    :: Tuple{T,T,T} ,  # target resolution
    target_matrix :: Tuple{D,D,D} ,  # target matrix size
    location      :: Tuple{T,T,T} ,  # target location of the sampling center
) where {T <: AbstractFloat, D <: Integer}
    data = MAT.matread(objbrain.matpath)["data"]
    x, y, z = size(data)
    matrix_size = (x, y, z)
    phantom_res = (0.5, 0.5, 0.5) .* 1e-3 # in meters

    x_grid = ((0:matrix_size[1]-1) .- (matrix_size[1]-1)/2) .* phantom_res[1]
    y_grid = ((0:matrix_size[2]-1) .- (matrix_size[2]-1)/2) .* phantom_res[2]
    z_grid = ((0:matrix_size[3]-1) .- (matrix_size[3]-1)/2) .* phantom_res[3]

    x_grid_new = ((0:target_matrix[1]-1) .- (target_matrix[1]-1)/2) .* target_res[1] .+ location[1]
    y_grid_new = ((0:target_matrix[2]-1) .- (target_matrix[2]-1)/2) .* target_res[2] .+ location[2]
    z_grid_new = ((0:target_matrix[3]-1) .- (target_matrix[3]-1)/2) .* target_res[3] .+ location[3]

    # itp = interpolate((x_grid, y_grid, z_grid), data, Gridded(Constant()))

    itp = interpolate(data, BSpline(Constant()), OnGrid())        
    itp = scale(itp, x_grid, y_grid, z_grid)                      

    eitp = extrapolate(itp, 0)
    out = [eitp(x,y,z) for x in x_grid_new, y in y_grid_new, z in z_grid_new]
    out = UInt8.(out)
end