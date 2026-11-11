function check_image_values(img)
    isempty(img) && throw(ArgumentError("An image must contain at least one value."))
    all(isfinite, img) || throw(DomainError(img, "Image values must be finite."))
    return nothing
end

"""Min-max normalize a finite image; constant images map to floating-point zeros."""
function normalization(img::AbstractArray{T,2}) where T<:Real
    check_image_values(img)
    lower, upper = extrema(img)
    upper == lower && return zero.(float.(img))
    return (img .- lower) ./ (upper - lower)
end

"""Standardize by population standard deviation; constant images map to zeros."""
function standardization(img::AbstractArray{T,2}) where T<:Real
    check_image_values(img)
    deviation = std(img; corrected=false)
    iszero(deviation) && return zero.(float.(img))
    return (img .- mean(img)) ./ deviation
end

"""
MSE
"""
function mymse(img::AbstractArray{T,2}, ref::AbstractArray{T,2}) where T<:Real
    return mean((img.- ref).^2)
end

"""
    out = get_center_range(x::Int64, x_range::Int64)

# Description
    This function returns the range of the center of the input range.

# Arguments
- `x`: (Int64) center of the range
- `x_range`: (Int64) range of the center

# Returns
- `out`: range of the center

```julia-repl
julia> out = get_center_range(200, 100)
```
"""
function get_center_range(x::Int64, x_range::Int64)
    center = Int64(floor(x/2))
    return center - Int64(floor(x_range/2))+1 : center + Int64(ceil(x_range/2))
end


"""
    out = get_center_crop(image::AbstractArray{T, 3}, out_x::Int64, out_y::Int64) where T <: Real

# Description
    This function crops the center of the input image to the specified size. If the input image is smaller than the output size, the output image will be padded with zeros.

# Arguments
- `image`: (AbstractArray{<:Number, 2}) input image array
- `out_x`: (Int64) output image width
- `out_y`: (Int64) output image height

# Returns
- `out`: croped image array

# Examples
```julia-repl
julia> out = get_center_crop(img, 100, 100)
```
"""
function  get_center_crop(image::AbstractArray{T}, out_x::Int64, out_y::Int64) where T <: Number
    nDim = ndims(image)
    @assert nDim == 2 || nDim == 3 "Input array must be 2D or 3D"
    out = get_center_crop(image, out_x, out_y, size(image, 3))
    return nDim == 2 ? dropdims(out; dims=3) : out
end



"""
    out = get_center_crop(image::AbstractArray{T,3}, out_x::Int64, out_y::Int64, out_z::Int64) where T <: Number

# Description
    This function crops the center of a 3D volume (or 2D image) to the specified size.
    If the input is smaller than the requested size in any dimension, the output will be padded with zeros.

# Arguments
- `image`: (AbstractArray{<:Number, 2 or 3}) input image/volume array  
- `out_x`: (Int64) output size in the x-direction (first dimension)  
- `out_y`: (Int64) output size in the y-direction (second dimension)  
- `out_z`: (Int64) output size in the z-direction (third dimension; use 1 for 2D input)  

# Returns
- `out`: cropped or padded image/volume array  

# Examples
```julia-repl
julia> out2d = get_center_crop(img2d, 100, 100, 1)
julia> out3d = get_center_crop(img3d, 64, 64, 32)
```
"""
function get_center_crop(
    image  :: AbstractArray{T} , 
    out_x  :: Int64            , 
    out_y  :: Int64            , 
    out_z  :: Int64            ,
) where T <: Number
    nDim = ndims(image)
    @assert nDim >= 2 "Input array must be at least 2D"
    @assert out_x > 0 && out_y > 0 && out_z > 0 "Output size must be positive"
    # reshape 2D -> 3D for uniform handling
    image = nDim == 2 ? reshape(image, size(image)..., 1) : image
    M, N, K = size(image)[1:3]

    size_extra = size(image)[4:end]
    out = zeros(T, out_x, out_y, out_z, size_extra...)

    if out_x <= M
        rangex_in = get_center_range(M, out_x)
        rangex_out = 1:out_x
    else
        rangex_in = 1:M
        rangex_out = get_center_range(out_x, M)
    end

    if out_y <= N
        rangey_in = get_center_range(N, out_y)
        rangey_out = 1:out_y
    else
        rangey_in = 1:N
        rangey_out = get_center_range(out_y, N)
    end

    if out_z <= K
        rangez_in = get_center_range(K, out_z)
        rangez_out = 1:out_z
    else
        rangez_in = 1:K
        rangez_out = get_center_range(out_z, K)
    end

    nd = ndims(image)
    ranges_in  = (rangex_in, rangey_in, rangez_in, ntuple(_ -> Colon(), nd-3)...)
    ranges_out = (rangex_out, rangey_out, rangez_out, ntuple(_ -> Colon(), nd-3)...)

    out[ranges_out...] .= image[ranges_in...]
    return  out
end
