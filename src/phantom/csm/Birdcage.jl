"""
    csm_Birdcage(N::Int64, nCoil::Int64, relative_radius::Float64)

# Description
    Computes the sensitivity maps for each coils that are arranged in a birdcage manner.

# Arguments
- `N::Int64`: The number of pixels in the x-direction.
- `nCoil::Int64`: The number of coils in the birdcage.
- `relative_radius::Float64`: The relative radius.

# Returns
   A complex array of size (nX, nY, nCoil) containing the sensitivity maps.
# Examples
```julia-repl
julia> smap = csm_Birdcage(300, 300, 32, 1.5)
julia> plot_imgs_subplots(abs.(smap), 4, 8)
```
"""
function csm_Birdcage(nX::Int64, nY::Int64, nCoil::Int64; relative_radius::Float64=1.5, verbose::Bool=false)
    if verbose
        @info nX=nX nY=nY Npartsx=nCoil
    end

    out = zeros(ComplexF64, nX, nY, nCoil)
    for c=0:nCoil-1
        coilx = relative_radius*cos(c*(2*pi/nCoil))
        coily = relative_radius*sin(c*(2*pi/nCoil))
        coil_phase = -c*(2*pi/nCoil)

        for y=0:nY-1
        y_co = (y - (nY/2))/(nY/2) - coily
        for x=0:nX-1
            x_co = (x - (nX/2))/(nX/2) - coilx
            rr = sqrt(x_co^2+y_co^2)
            phi = atan(x_co, -y_co) + coil_phase
            out[x+1,y+1, c+1] = 1/(rr) * exp(1im*phi)
        end
        end
    end
    norm = sqrt.(sum(abs.(out) .^ 2, dims=3))
    return out./norm
end