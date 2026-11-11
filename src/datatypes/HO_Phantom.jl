# Recognize the default static model by identity, never by sampling a custom function.
no_displacement(x, y, z, t) = 0

"""
    obj = HO_Phantom(; x, kwargs...)
    obj = HO_Phantom{T}(; x, kwargs...)
    obj = HO_Phantom(name, x, y, z, ρ, T1, T2, T2s, Δw, Dλ1, Dλ2, Dθ, ux, uy, uz, csm)

# Description
	The HO_Phantom struct. Most of its field names are vectors, with each element associated with
a property value representing a spin. This struct serves as an input for the simulation.

Omitting `csm` creates one ideal receive coil with sensitivity `1 + 0im` at
every spin: unit amplitude, zero additional phase, and no signal normalization.
This is a receive-sensitivity default, not a transmit B1 model. For Float32 or
Float64 coordinates, omitted numeric arrays preserve that precision. An explicitly supplied
CSM retains the existing constructor behavior; it is not normalized or replaced.

# Arguments
- `name`: (`::String`) phantom name
- `x`: (`::AbstractVector{T<:Real}`, `[m]`) spin x-position vector
- `y`: (`::AbstractVector{T<:Real}`, `[m]`) spin y-position vector
- `z`: (`::AbstractVector{T<:Real}`, `[m]`) spin z-position vector
- `ρ`: (`::AbstractVector{T<:Real}`) spin proton density vector
- `T1`: (`::AbstractVector{T<:Real}`, `[s]`) spin T1 parameter vector
- `T2`: (`::AbstractVector{T<:Real}`, `[s]`) spin T2 parameter vector
- `T2s`: (`::AbstractVector{T<:Real}`, `[s]`) spin T2s parameter vector
- `Δw`: (`::AbstractVector{T<:Real}`, `[rad/s]`) spin off-resonance parameter vector
- `Dλ1`: (`::AbstractVector{T<:Real}`) spin Dλ1 (diffusion) parameter vector
- `Dλ2`: (`::AbstractVector{T<:Real}`) spin Dλ2 (diffusion) parameter vector
- `Dθ`: (`::AbstractVector{T<:Real}`) spin Dθ (diffusion) parameter vector
- `ux`: (`::Function`) displacement field in the x-axis
- `uy`: (`::Function`) displacement field in the y-axis
- `uz`: (`::Function`) displacement field in the z-axis
- `csm`: (`::AbstractArray{Complex{T}, 2}`) receive-sensitivity matrix of shape
  `(length(x), nCoil)`; the number of coils is `size(csm, 2)`.

# Returns
- `obj`: (`::HO_Phantom`) HO_Phantom struct

# Examples
```julia-repl
julia> obj = HO_Phantom(x=[0.0])
julia> obj.ρ
```
"""
Base.@kwdef mutable struct HO_Phantom{T<:Real}
    name::String = "spins"
	x::AbstractVector{T}
	y::AbstractVector{T} = zeros(eltype(x), size(x))
	z::AbstractVector{T} = zeros(eltype(x), size(x))
	ρ::AbstractVector{T} = ones(eltype(x), size(x))
	# Keep the finite default representable before a typed constructor converts it.
	T1::AbstractVector{T} = fill(1_000_000 * one(promote_type(eltype(x), Float32)), size(x))
	T2::AbstractVector{T} = fill(1_000_000 * one(promote_type(eltype(x), Float32)), size(x))
	T2s::AbstractVector{T} = fill(1_000_000 * one(promote_type(eltype(x), Float32)), size(x))
	#Off-resonance related
	Δw::AbstractVector{T} = zeros(eltype(x), size(x))
	#χ::Vector{SusceptibilityModel}
	#Diffusion
	Dλ1::AbstractVector{T} = zeros(eltype(x), size(x))
	Dλ2::AbstractVector{T} = zeros(eltype(x), size(x))
	Dθ::AbstractVector{T} = zeros(eltype(x), size(x))
	#Diff::Vector{DiffusionModel}  #Diffusion map
	#Motion
	ux::Function = no_displacement
	uy::Function = no_displacement
	uz::Function = no_displacement
    #Coil-Sensitivity
    csm::AbstractArray{Complex{T}, 2} = ones(Complex{eltype(x)}, length(x), 1)
end

@functor HO_Phantom
Base.show(io::IO, s::HO_Phantom) = begin
	nSpin, nCoil = size(s.csm)
	if isempty(s.x)
		print(io, "HO_Phantom[$(s.name) | nCoil=$(nCoil) | nSpin=$(nSpin) | empty ]")
		return nothing
	end
	x_min, x_max = round(minimum(s.x)*1e2, digits=2), round(maximum(s.x)*1e2, digits=2)
	y_min, y_max = round(minimum(s.y)*1e2, digits=2), round(maximum(s.y)*1e2, digits=2)
	z_min, z_max = round(minimum(s.z)*1e2, digits=2), round(maximum(s.z)*1e2, digits=2)
	print(io, "HO_Phantom[$(s.name) | nCoil=$(nCoil) | nSpin=$(nSpin) | x=$(x_min):$(x_max) cm | y=$(y_min):$(y_max) cm | z=$(z_min):$(z_max) cm ]")
end

"""Size and length of a phantom"""
size(x::HO_Phantom) = size(x.ρ)
Base.length(x::HO_Phantom) = length(x.ρ)
# To enable to iterate and broadcast over the HO_Phantom
Base.iterate(x::HO_Phantom) = (x[1], 2)
Base.iterate(x::HO_Phantom, i::Integer) = (i <= length(x)) ? (x[i], i+1) : nothing
Base.lastindex(x::HO_Phantom) = length(x)
Base.getindex(x::HO_Phantom, i::Integer) = x[i:i]

"""Compare two phantoms"""
Base.isapprox(obj1::HO_Phantom, obj2::HO_Phantom)  = begin
    obj1.x     ≈ obj2.x    &&
    obj1.y     ≈ obj2.y    &&
    obj1.z     ≈ obj2.z    &&
    obj1.ρ     ≈ obj2.ρ    &&
    obj1.T1    ≈ obj2.T1   &&
    obj1.T2    ≈ obj2.T2   &&
    obj1.T2s   ≈ obj2.T2s  &&
    obj1.Δw    ≈ obj2.Δw   &&
    obj1.Dλ1   ≈ obj2.Dλ1  &&
    obj1.Dλ2   ≈ obj2.Dλ2  &&
    obj1.Dθ    ≈ obj2.Dθ   &&
	obj1.csm  ≈ obj2.csm 
end

"""
Separate object spins in a sub-group
"""
Base.getindex(obj::HO_Phantom, p::AbstractRange) = begin
	HO_Phantom(name=obj.name,
			x=obj.x[p],
			y=obj.y[p],
			z=obj.z[p],
			ρ=obj.ρ[p],
			T1=obj.T1[p],
			T2=obj.T2[p],
			T2s=obj.T2s[p],
			Δw=obj.Δw[p],
			#Diff=obj.Diff[p], #TODO!
			Dλ1=obj.Dλ1[p],
			Dλ2=obj.Dλ2[p],
			Dθ=obj.Dθ[p],
			#Χ=obj.Χ[p], #TODO!
			ux=obj.ux,
			uy=obj.uy,
			uz=obj.uz,
			csm=obj.csm[p,:]
			)
end

"""Separate object spins in a sub-group (lightweigth)."""
Base.view(obj::HO_Phantom, p::AbstractRange) = begin
	@views HO_Phantom(name=obj.name,
			x=obj.x[p],
			y=obj.y[p],
			z=obj.z[p],
			ρ=obj.ρ[p],
			T1=obj.T1[p],
			T2=obj.T2[p],
			T2s=obj.T2s[p],
			Δw=obj.Δw[p],
			#Diff=obj.Diff[p], #TODO!
			Dλ1=obj.Dλ1[p],
			Dλ2=obj.Dλ2[p],
			Dθ=obj.Dθ[p],
			#Χ=obj.Χ[p], #TODO!
			ux=obj.ux,
			uy=obj.uy,
			uz=obj.uz,
			csm=obj.csm[p,:]
			)
end

"""Addition of phantoms"""
+(s1::HO_Phantom,s2::HO_Phantom) = begin
	HO_Phantom(name=s1.name*"+"*s2.name,
		x=[s1.x;s2.x],
		y=[s1.y;s2.y],
		z=[s1.z;s2.z],
		ρ=[s1.ρ;s2.ρ],
		T1=[s1.T1;s2.T1],
		T2=[s1.T2;s2.T2],
		T2s=[s1.T2s;s2.T2s],
		Δw=[s1.Δw;s2.Δw],
		#Diff=obj.Diff[p], #TODO!
		Dλ1=[s1.Dλ1;s2.Dλ1],
		Dλ2=[s1.Dλ2;s2.Dλ2],
		Dθ=[s1.Dθ;s2.Dθ],
		#Χ=obj.Χ[p], #TODO!
		ux=s1.ux,
		uy=s1.uy,
		uz=s1.uz,
		csm=[s1.csm;s2.csm]
	)
end

"""Scalar multiplication of a phantom"""
*(α::Real,obj::HO_Phantom) = begin
	HO_Phantom(name=obj.name,
		x=obj.x,
		y=obj.y,
		z=obj.z,
		ρ=α*obj.ρ, #Only affects the proton density
		T1=obj.T1,
		T2=obj.T2,
		T2s=obj.T2s,
		Δw=obj.Δw,
		#Diff=obj.Diff[p], #TODO!
		Dλ1=obj.Dλ1,
		Dλ2=obj.Dλ2,
		Dθ=obj.Dθ,
		#Χ=obj.Χ[p], #TODO!
		ux=obj.ux,
		uy=obj.uy,
		uz=obj.uz,
		csm=obj.csm
	)
end

