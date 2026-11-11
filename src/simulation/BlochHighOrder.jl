"""
    BHO = BlochHighOrder(mask::String)
    BHO = BlochHighOrder(mask::String, Δw_excitation::Bool, Δw_precession::Bool)
    BHO = BlochHighOrder(; ho0=true, ho1=true, ho2=true, ho3=true,
                         Δw_excitation=true, Δw_precession=true)

A simulation method for dynamic fields through third spatial order. High-order
fields affect free precession; RF excitation retains the nominal-gradient model.

# Arguments
- `ho0`: (`::Bool`), zeroth order
- `ho1`: (`::Bool`), first order
- `ho2`: (`::Bool`), second order
- `ho3`: (`::Bool`), third order
- `Δw_excitation`: (`::Bool`), whether to include ΔB₀ in excitation
- `Δw_precession`: (`::Bool`), whether to include ΔB₀ in precession

# Returns
- `BHO`: (`::BlochHighOrder`) BlochHighOrder struct
"""
Base.@kwdef struct BlochHighOrder <: SimulationMethod 
    ho0::Bool = true
    ho1::Bool = true
    ho2::Bool = true
    ho3::Bool = true
    Δw_excitation::Bool = true
    Δw_precession::Bool = true
    name::String = string(Int(ho0)) * string(Int(ho1)) * string(Int(ho2)) * string(Int(ho3))
    function BlochHighOrder(ho0, ho1, ho2, ho3, Δw_excitation, Δw_precession, name)
        expected = join(string(Int(Bool(flag))) for flag in (ho0, ho1, ho2, ho3))
        name == expected || throw(ArgumentError("The field-order name must match ho0 through ho3."))
        return new(ho0, ho1, ho2, ho3, Δw_excitation, Δw_precession, name)
    end
end

BlochHighOrder(name::String) = BlochHighOrder(name, true, true)

function BlochHighOrder(name::String, Δw_excitation::Bool, Δw_precession::Bool)
    ncodeunits(name) == 4 && all(flag -> flag in ('0', '1'), name) ||
        throw(ArgumentError("The field-order mask must contain four binary digits, from order zero to three."))
    flags = Tuple(flag == '1' for flag in name)
    return BlochHighOrder(flags..., Δw_excitation, Δw_precession, name)
end

Base.show(io::IO, b::BlochHighOrder) = begin
	print(io, "BlochHighOrder[[$(b.name)] zero order=$(b.ho0) | first order=$(b.ho1) | second order=$(b.ho2) | third order=$(b.ho3) | Δw_excitation=$(b.Δw_excitation) | Δw_precession=$(b.Δw_precession)]")
end
