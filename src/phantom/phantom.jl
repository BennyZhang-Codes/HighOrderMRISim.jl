# definition of the abstract PhantomType and BrainPhantom struct
include("PhantomType.jl")
export BrainPhantom

include("SpinProperty.jl")
export SpinProperty

include("load_phantom_mat.jl")
export load_phantom_mat

# include the BrainPhantom module
include("hophantom_brain.jl")
export hophantom_brain

# include the Coil-Sensitivity Map (CSM) module
include("csm/csm.jl")

include("b0map/b0map.jl")

# function to print the information of a Phantom object
function info(s::Phantom)
	print("Phantom[$(s.name) | nSpin=$(length(s.x)) | x=$(minimum(s.x)*1e2):$(maximum(s.x)*1e2) cm | y=$(minimum(s.y)*1e2):$(maximum(s.y)*1e2) cm | z=$(minimum(s.z)*1e2):$(maximum(s.z)*1e2) cm ]")
    print("\n")
end

export info

