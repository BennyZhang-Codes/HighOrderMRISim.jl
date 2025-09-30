csm_list = [
    :real_32cha,
]
export csm_list

function load_csm(
    type          :: Symbol       , 
    csm_nCoil     :: D            ,  # number of coils
    target_res    :: Tuple{T,T,T} ,  # target resolution
    target_matrix :: Tuple{D,D,D} ,  # target matrix size
    location      :: Tuple{T,T,T} ;  # target location of the sampling center
    verbose       :: Bool = false ,
) where {T <: AbstractFloat, D <: Integer}
    @assert type in csm_list "type must be one of the following: :real_32cha"
    if type == :real_32cha
        csm = csm_Real_32cha(target_res, target_matrix, location; verbose=verbose)
    end
    return csm
end