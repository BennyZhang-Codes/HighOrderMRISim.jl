csm_list = [
    :real_32cha,
]
export csm_list

"""
    load_csm(:real_32cha, nCoil, target_res, target_matrix, location; verbose=false)

Load the first `nCoil` channels (1 to 32) of the real receive map. The full map
retains its existing normalization; selecting a subset does not renormalize or
otherwise change the selected channels' amplitudes and phases. The default
32-channel case is unchanged. Spatial inputs use meters.
"""
function load_csm(
    type          :: Symbol       , 
    csm_nCoil     :: D            ,  # number of coils
    target_res    :: Tuple{T,T,T} ,  # target resolution
    target_matrix :: Tuple{D,D,D} ,  # target matrix size
    location      :: Tuple{T,T,T} ;  # target location of the sampling center
    verbose       :: Bool = false ,
) where {T <: AbstractFloat, D <: Integer}
    @assert type in csm_list "type must be one of the following: :real_32cha"
    1 <= csm_nCoil <= 32 || throw(ArgumentError("The real receive map supports 1 to 32 coils."))
    if type == :real_32cha
        csm = csm_Real_32cha(target_res, target_matrix, location; verbose=verbose)
    end
    return csm_nCoil == size(csm, 4) ? csm : csm[:, :, :, 1:csm_nCoil]
end
