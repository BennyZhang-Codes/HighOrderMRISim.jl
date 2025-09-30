
function hophantom_brain(
    objbrain  :: BrainPhantom                 , # PhantomType
    target_res    :: Tuple{T,T,T}             , # target resolution
    target_matrix :: Tuple{D,D,D}             , # target matrix size
    location      :: Tuple{T,T,T}             , # target location of the sampling center
    offcenter     :: Tuple{T,T,T}             ; # offcenter shift of the grid

    b0_type   :: Symbol         = :fat        , # load B0 map
    csm_type  :: Symbol         = :real_32cha , # coil type
    csm_nCoil :: Int64          = 32          ,  
    verbose   :: Bool           = false       ,
) :: HO_Phantom where {T <: AbstractFloat, D <: Integer}

    class = load_phantom_mat(objbrain, target_res, target_matrix, location)

    nX, nY, nZ  = target_matrix
    fov         = target_matrix .* target_res;

    # Generate 3D grid
    xyz = generate_grid(collect(fov), collect(target_matrix))
    x = xyz[:, :, :, 1] .+ offcenter[1]; 
    y = xyz[:, :, :, 2] .+ offcenter[2]; 
    z = xyz[:, :, :, 3] .+ offcenter[3]; 

    # Define spin property vectors
    T1, T2, T2s, ρ = SpinProperty_1p5T(class)

    # Define B0map vectors
    if b0_type == :fat
        Δw_fat = -220*2π
        Δw = (class.==93 )*Δw_fat .+ #FAT1
            (class.==209)*Δw_fat    #FAT2
    else    
        fieldmap = load_b0map(b0_type, target_res, target_matrix, location; verbose=verbose)
        Δw = fieldmap*2π
    end

    # Define Coil-Sensitivity Map (CSM) vectors
    csm  = load_csm(csm_type, csm_nCoil, target_res, target_matrix, location; verbose=verbose)
    nCha = size(csm)[end]

    # Define and return the Phantom struct
    obj = HO_Phantom{Float64}(
        name = "brain3D_$(nX)x$(nY)x$(nZ)_Cha$(nCha)",
        x   =    x[ρ.!=0],
        y   =    y[ρ.!=0],
        z   =    z[ρ.!=0],
        ρ   =    ρ[ρ.!=0],
        T1  =   T1[ρ.!=0],
        T2  =   T2[ρ.!=0],
        T2s =  T2s[ρ.!=0],
        Δw  =   Δw[ρ.!=0],
        csm =  csm[ρ.!=0,:],
    )
	return obj
end
  


function hophantom_brain(
    objbrain  :: BrainPhantom                 , # PhantomType
    key       :: Symbol                       ,
    target_res    :: Tuple{T,T,T}             , # target resolution
    target_matrix :: Tuple{D,D,D}             , # target matrix size
    location      :: Tuple{T,T,T}             , # target location of the sampling center
    offcenter     :: Tuple{T,T,T}             ; # offcenter shift of the grid

    b0_type   :: Symbol         = :fat        , # load B0 map
    csm_type  :: Symbol         = :real_32cha , # coil type
    csm_nCoil :: Int64          = 32          ,  
    verbose   :: Bool           = false       ,
) ::AbstractArray where {T <: AbstractFloat, D <: Integer}
    @assert key in [:ρ, :T2, :T2s, :T1, :class, :headmask, :brainmask, :Δw, :csm, :grid] "key must be ρ, T2, T2s, T1, Δw, raw, headmask or brainmask"

    if key == :grid
        # Generate 3D grid
        xyz = generate_grid(collect(target_matrix .* target_res), collect(target_matrix))
        xyz[:, :, :, 1] .+= offcenter[1]; 
        xyz[:, :, :, 2] .+= offcenter[2]; 
        xyz[:, :, :, 3] .+= offcenter[3]; 
        return xyz
    end

    class = load_phantom_mat(objbrain, target_res, target_matrix, location)
    # Define spin property vectors
    T1, T2, T2s, ρ = SpinProperty_1p5T(class)

    if key == :ρ
        img = ρ
    elseif key == :T2
        img = T2
    elseif key == :T2s
        img = T2s
    elseif key == :T1
        img = T1
    elseif key == :class
        img = class
    elseif key == :headmask
        img = (class.>0)*1 #All
    elseif key == :brainmask
        img = 
        (class.==23)*1 .+ #CSF
        (class.==46)*1 .+ #GM
        (class.==70)*1 .+ #WM
        (class.==93)*0 .+ #FAT1
        (class.==116)*0 .+ #MUSCLE
        (class.==139)*0 .+ #SKIN/MUSCLE
        (class.==162)*0 .+ #SKULL
        (class.==185)*0 .+ #VESSELS
        (class.==209)*0 .+ #FAT2
        (class.==232)*0 .+ #DURA
        (class.==255)*0 #MARROW
    elseif key == :Δw
        # Define B0map vectors
        if b0_type == :fat
            Δw_fat = -220*2π
            Δw = (class.==93 )*Δw_fat .+ #FAT1
                (class.==209)*Δw_fat    #FAT2
        else    
            fieldmap = load_b0map(b0_type, target_res, target_matrix, location; verbose=verbose)
            Δw = fieldmap*2π
        end
        img = Δw
    elseif key == :csm
        # Define Coil-Sensitivity Map (CSM) vectors
        csm  = load_csm(csm_type, csm_nCoil, target_res, target_matrix, location; verbose=verbose)
        img = csm
    end
    return img
end
