

function sim_output_dim(obj::HO_Phantom{T}, seq::Sequence, sys::Scanner, sim_method::BlochHighOrder) where {T<:Real}
    _, nCoil = size(obj.csm)
    return (sum(seq.ADC.N), nCoil) #Nt x Ncoils, This should consider the coil info from sys
end

"""Magnetization initialization for Bloch simulation method."""
function initialize_spins_state(obj::HO_Phantom{T}, sim_method::Union{Bloch,BlochHighOrder}) where {T<:Real}
    Nspins = length(obj)
    Mxy = zeros(Complex{T}, Nspins)
    Mz = copy(obj.ρ)
    Xt = Mag{T}(Mxy, Mz)
    return Xt, obj
end

function combine_array_signal!(sig, states, csm)
    sig .= transpose(states) * csm
    return nothing
end

function combine_array_signal!(sig, states::CUDA.CuVecOrMat{Complex{T}},
    csm::AbstractMatrix{Complex{T}}) where {T<:Union{Float32,Float64}}
    if isempty(states)
        fill!(sig, zero(Complex{T}))
        return nothing
    end
    # Bypass generic GPU matmul for CSM row views. cuBLAS requires contiguous
    # columns and a positive leading dimension large enough for every row.
    coils = if csm isa CUDA.StridedCuMatrix && stride(csm, 1) == 1 &&
        stride(csm, 2) >= max(1, size(csm, 1))
        csm
    else
        CUDA.CuArray(csm)
    end
    output = similar(states, size(sig))
    CUDA.CUBLAS.gemmEx!('T', 'N', one(Complex{T}), reshape(states, size(states, 1), :),
        coils, zero(Complex{T}), output)
    sig .= output
    return nothing
end

function run_spin_precession!(p::HO_Phantom{T}, hoseqd::HO_DiscreteSequence{T}, sig::AbstractArray{Complex{T}},
    M::Mag{T}, sim_method::BlochHighOrder) where {T<:Real}
    seq = hoseqd.seqd
    #Simulation
    #Coil sensitivity
    csm = p.csm  # nSpin x nCoil
    #Motion
    xt = p.x .+ p.ux(p.x, p.y, p.z, seq.t')
    yt = p.y .+ p.uy(p.x, p.y, p.z, seq.t')
    zt = p.z .+ p.uz(p.x, p.y, p.z, seq.t')
    #Effective field
    Bzh1 = xt .* hoseqd.h1'
    Bzh2 = yt .* hoseqd.h2'
    Bzh3 = zt .* hoseqd.h3'
    Bzh4 = (xt .* yt) .* hoseqd.h4'
    Bzh5 = (zt .* yt) .* hoseqd.h5'
    Bzh6 = (3zt.^2-(xt.^2 .+ yt.^2 .+ zt.^2)) .* hoseqd.h6'
    Bzh7 = (xt .* zt) .* hoseqd.h7'
    Bzh8 = (xt.^2 .- yt.^2) .* hoseqd.h8'
    Bzh9 = (3 .* yt .* xt.^2 .- yt.^3) .* hoseqd.h9'
    Bzh10 = (xt .* zt .* yt) .* hoseqd.h10'
    Bzh11 = (5 .* zt.^2 .- (xt.^2 .+ yt.^2 .+ zt.^2)) .* yt .* hoseqd.h11'
    Bzh12 = (5 .* zt.^3 .- 3 .* zt .* (xt.^2 .+ yt.^2 .+ zt.^2)) .* hoseqd.h12'
    Bzh13 = (5 .* zt.^2 .- (xt.^2 .+ yt.^2 .+ zt.^2)) .* xt .* hoseqd.h13'
    Bzh14 = (xt.^2 .* zt .- yt.^2 .* zt) .* hoseqd.h14'
    Bzh15 = (xt.^3 .- 3 .* xt .* yt.^2) .* hoseqd.h15'

    Bz0 = sim_method.ho0 ? hoseqd.h0' : 0
    Bz1 = sim_method.ho1 ? Bzh1 .+ Bzh2 .+ Bzh3 : xt .* seq.Gx' .+ yt .* seq.Gy' .+ zt .* seq.Gz'
    Bz2 = sim_method.ho2 ? Bzh4 .+ Bzh5 .+ Bzh6 .+ Bzh7 .+ Bzh8 : 0
    Bz3 = sim_method.ho3 ? Bzh9 .+ Bzh10 .+ Bzh11 .+ Bzh12 .+ Bzh13 .+ Bzh14 .+ Bzh15 : 0

    Bz = sim_method.Δw_precession ? Bz0 .+ Bz1 .+ Bz2 .+ Bz3 .+ p.Δw / T(2π * γ) : Bz0 .+ Bz1 .+ Bz2 .+ Bz3
    #Rotation
    if !isempty(sig)
        ϕ = T(-2π * γ) .* cumtrapz(seq.Δt', Bz)
    else
        ϕ = T(-2π * γ) .* trapz(seq.Δt', Bz)
    end
    #Mxy precession and relaxation, and Mz relaxation
    tp = cumsum(seq.Δt) # t' = t - t0
    dur = sum(seq.Δt)   # Total length, used for signal relaxation
    Mxy = [M.xy M.xy .* exp.(1im .* ϕ .- tp' ./ p.T2)] #This assumes Δw and T2 are constant in time
    M.xy .= Mxy[:, end]
    M.z  .= M.z .* exp.(-dur ./ p.T1) .+ p.ρ .* (1 .- exp.(-dur ./ p.T1))
    #Acquired signal
    if !isempty(sig)
        adc_indices = findall(cpu(seq.ADC[2:end])) .+ 1
        combine_array_signal!(sig, Mxy[:, adc_indices], csm)
    end
    return nothing
end





function run_spin_excitation!(p::HO_Phantom{T}, hoseqd::HO_DiscreteSequence{T}, sig::AbstractArray{Complex{T}},
    M::Mag{T}, sim_method::BlochHighOrder) where {T<:Real}
    seq = hoseqd.seqd
    #Simulation
    # KomaMRI now samples RF frequency and rotating-frame phase separately.
    # High-order terms remain disabled during RF, preserving the legacy model.
    M.xy .*= cis.(-seq.ψ[1:1])
    adc = cpu(seq.ADC)
    sample = 1
    for (i, s) in enumerate(seq)
        #Motion
        xt = p.x .+ p.ux(p.x, p.y, p.z, s.t)
        yt = p.y .+ p.uy(p.x, p.y, p.z, s.t)
        zt = p.z .+ p.uz(p.x, p.y, p.z, s.t)
        #Effective field
        # ΔB_0 = (B_0 - ω_rf/γ), Need to add a component here to model scanner's dB0(xt,yt,zt)
        ΔBz = sim_method.Δw_excitation ? p.Δw ./ T(2π * γ) .- s.Δf ./ T(γ) : 0 .- s.Δf ./ T(γ)    
        Bz = (s.Gx .* xt .+ s.Gy .* yt .+ s.Gz .* zt) .+ ΔBz
        B = sqrt.(abs.(s.B1) .^ 2 .+ abs.(Bz) .^ 2)
        B[B .== 0] .= eps(T)
        #Spinor Rotation
        φ = T(-2π * γ) * (B .* s.Δt) # TODO: Use trapezoidal integration here (?),  this is just Forward Euler
        mul!( Q(φ, s.B1 ./ B, Bz ./ B), M )
        #Relaxation
        M.xy .= M.xy .* exp.(-s.Δt ./ p.T2)
        M.z  .= M.z  .* exp.(-s.Δt ./ p.T1) .+ p.ρ .* (1 .- exp.(-s.Δt ./ p.T1))
        if adc[i+1]
            Mxy = M.xy .* cis.(seq.ψ[i+1:i+1])
            combine_array_signal!(view(sig, sample:sample, :), Mxy, p.csm)
            sample += 1
        end
    end
    M.xy .*= cis.(seq.ψ[end:end])
    return nothing
end
