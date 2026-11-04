# Concrete arrays cross the kernel boundary; the public legacy containers stay
# unchanged. Fused precession uses partial sums; its ADC/packed buffers only
# need RF capacity. GEMM keeps full ADC capacity and remains the default.
const HIGH_ORDER_COIL_COMBINATIONS = ("gemm", "reduction", "fused_warp", "fused_tiled")

struct HighOrderGPUWorkspace{A,C,P,F,B}
    adc_states::A
    coil_signal::C
    coil_partials::P
    fusion::F
    backend::B
    groupsize::Int
end

function HighOrderGPUWorkspace(obj::HO_Phantom{T}, max_adc, backend;
                               groupsize=256, coil_combination="gemm",
                               adc_tile=8, coil_tile=1, max_rf_adc=max_adc) where {T}
    max_adc isa Integer && max_adc >= 0 ||
        throw(ArgumentError("The maximum ADC count must be a nonnegative integer."))
    max_rf_adc isa Integer && 0 <= max_rf_adc <= max_adc ||
        throw(ArgumentError("The RF ADC count must be an integer between zero and max_adc."))
    coil_combination in HIGH_ORDER_COIL_COMBINATIONS ||
        throw(ArgumentError("Unknown coil_combination: $coil_combination."))
    fused = coil_combination in ("fused_warp", "fused_tiled")
    !fused || (groupsize >= 32 && groupsize <= 1024 && groupsize % 32 == 0) ||
        throw(ArgumentError("CUDA fused groupsize must be a multiple of 32, at most 1024."))
    A, C = coil_combination == "fused_warp" ? (1, 1) : (adc_tile, coil_tile)
    if fused
        A isa Integer && C isa Integer && A in (1, 2, 4, 8) && C in (1, 2, 4) ||
            throw(ArgumentError("Fused ADC/coil tiles must be selected from 1/2/4/8 and 1/2/4."))
        A, C = Int(A), Int(C)
        groupsize * A * sizeof(Complex{T}) <=
            CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) ||
            throw(ArgumentError("The fused ADC tile exceeds the device shared-memory limit."))
    end
    buffer_adc = fused ? max_rf_adc : max_adc
    states = similar(obj.x, Complex{T}, length(obj), buffer_adc)
    partials = coil_combination == "reduction" ?
        high_order_coil_partials(states, size(obj.csm, 2)) : nothing
    fusion = fused ? (
        partials=similar(obj.x, Complex{T}, cld(length(obj), groupsize), max_adc * size(obj.csm, 2)),
        adc_tile=Val(A), coil_tile=Val(C),
    ) : nothing
    return HighOrderGPUWorkspace(
        states, coil_combination != "reduction" ?
            similar(obj.x, Complex{T}, buffer_adc * size(obj.csm, 2)) : nothing,
        partials, fusion, backend, groupsize,
    )
end

has_static_coordinates(obj::HO_Phantom) =
    obj.ux === no_displacement && obj.uy === no_displacement && obj.uz === no_displacement

@inline function high_order_basis(x::T, y::T, z::T) where {T}
    x², y², z² = x*x, y*y, z*z
    r² = x² + y² + z²
    return (
        one(T), x, y, z, x*y, z*y, 3z²-r², x*z, x²-y²,
        3y*x²-y*y², x*z*y, (5z²-r²)*y, 5z*z²-3z*r²,
        (5z²-r²)*x, x²*z-y²*z, x*x²-3x*y²,
    )
end

@inline function high_order_field(basis, coefficients, gradients, Δw::T, node,
                                ::Val{ORDERS}) where {T,ORDERS}
    ho0, ho1, ho2, ho3, off_resonance = ORDERS
    Bz0 = ho0 ? coefficients[1][node] : zero(T)
    Bz1 = if ho1
        basis[2]*coefficients[2][node] + basis[3]*coefficients[3][node] +
        basis[4]*coefficients[4][node]
    else
        basis[2]*gradients[1][node] + basis[3]*gradients[2][node] +
        basis[4]*gradients[3][node]
    end
    Bz2 = ho2 ? basis[5]*coefficients[5][node] + basis[6]*coefficients[6][node] +
        basis[7]*coefficients[7][node] + basis[8]*coefficients[8][node] +
        basis[9]*coefficients[9][node] : zero(T)
    Bz3 = ho3 ? basis[10]*coefficients[10][node] + basis[11]*coefficients[11][node] +
        basis[12]*coefficients[12][node] + basis[13]*coefficients[13][node] +
        basis[14]*coefficients[14][node] + basis[15]*coefficients[15][node] +
        basis[16]*coefficients[16][node] : zero(T)
    Bz = Bz0 + Bz1 + Bz2 + Bz3
    return off_resonance ? Bz + Δw / T(2π * γ) : Bz
end

@kernel function high_order_precession_kernel!(
    Mxy, Mz, adc_states, x, y, z, ρ, T1, T2, Δw,
    coefficients, gradients, Δt, adc, orders,
)
    spin = @index(Global, Linear)
    T = eltype(x)
    @inbounds begin
        basis = high_order_basis(x[spin], y[spin], z[spin])
        Bz_prev = high_order_field(basis, coefficients, gradients, Δw[spin], 1, orders)
        initial = Mxy[spin]
        ϕ, t = zero(T), zero(T)
        sample = 0
        for node in 2:length(adc)
            dt = Δt[node-1]
            t += dt
            Bz_next = high_order_field(basis, coefficients, gradients, Δw[spin], node, orders)
            ϕ += T(-π * γ) * (Bz_prev + Bz_next) * dt
            if adc[node]
                sample += 1
                adc_states[spin, sample] = initial * exp(-t / T2[spin]) * cis(ϕ)
            end
            Bz_prev = Bz_next
        end
        Mxy[spin] = initial * exp(-t / T2[spin]) * cis(ϕ)
        E1 = exp(-t / T1[spin])
        Mz[spin] = Mz[spin]*E1 + ρ[spin]*(one(T)-E1)
    end
end

function run_high_order_precession!(workspace, obj, hoseqd, sig, M, method)
    if !isnothing(workspace.fusion) && !isempty(sig)
        run_fused_high_order_precession!(workspace, obj, hoseqd, sig, M, method)
        return nothing
    end
    seq = hoseqd.seqd
    coefficients = ntuple(i -> getfield(hoseqd, Symbol("h", i-1)), 16)
    orders = Val((method.ho0, method.ho1, method.ho2, method.ho3, method.Δw_precession))
    high_order_precession_kernel!(workspace.backend, workspace.groupsize)(
        M.xy, M.z, workspace.adc_states, obj.x, obj.y, obj.z,
        obj.ρ, obj.T1, obj.T2, obj.Δw, coefficients,
        (seq.Gx, seq.Gy, seq.Gz), seq.Δt, seq.ADC, orders;
        ndrange=length(obj),
    )
    combine_high_order_signal!(sig, workspace, obj.csm)
    return nothing
end

function combine_high_order_signal!(sig, workspace, csm)
    isempty(sig) && return nothing
    states = view(workspace.adc_states, :, 1:size(sig, 1))
    if !isnothing(workspace.coil_partials)
        reduce_high_order_signal!(sig, states, csm, workspace.coil_partials, workspace.backend)
        return nothing
    end
    # Keep GEMM output dense: a strided legacy signal view can select generic
    # GPU matmul instead of cuBLAS. Repack a reusable buffer, then copy once.
    output = reshape(view(workspace.coil_signal, 1:length(sig)), size(sig))
    # The existing CSM convention uses transpose, not complex conjugation.
    LinearAlgebra.mul!(output, transpose(states), csm)
    sig .= output
    return nothing
end

# Same hard-pulse spinor and relaxation order as the compatibility reference.
@inline function rotate_high_order_spin(Mxy::Complex{T}, Mz::T, B1::Complex{T},
                                       Bz::T, dt::T) where {T}
    B = sqrt(abs2(B1) + abs2(Bz))
    B = iszero(B) ? eps(T) : B
    φ = T(-2π * γ) * (B * dt)
    s, c = sincos(φ / T(2))
    α = complex(c, -(Bz / B) * s)
    β = complex(zero(T), -one(T)) * (B1 / B) * s
    xy_next = T(2)*conj(α)*β*Mz + conj(α)^2*Mxy - β^2*conj(Mxy)
    z_next = (abs2(α)-abs2(β))*Mz - T(2)*real(α*β*conj(Mxy))
    return xy_next, z_next
end

@kernel function high_order_excitation_kernel!(
    Mxy, Mz, adc_states, x, y, z, ρ, T1, T2, Δw,
    Gx, Gy, Gz, B1, Δf, ψ, Δt, adc, ::Val{OFF_RESONANCE},
) where {OFF_RESONANCE}
    spin = @index(Global, Linear)
    T = eltype(x)
    @inbounds begin
        mx = Mxy[spin] * cis(-ψ[1])
        mz = Mz[spin]
        off = OFF_RESONANCE ? Δw[spin] / T(2π * γ) : zero(T)
        sample = 0
        for node in eachindex(Δt)
            dt = Δt[node]
            Bz = Gx[node]*x[spin] + Gy[node]*y[spin] + Gz[node]*z[spin] +
                 off - Δf[node] / T(γ)
            mx, mz = rotate_high_order_spin(mx, mz, B1[node], Bz, dt)
            mx *= exp(-dt / T2[spin])
            E1 = exp(-dt / T1[spin])
            mz = mz*E1 + ρ[spin]*(one(T)-E1)
            if adc[node+1]
                sample += 1
                adc_states[spin, sample] = mx * cis(ψ[node+1])
            end
        end
        Mxy[spin] = mx * cis(ψ[end])
        Mz[spin] = mz
    end
end

function run_high_order_excitation!(workspace, obj, hoseqd, sig, M, method)
    seq = hoseqd.seqd
    high_order_excitation_kernel!(workspace.backend, workspace.groupsize)(
        M.xy, M.z, workspace.adc_states, obj.x, obj.y, obj.z,
        obj.ρ, obj.T1, obj.T2, obj.Δw, seq.Gx, seq.Gy, seq.Gz,
        seq.B1, seq.Δf, seq.ψ, seq.Δt, seq.ADC, Val(method.Δw_excitation);
        ndrange=length(obj),
    )
    combine_high_order_signal!(sig, workspace, obj.csm)
    return nothing
end
