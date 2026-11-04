# CUDA-only subgroup reduction. Every participating subgroup has all 32 lanes,
# including the padded lanes of a partial spin tile. No fast math or atomics.
@inline function high_order_warp_sum(value::Complex)
    for offset in (16, 8, 4, 2, 1)
        value += CUDA.shfl_down_sync(0xffffffff, value, offset)
    end
    return value
end

# Julia 1.12 map(::Tuple) allocates for 32 or more entries. A static ntuple
# preserves register tuples and does not capture a reassigned accumulator.
@inline function high_order_warp_sums(values::NTuple{N}) where {N}
    return ntuple(j -> high_order_warp_sum(values[j]), Val(N))
end

@inline function high_order_outer_accumulate(acc, samples, weights,
                                            ::Val{A}, ::Val{C}) where {A,C}
    return ntuple(Val(A*C)) do j
        acc[j] + samples[(j-1) % A + 1] * weights[(j-1) ÷ A + 1]
    end
end

@inline function high_order_coil_weights(csm, spin, first_coil, nspins,
                                         ::Val{C}) where {C}
    return ntuple(Val(C)) do j
        coil = first_coil + j - 1
        spin <= nspins && coil <= size(csm, 2) ? (@inbounds csm[spin, coil]) : zero(eltype(csm))
    end
end

# Warps process separate coil tiles. Shared ADC states are reused across coils;
# each CSM load feeds every ADC in the tile. Only two block barriers per tile.
@inline function reduce_high_order_adc_tile!(partials, scratch, csm, thread, group,
                                            nspins, nadc, sample, count, N,
                                            ::Val{A}, ::Val{C}) where {A,C}
    CUDA.sync_threads()
    lane = (thread-1) % 32 + 1
    warp = (thread-1) ÷ 32
    nwarps = N ÷ 32
    T = eltype(partials)
    first_coil = warp*C + 1
    while first_coil <= size(csm, 2)
        acc = ntuple(_ -> zero(T), Val(A*C))
        for local_spin in lane:32:N
            spin = (group-1)*N + local_spin
            weights = high_order_coil_weights(csm, spin, first_coil, nspins, Val(C))
            samples = ntuple(Val(A)) do j
                j <= count ? (@inbounds scratch[local_spin, j]) : zero(T)
            end
            acc = high_order_outer_accumulate(acc, samples, weights, Val(A), Val(C))
        end
        sums = high_order_warp_sums(acc)
        if lane == 1
            for j in 1:A*C
                adc_in_tile = (j-1) % A + 1
                coil = first_coil + (j-1) ÷ A
                if adc_in_tile <= count && coil <= size(csm, 2)
                    channel = sample-count+adc_in_tile + (coil-1)*nadc
                    @inbounds partials[group, channel] = sums[j]
                end
            end
        end
        first_coil += nwarps*C
    end
    # All warps must finish reading before spin evolution reuses scratch.
    CUDA.sync_threads()
    return nothing
end

@kernel unsafe_indices=true function high_order_fused_precession_kernel!(
    Mxy, Mz, partials, x, y, z, ρ, T1, T2, Δw, coefficients, gradients,
    Δt, adc, orders, csm, nspins, nadc, ::Val{A}, ::Val{C},
) where {A,C}
    @uniform N = @groupsize()[1]
    thread = @index(Local, Linear)
    group = @index(Group, Linear)
    spin = (group-1)*N + thread
    active = spin <= nspins
    safe_spin = min(spin, nspins)
    scratch = @localmem eltype(Mxy) (N, A)
    T = eltype(x)
    @inbounds begin
        basis = high_order_basis(x[safe_spin], y[safe_spin], z[safe_spin])
        previous = high_order_field(basis, coefficients, gradients, Δw[safe_spin], 1, orders)
        initial = Mxy[safe_spin]
        ϕ, t = zero(T), zero(T)
        sample, count = 0, 0
        for node in 2:length(adc)
            dt = Δt[node-1]
            t += dt
            next_field = high_order_field(basis, coefficients, gradients, Δw[safe_spin], node, orders)
            ϕ += T(-π*γ) * (previous+next_field) * dt
            if adc[node]
                sample += 1
                count += 1
                value = initial * exp(-t/T2[safe_spin]) * cis(ϕ)
                scratch[thread, count] = active ? value : zero(eltype(Mxy))
                if count == A
                    reduce_high_order_adc_tile!(partials, scratch, csm, thread, group,
                        nspins, nadc, sample, count, N, Val(A), Val(C))
                    count = 0
                end
            end
            previous = next_field
        end
        if count > 0
            reduce_high_order_adc_tile!(partials, scratch, csm, thread, group,
                nspins, nadc, sample, count, N, Val(A), Val(C))
        end
        if active
            Mxy[spin] = initial * exp(-t/T2[spin]) * cis(ϕ)
            E1 = exp(-t/T1[spin])
            Mz[spin] = Mz[spin]*E1 + ρ[spin]*(one(T)-E1)
        end
    end
end

function run_fused_high_order_precession!(workspace, obj, hoseqd, sig, M, method)
    seq, fusion = hoseqd.seqd, workspace.fusion
    coefficients = ntuple(i -> getfield(hoseqd, Symbol("h", i-1)), 16)
    orders = Val((method.ho0, method.ho1, method.ho2, method.ho3, method.Δw_precession))
    high_order_fused_precession_kernel!(workspace.backend, workspace.groupsize)(
        M.xy, M.z, fusion.partials, obj.x, obj.y, obj.z, obj.ρ, obj.T1, obj.T2,
        obj.Δw, coefficients, (seq.Gx, seq.Gy, seq.Gz), seq.Δt, seq.ADC, orders,
        obj.csm, length(obj), size(sig, 1), fusion.adc_tile, fusion.coil_tile;
        ndrange=cld(length(obj), workspace.groupsize)*workspace.groupsize,
    )
    high_order_coil_finalize_kernel!(workspace.backend, 256)(
        sig, fusion.partials, size(fusion.partials, 1), size(sig, 1);
        ndrange=256*length(sig),
    )
    return nothing
end
