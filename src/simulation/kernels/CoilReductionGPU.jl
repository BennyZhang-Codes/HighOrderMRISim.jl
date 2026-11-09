# Each workgroup combines one ADC/coil and one spin tile. All lanes participate
# in the reduction, including padded lanes in the final tile. No atomics or
# spin-by-ADC-by-coil weighted array is needed. CSM values are not conjugated.
@inline function high_order_coil_sum(states, csm, sample, coil, first_spin, stride,
                                    ::Val{ITEMS}) where {ITEMS}
    value = zero(eltype(states))
    @inbounds for item in 0:ITEMS-1
        spin = first_spin + item * stride
        if spin <= size(states, 1)
            value += states[spin, sample] * csm[spin, coil]
        end
    end
    return value
end

@kernel unsafe_indices=true function high_order_coil_partials_kernel!(
    partials, states, csm, nadc, ::Val{ITEMS},
) where {ITEMS}
    @uniform N = @groupsize()[1]
    lane = @index(Local, Linear)
    group = @index(Group, Linear)
    ntiles = size(partials, 1)
    tile = (group - 1) % ntiles + 1
    channel = (group - 1) ÷ ntiles + 1
    sample = (channel - 1) % nadc + 1
    coil = (channel - 1) ÷ nadc + 1
    scratch = @localmem eltype(partials) (N,)
    first_spin = (tile - 1) * (N * ITEMS) + lane
    scratch[lane] = high_order_coil_sum(states, csm, sample, coil, first_spin, N, Val(ITEMS))
    @synchronize
    offset = N ÷ 2
    while offset > 0
        if lane <= offset
            @inbounds scratch[lane] += scratch[lane + offset]
        end
        @synchronize
        offset ÷= 2
    end
    if lane == 1
        @inbounds partials[tile, channel] = scratch[1]
    end
end

@kernel unsafe_indices=true function high_order_coil_finalize_kernel!(
    sig, partials, ntiles, nadc,
)
    @uniform N = @groupsize()[1]
    lane = @index(Local, Linear)
    channel = @index(Group, Linear)
    scratch = @localmem eltype(partials) (N,)
    value = zero(eltype(partials))
    @inbounds for tile in lane:N:ntiles
        value += partials[tile, channel]
    end
    scratch[lane] = value
    @synchronize
    offset = N ÷ 2
    while offset > 0
        if lane <= offset
            @inbounds scratch[lane] += scratch[lane + offset]
        end
        @synchronize
        offset ÷= 2
    end
    if lane == 1
        sample = (channel - 1) % nadc + 1
        coil = (channel - 1) ÷ nadc + 1
        @inbounds sig[sample, coil] = scratch[1]
    end
end

function high_order_coil_partials(states, ncoils; groupsize=256, items=8)
    groupsize isa Integer && ispow2(groupsize) && groupsize <= 1024 ||
        throw(ArgumentError("Reduction groupsize must be a power of two at most 1024."))
    items isa Integer && items > 0 || throw(ArgumentError("Reduction items must be a positive integer."))
    return similar(states, cld(size(states, 1), groupsize * items), size(states, 2) * ncoils)
end

function reduce_high_order_signal!(sig, states, csm, partials, backend;
                                   groupsize=256, items=8)
    isempty(sig) && return nothing
    nspins, nadc = size(states)
    ntiles = cld(nspins, groupsize * items)
    size(csm, 1) == nspins && size(sig) == (nadc, size(csm, 2)) ||
        throw(DimensionMismatch("States, CSM, and signal dimensions do not match."))
    size(partials, 1) == ntiles && size(partials, 2) >= length(sig) ||
        throw(DimensionMismatch("The reduction workspace is too small or has the wrong spin tiling."))
    if iszero(nspins)
        fill!(sig, zero(eltype(sig)))
        return nothing
    end
    high_order_coil_partials_kernel!(backend, groupsize)(
        partials, states, csm, nadc, Val(items);
        ndrange=ntiles * groupsize * length(sig),
    )
    high_order_coil_finalize_kernel!(backend, groupsize)(
        sig, partials, ntiles, nadc; ndrange=groupsize * length(sig),
    )
    return nothing
end
