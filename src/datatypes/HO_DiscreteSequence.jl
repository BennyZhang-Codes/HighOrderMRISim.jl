import KomaMRIBase: discretize

"""
    seqd = HO_DiscreteSequence(seqd, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)

# Description
    A sampled version of a HO_Sequence struct, containing vectors for event amplitudes at specified
times. DiscreteSequence is the struct used for simulation.

# Arguments
- `seqd`: (`::DiscreteSequence`) 
- `h0`: (`::AbstractVector{T<:Real}`, `[T]`)    `1`
- `h1`: (`::AbstractVector{T<:Real}`, `[T/m]`)  `x`
- `h2`: (`::AbstractVector{T<:Real}`, `[T/m]`)  `y`
- `h3`: (`::AbstractVector{T<:Real}`, `[T/m]`)  `z`
- `h4`: (`::AbstractVector{T<:Real}`, `[T/m²]`) `xy`
- `h5`: (`::AbstractVector{T<:Real}`, `[T/m²]`) `zy`
- `h6`: (`::AbstractVector{T<:Real}`, `[T/m²]`) `3z² - (x²	+ y² + z²)`
- `h7`: (`::AbstractVector{T<:Real}`, `[T/m²]`) `xz`
- `h8`: (`::AbstractVector{T<:Real}`, `[T/m²]`) `x² - y²`
- `h9`: (`::AbstractVector{T<:Real}`, `[T/m³]`)  `3yx² - y³`
- `h10`: (`::AbstractVector{T<:Real}`, `[T/m³]`) `xzy`
- `h11`: (`::AbstractVector{T<:Real}`, `[T/m³]`) `5z² - (x² + y² + z²))y`
- `h12`: (`::AbstractVector{T<:Real}`, `[T/m³]`) `5z³ - 3z(x² + y² + z²)`
- `h13`: (`::AbstractVector{T<:Real}`, `[T/m³]`) `5z² - (x² + y² + z²))x`
- `h14`: (`::AbstractVector{T<:Real}`, `[T/m³]`) `x²z - y²z`
- `h15`: (`::AbstractVector{T<:Real}`, `[T/m³]`) `x³ - 3xy²`
# Returns
- `hoseqd`: (`::HO_DiscreteSequence`) HO_DiscreteSequence struct
"""
struct HO_DiscreteSequence{T<:Real,S<:DiscreteSequence{T},H<:AbstractVector{T}}
    seqd::S
    h0::H
    h1::H
    h2::H
    h3::H
    h4::H
    h5::H
    h6::H
    h7::H
    h8::H
    h9::H
    h10::H
    h11::H
    h12::H
    h13::H
    h14::H
    h15::H
end

function HO_DiscreteSequence(seqd::S, fields::Vararg{AbstractVector{T},16}) where {T,S<:DiscreteSequence{T}}
    # Ordinary homogeneous CPU/CUDA storage stays concrete. Mixed vector
    # implementations retain the generality of the former abstract fields.
    H = foldl(typejoin, map(typeof, fields))
    return HO_DiscreteSequence{T,S,H}(seqd, fields...)
end
HO_DiscreteSequence{T}(seqd::DiscreteSequence{T}, fields::Vararg{AbstractVector{T},16}) where {T} =
    HO_DiscreteSequence(seqd, fields...)
@functor HO_DiscreteSequence

@inline high_order_coefficients(hoseqd::HO_DiscreteSequence) =
    ntuple(i -> getfield(hoseqd, i+1), Val(16))

Base.length(hoseqd::HO_DiscreteSequence) = length(hoseqd.seqd)
Base.getindex(hoseqd::HO_DiscreteSequence, i::Integer) =
    HO_DiscreteSequence(hoseqd.seqd[i], ntuple(Val(16)) do term
        getfield(hoseqd, term+1)[i:i]
    end...)
Base.getindex(hoseqd::HO_DiscreteSequence, nodes::UnitRange) =
    HO_DiscreteSequence(hoseqd.seqd[nodes], ntuple(Val(16)) do term
        getfield(hoseqd, term+1)[nodes]
    end...)
Base.view(hoseqd::HO_DiscreteSequence, nodes::UnitRange) =
    HO_DiscreteSequence(view(hoseqd.seqd, nodes), ntuple(Val(16)) do term
        view(getfield(hoseqd, term+1), nodes)
    end...)
Base.iterate(hoseqd::HO_DiscreteSequence) =
    isempty(hoseqd.seqd.Δt) ? nothing : (hoseqd[1], 2)
Base.iterate(hoseqd::HO_DiscreteSequence, i) =
    i <= length(hoseqd) ? (hoseqd[i], i + 1) : nothing

function high_order_sampling_rule(params)
    return get(params, "sampling_rule") do
        MaxStepSizeRule(
            get(params, "Δt", 1e-3), get(params, "Δt_rf", 5e-5);
            preserve_samples=get(params, "preserve_samples", (:gradients, :rf_center)),
        )
    end
end

# Keep Koma's boundary convention, but only merge samples when the high-order
# coefficients also agree. A field jump must retain both sides at the same time.
function append_high_order_block!(out, fields, values, coefficients, t0)
    isempty(values.t) && return nothing
    first_t = t0 + first(values.t)
    if !isempty(out.t) && KomaMRIBase.same_sampling_time(first_t, last(out.t))
        first_t = last(out.t)
    end
    first_row = firstindex(values.t)
    if isempty(out.t)
        KomaMRIBase.append_adc_start_padding!(out, values, first_t)
        if !isempty(out.t)
            foreach((dst, src) -> push!(dst, first(src)), fields, coefficients)
        end
    elseif first_t == last(out.t) &&
            KomaMRIBase.same_boundary_sample(out, values) &&
            all(last(dst) == first(src) for (dst, src) in zip(fields, coefficients))
        first_row += 1
    else
        push!(out.Δt, first_t - last(out.t))
        push!(out.excitation_bool, false)
    end
    rows = first_row:lastindex(values.t)
    for t in view(values.t, rows)
        push!(out.t, t == first(values.t) ? first_t : t0 + t)
    end
    foreach(
        (dst, src) -> append!(dst, view(src, rows)),
        KomaMRIBase.table_columns(out), KomaMRIBase.table_columns(values),
    )
    foreach((dst, src) -> append!(dst, view(src, rows)), fields, coefficients)
    append!(out.Δt, values.Δt)
    append!(out.excitation_bool, values.excitation_bool)
    return nothing
end

"""
    discretize(hoseq::HO_Sequence; sampling_params=Dict(), sampling_rule=nothing,
               freq_in_phase=false)

Sample native sequence events and all sixteen high-order waveforms on one grid.
The legacy `sampling_params` dictionary remains supported. An explicit
`sampling_rule` takes precedence. Repeated times preserve field discontinuities.
Each field has one value per time point; `Δt` has one value per interval.
"""
function discretize(
    hoseq::HO_Sequence; sampling_params=Dict{String,Any}(),
    sampling_rule=nothing,
    freq_in_phase=get(sampling_params, "freq_in_phase", false),
)::HO_DiscreteSequence
    rule = isnothing(sampling_rule) ? high_order_sampling_rule(sampling_params) : sampling_rule
    seq = simulation_sequence(hoseq)
    starts = get_block_start_times(seq)
    out = DiscreteSequence()
    fields = ntuple(_ -> Float64[], 16)
    maxdt = rule isa MaxStepSizeRule ? rule.Δt : get(sampling_params, "Δt", Inf)
    for block in 1:length(seq)
        waveforms = KomaMRIBase.block_event_waveforms(seq, block; freq_in_phase)
        high_order = ntuple(term -> KomaMRIBase.event_samples(hoseq.GR_dfc[term, block]), 16)
        boundaries = [0.0, seq.DUR[block]]
        KomaMRIBase.align_sampling_times!(
            (w.t for w in values(waveforms))...,
            (w.t for w in high_order)..., boundaries,
        )
        key_times = KomaMRIBase.merge_sampling_times(
            boundaries, (w.t for w in high_order)...,
        )
        extra_times = Float64[]
        for waveform in high_order
            KomaMRIBase.append_max_step_sampling_times!(
                extra_times, key_times, waveform.t, maxdt,
            )
        end
        key_times = KomaMRIBase.merge_sampling_times(key_times, unique!(sort!(extra_times)))
        eval_times = KomaMRIBase.block_sampling_times(
            seq, block; sampling_rule=rule, waveforms, motion_times=key_times,
        )
        sampled = KomaMRIBase.sample_sequence_block(
            waveforms, eval_times;
            rf_center=KomaMRIBase.rf_center_time(seq.RF[1, block]), freq_in_phase,
        )
        coefficients = map(high_order) do waveform
            KomaMRIBase.linear_interpolate_samples(waveform, eval_times; default=0.0)
        end
        append_high_order_block!(out, fields, sampled, coefficients, starts[block])
    end
    return HO_DiscreteSequence(out, fields...)
end
