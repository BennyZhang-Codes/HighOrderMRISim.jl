import KomaMRIBase: get_samples, get_grads, get_kspace
import KomaMRIBase: is_Gx_on, is_Gy_on, is_Gz_on, is_ADC_on, is_RF_on
"""
    hoseq = HO_Sequence()
    hoseq = HO_Sequence(GR_dfc::Array{Grad,2})
    hoseq = HO_Sequence(SEQ::Sequence, GR_dfc::Array{Grad,2})
	hoseq = HO_Sequence(SEQ::Sequence)
	hoseq = HO_Sequence(GR_dfc::Array{Grad,2}, SEQ::Sequence)
	hoseq = HO_Sequence(GR_dfc::Array{Grad,1})
	hoseq = HO_Sequence(SEQ::Sequence, GR_dfc::Array{Grad,1}, DUR)

# Description
	The HO_Sequence struct. It contains MRI sequence and corresponding DFC measured gradients.

# Arguments
- `SEQ`: (`::Sequence`) MRI sequence.
- `GR_dfc`: (`::Matrix{Grad}`) gradient matrix for DFC. Rows for amplitudes and columns are for blocks
\n 	h0 := 1 
\n  h1 := x
\n  h2 := y 
\n  h3 := z
\n  h4 := xy
\n  h5 := yz
\n  h6 := 3z² - (x²	+ y² + z²)
\n  h7 := xz
\n  h8 := x² - y²
\n  h9 := 3yx² - y³
\n  h10 := xzy
\n  h11 := (5z² - (x² + y² + z²))y
\n  h12 := 5z³ - 3z(x² + y² + z²)
\n  h13 := (5z² - (x² + y² + z²))x
\n  h14 := x²z - y²z
\n  h15 := x³ - 3xy²
- `DUR`: (`::Vector`, `[s]`) duration block vector

# Returns
- `hoseq`: (`::HO_Sequence`) HO_Sequence struct
"""
mutable struct HO_Sequence
	SEQ::Sequence
	GR_dfc::Array{Grad,2}	  #Sequence in (h0, ..., h8) and time
	DUR::Vector				  #Duration of each block, this enables delays after RF pulses to satisfy ring-down times
	HO_Sequence(SEQ, GR_dfc, DUR) = begin
		@assert size(SEQ)[1] .== size(GR_dfc, 2)  "The number of SEQ and GR_dfc objects must be the same."
		M, N = size(GR_dfc)
		1 <= M <= 16 || throw(ArgumentError("Expected 1 to 16 high-order field channels."))
		new(SEQ,
		    Grad[i <= M ? GR_dfc[i,j] : Grad(0, 0) for i in 1:16, j in 1:N],
			maximum([SEQ.DUR GR_dfc.dur DUR],dims=2)[:])
	end
end

# Main Constructors
function HO_Sequence(GR_dfc::AbstractMatrix{<:Grad})
	M, N = size(GR_dfc)
    1 <= M <= 16 || throw(ArgumentError("Expected 1 to 16 high-order field channels."))
    gr_dfc = Grad[i <= M ? GR_dfc[i,j] : Grad(0, 0) for i in 1:16, j in 1:N]
	seq = Sequence(gr_dfc[2:4, :])
	dur = maximum([seq.DUR gr_dfc.dur],dims=2)[:]
    return HO_Sequence(seq, gr_dfc, dur)
end

function HO_Sequence(SEQ::Sequence, GR_dfc::AbstractMatrix{<:Grad})
	@assert size(SEQ)[1] .== size(GR_dfc, 2) "The number of SEQ and GR_dfc objects must be the same."
	M, N = size(GR_dfc)
    1 <= M <= 16 || throw(ArgumentError("Expected 1 to 16 high-order field channels."))
    gr_dfc = [i <= M ? GR_dfc[i,j] : Grad(0, 0) for i in 1:16, j in 1:N]
	dur = maximum([SEQ.DUR gr_dfc.dur],dims=2)[:]
    return HO_Sequence(SEQ, gr_dfc, dur)
end

function HO_Sequence(SEQ::Sequence)
	gr_dfc = [Grad(0, 0) for _ in 1:16, _ in 1:size(SEQ)[1]]
	dur = maximum([SEQ.DUR gr_dfc.dur],dims=2)[:]
    return HO_Sequence(SEQ, gr_dfc, dur)
end

# Other constructors
HO_Sequence(GR_dfc::AbstractVector{<:Grad}) = HO_Sequence(reshape(GR_dfc,1,:))
HO_Sequence(SEQ::Sequence, GR_dfc::AbstractVector{<:Grad}, DUR::Real) = HO_Sequence(SEQ, reshape(GR_dfc, :, 1), [DUR])
HO_Sequence() = HO_Sequence([Grad(0, 0)])


#Sequence operations
Base.length(x::HO_Sequence) = length(x.DUR)
Base.iterate(x::HO_Sequence) = (HO_Sequence(x.SEQ[1], x.GR_dfc[:,1], x.DUR[1]), 2)
Base.iterate(x::HO_Sequence, i::Integer) = (i <= length(x)) ? (HO_Sequence(x.SEQ[i], x.GR_dfc[:,i], x.DUR[i]), i+1) : nothing
Base.getindex(x::HO_Sequence, i::UnitRange{Int}) = HO_Sequence(x.SEQ[i], x.GR_dfc[:,i], x.DUR[i])
Base.getindex(x::HO_Sequence, i::Int) = HO_Sequence(x.SEQ[i], x.GR_dfc[:,i], x.DUR[i])
Base.getindex(x::HO_Sequence, i::BitArray{1}) = any(i) ? HO_Sequence(x.SEQ[i], x.GR_dfc[:,i], x.DUR[i]) : nothing
Base.getindex(x::HO_Sequence, i::Array{Bool,1}) = any(i) ? HO_Sequence(x.SEQ[i], x.GR_dfc[:,i], x.DUR[i]) : nothing
Base.lastindex(x::HO_Sequence) = length(x.DUR)
# copy(x::HO_Sequence) where HO_Sequence = HO_Sequence(copy(x.SEQ), deepcopy(x.GR_dfc), deepcopy(x.DUR))


#Arithmetic operations
+(x::HO_Sequence, y::HO_Sequence) = HO_Sequence(x.SEQ + y.SEQ, [x.GR_dfc  y.GR_dfc], [x.DUR; y.DUR])
-(x::HO_Sequence, y::HO_Sequence) = HO_Sequence(x.SEQ + y.SEQ, [x.GR_dfc -y.GR_dfc], [x.DUR; y.DUR])
-(x::HO_Sequence) = HO_Sequence(-x.SEQ, -x.GR_dfc, x.DUR)
*(x::HO_Sequence, α::Real) = HO_Sequence(α*x.SEQ, α*x.GR_dfc, x.DUR)
*(α::Real, x::HO_Sequence) = HO_Sequence(α*x.SEQ, α*x.GR_dfc, x.DUR)
*(x::HO_Sequence, α::ComplexF64) = HO_Sequence(α*x.SEQ, x.GR_dfc, x.DUR)
*(α::ComplexF64, x::HO_Sequence) = HO_Sequence(α*x.SEQ, x.GR_dfc, x.DUR)
/(x::HO_Sequence, α::Real) = HO_Sequence(x.SEQ/α, x.GR_dfc, x.DUR)

#Sequence object functions
size(x::HO_Sequence) = size(x.GR_dfc[1,:])

# Event amplitudes may be edited after construction. Reconcile durations before sampling
# without mutating the caller's native sequence or its Pulseq extensions.
function simulation_sequence(hoseq::HO_Sequence)
    seq = hoseq.SEQ
    durations = max.(seq.DUR, hoseq.DUR, vec(dur(hoseq.GR_dfc)))
    return Sequence(seq.GR, seq.RF, seq.ADC, durations, seq.EXT, seq.DEF)
end

"""
    str = show(io::IO, s::HO_Sequence)

# Description
	Displays information about the Sequence struct `s` in the julia REPL.

# Arguments
- `s`: (`::HO_Sequence`) Sequence struct

# Returns
- `str` (`::String`) output string message
"""
Base.show(io::IO, hs::HO_Sequence) = begin
	s = hs.SEQ
	compact = get(io, :compact, false)
	if !compact
		nGRs = sum(is_Gx_on.(s)) + sum(is_Gy_on.(s)) + sum(is_Gz_on.(s))
		print(io, "HO_Sequence[ τ = $(round(dur(s)*1e3;digits=3)) ms | blocks: $(length(s)) | ADC: $(sum(is_ADC_on.(s))) | GR: $nGRs | RF: $(sum(is_RF_on.(s))) | DEF: $(length(s.DEF)) ]")
	else
		print(io, "HO_Sequence[τ = $(round(dur(s)*1e3;digits=3)) ms]")
	end
end

"""
    samples = get_samples(hoseq::HO_Sequence; off_val=0, max_rf_samples=Inf)

# Description
	Returns the samples of the events in `hoseq`.

# Arguments
- `hoseq`: (`::HO_Sequence`) HO_Sequence struct

# Keywords
- `off_val`: (`::Number`, `=0`) offset value for amplitude. Typically used to hide points in
    plots by setting it to `Inf`
- `max_rf_samples`: (`::Integer`, `=Inf`) maximum number of samples for the RF struct

# Returns
- `samples`: (`::NamedTuple`) contains samples for `h0`, `h1`, `h2`, `h3`, `h4`, `h5`, `h6`,
    `h7`, `h8`, `gx`, `gy`, `gz`, `rf`, and `adc` events. Each event, represented by 
	`e::NamedTuple`, includes time samples (`e.t`) and amplitude samples (`e.A`)
"""
function get_samples(hoseq::HO_Sequence; off_val=0, max_rf_samples=Inf)
    samples = get_samples(simulation_sequence(hoseq))
    rf = samples.rf
    if max_rf_samples != Inf
        max_rf_samples isa Integer && max_rf_samples > 0 ||
            throw(ArgumentError("max_rf_samples must be a positive integer or Inf."))
        if length(rf.t) > max_rf_samples
            indices = unique(round.(Int, range(1, length(rf.t); length=max_rf_samples)))
            rf = (; t=rf.t[indices], A=rf.A[indices])
        end
    end
    fields = ntuple(16) do term
        t, A = get_theo_Gi(hoseq, term)
        if !iszero(off_val)
            A = copy(A)
            A[iszero.(A)] .= off_val
        end
        (; t, A)
    end
    names = ntuple(i -> Symbol("h", i - 1), 16)
    nominal = (; gx=samples.gx, gy=samples.gy, gz=samples.gz,
                rf, adc=samples.adc)
    return merge(nominal, NamedTuple{names}(fields))
end

function get_theo_Gi(hoseq::HO_Sequence, term)
    seq = simulation_sequence(hoseq)
    starts = get_block_start_times(seq)
    t, A = Float64[], Float64[]
    for block in 1:length(seq)
        grad = hoseq.GR_dfc[term, block]
        append!(t, times(grad) .+ starts[block])
        append!(A, ampls(grad))
    end
    return t, A
end

function get_grads(hoseq::HO_Sequence, t::Vector)
    order = sortperm(t)
    return ntuple(16) do term
        knots, amplitudes = get_theo_Gi(hoseq, term)
        samples = KomaMRIBase.linear_interpolate_samples(
            (; t=knots, A=amplitudes), t[order]; default=0.0,
        )
        values = similar(samples)
        values[order] .= samples
        values
    end
end

"""
    K_nominal, K_nominal_adc, K_dfc, K_dfc_adc = get_kspace(hoseq::HO_Sequence; Δt=1)

# Description
	Outputs the designed k-space trajectory of the Sequence `hoseq`.

# Arguments
- `hoseq`: (`::HO_Sequence`) HO_Sequence struct
- `Δt`: (`::Real`, `=1`, `[s]`) nominal delta time separation between two time samples
    for ADC acquisition and Gradients

# Returns
- `K_nominal`: (`3-column ::Matrix{Float64}`) nominal kspace
- `K_nominal_adc`: (`3-column ::Matrix{Float64}`) nominal adc kspace
- `K_dfc`: (`16-column ::Matrix{Float64}`) generalized DFC encoding coefficients
- `K_dfc_adc`: (`16-column ::Matrix{Float64}`) ADC-sampled DFC encoding coefficients
"""
function get_kspace(
    hoseq::HO_Sequence; Δt=1,
    skip_rf=falses(sum(is_RF_on.(hoseq.SEQ))),
    sampling_rule=MaxStepSizeRule(Δt, 5e-5),
)
    hoseqd = discretize(hoseq; sampling_rule)
    seqd = hoseqd.seqd
    t = seqd.t[1:end-1]
    nominal = (seqd.Gx, seqd.Gy, seqd.Gz)
    measured = ntuple(i -> getfield(hoseqd, Symbol("h", i - 1)), 16)
    seq = simulation_sequence(hoseq)
    K_nominal, K_nominal_adc = grad2kspace(seq, nominal, t, seqd.Δt, skip_rf)
    K_dfc, K_dfc_adc = grad2kspace(hoseq, measured, t, seqd.Δt, skip_rf)
    return K_nominal, K_nominal_adc, K_dfc, K_dfc_adc
end

grad2kspace(hoseq::HO_Sequence, G, t, Δt, skip_rf) =
    grad2kspace(simulation_sequence(hoseq), G, t, Δt, skip_rf)

function grad2kspace(seq::Sequence, G, t, Δt, skip_rf)
    isempty(t) && throw(ArgumentError("Cannot integrate an empty sequence."))
    idx_rf, rf_types = KomaMRIBase.get_RF_types(seq, t)
    length(skip_rf) == length(rf_types) ||
        throw(DimensionMismatch("skip_rf must contain one flag per RF event."))
    parts = kfoldperm(length(t), 1; breaks=idx_rf)
    k = zeros(length(t), length(G))
    for term in 1:length(G)
        previous = 0.0
        for (rf, part) in enumerate(parts)
            k[part, term] .= cumtrapz(
                Δt[part], G[term][first(part):last(part)+1],
            )
            if rf > 1
                rf_type = rf_types[rf-1]
                sign = skip_rf[rf-1] ? 1 :
                    rf_type isa Excitation ? 0 : rf_type isa Refocusing ? -1 : 1
                k[part, term] .+= sign * previous
            end
            previous = k[last(part), term]
        end
    end
    kspace = γ .* k
    sample_times = t .+ Δt
    adc_times = get_adc_sampling_times(seq)
    kspace_adc = zeros(length(adc_times), length(G))
    for term in 1:length(G)
        kspace_adc[:, term] .= KomaMRIBase.linear_interpolate_samples(
            (; t=sample_times, A=view(kspace, :, term)), adc_times; default=0.0,
        )
    end
    return kspace, kspace_adc
end
