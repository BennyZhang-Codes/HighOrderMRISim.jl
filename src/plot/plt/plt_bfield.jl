"""
    fig = plt_bfield(bfield::AbstractArray{<:Real, 2}, dt::Float64; kwargs...)

# Description
    Plots spherical-harmonic field coefficients through second or third order.

# Arguments
- `bfield::AbstractArray{<:Real, 2}`: `[nSample, nTerms]` field coefficients; 9 or 16 columns.
- `dt::Float64`: sampling time in seconds.

# Keywords
- `units`: `:mT` (default) or `:T`; labels the supplied values without rescaling them.
  The corresponding order-n unit is mT/m^n or T/m^n, not radians.
- `times`: optional sample times in seconds. Otherwise the legacy `dt:dt:nSample*dt` axis is used.
- `width`: (`::Real`, `=15`) figure's width
- `height`: (`::Real`, `=7`) figure's height
- `fontsize_label`: (`::Real`, `=7`) font size of labels
- `fontsize_legend`: (`::Real`, `=7`) font size of legend
- `fontsize_ticklabel`: (`::Real`, `=6`) font size of tick labels
- `color_facecolor`: (`::String`, `="#1F1F1F"`) background color of the figure  
- `color_label`: (`::String`, `="#CCCCCC"`) color of labels and tick labels
- `linewidth`: (`::Real`, `=0.5`) line width
- `ticklength`: (`::Real`, `=1.5`) length of ticks
- `pad_label`: (`::Real`, `=2`) padding between labels and tick labels
- `pad_labeltick`: (`::Real`, `=2`) padding between tick labels and axis

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_bfield(bfield, dt)
```
"""
function plt_bfield(
    bfield::AbstractArray{<:Real, 2},
    dt::Float64;
    units              = :mT      ,
    times              = nothing  ,
    width              = 15       ,
    height             = 7        ,
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    )
    nSample, nTerm = size(bfield)
    @assert nTerm in (9, 16) "nTerm must be 9 or 16 for up to 2nd or 3rd order spherical harmonics"
    units in (:mT, :T) || throw(ArgumentError("units must be :mT or :T."))
    unit_labels = coefficient_plot_units(units)
    t = coefficient_plot_times(nSample, dt, times)
    nRow = nTerm == 9 ? 3 : 4
    fig, axs = plt.subplots(nrows=nRow, ncols=1, figsize=(width/2.53999863, height/2.53999863), sharex=true, facecolor=color_facecolor, squeeze=false)
    
    for ax in axs
        ax.set_xlim(t[1]-1, t[end]+1);
        ax.set_facecolor(color_facecolor)
        ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
            color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
        for spine in ax.spines  # "left", "right", "bottom", "top"
            # ax.spines[spine].set_linewidth(linewidth)
            ax.spines[spine].set_visible(false)
        end
    end
    
    term_ranges = (1:1, 2:4, 5:9, 10:16)
    order_labels = ("0th", "1st", "2nd", "3rd")
    for row in 1:nRow
        ax = axs[row, 1]
        for term in term_ranges[row]
            ax.plot(t, bfield[:, term], linewidth=linewidth, color=spha_color[term], label=spha_basis_latex[term])
        end
        ax.set_ylabel("$(order_labels[row]) order [$(unit_labels[row])]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
        ax.legend(bbox_to_anchor=(0.01, 1.02), fontsize=fontsize_legend, labelcolor=color_label,
                scatteryoffsets=[0.5],
                ncols=row == 4 ? 4 : 5, loc="lower left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end
    axs[end, 1].set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    
    fig.align_ylabels()
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return fig
end


"""
    fig = plt_bfield_com(bfield1::AbstractArray{<:Real, 2}, bfield2::AbstractArray{<:Real, 2}, dt::Float64; kwargs...)

# Description
    Compares spherical-harmonic field coefficients through second or third order.

# Arguments
- `bfield1`, `bfield2`: equally sized `[nSample, nTerms]` field coefficients; 9 or 16 columns.
- `dt::Float64`: sampling time in seconds.

# Keywords
- `units`: `:mT` (default) or `:T`; labels values without rescaling them.
- `times`: optional common sample times in seconds; otherwise the legacy `dt:dt:nSample*dt` axis is used.
- `width`: (`::Real`, `=15`) figure's width
- `height`: (`::Real`, `=7`) figure's height
- `fontsize_label`: (`::Real`, `=7`) font size of labels
- `fontsize_legend`: (`::Real`, `=7`) font size of legend
- `fontsize_ticklabel`: (`::Real`, `=6`) font size of tick labels
- `color_facecolor`: (`::String`, `="#1F1F1F"`) background color of the figure  
- `color_label`: (`::String`, `="#CCCCCC"`) color of labels and tick labels
- `linewidth`: (`::Real`, `=0.5`) line width
- `ticklength`: (`::Real`, `=1.5`) length of ticks
- `pad_label`: (`::Real`, `=2`) padding between labels and tick labels
- `pad_labeltick`: (`::Real`, `=2`) padding between tick labels and axis

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_bfield_com(bfield1, bfield2, dt)
```
"""
function plt_bfield_com(
    bfield1::AbstractArray{<:Real, 2},
    bfield2::AbstractArray{<:Real, 2},
    dt::Float64;
    units              = :mT      ,
    times              = nothing  ,
    width              = 15       ,
    height             = 8        ,
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    )

    size(bfield1) == size(bfield2) || throw(DimensionMismatch("The field coefficient matrices must have the same size."))
    nSample, nTerm = size(bfield1)
    @assert nTerm in (9, 16) "nTerm must be 9 or 16 for up to 2nd or 3rd order spherical harmonics"
    units in (:mT, :T) || throw(ArgumentError("units must be :mT or :T."))
    unit_labels = coefficient_plot_units(units)
    t = coefficient_plot_times(nSample, dt, times)
    nRow = nCol = nTerm == 9 ? 3 : 4
    fig, axs = plt.subplots(nrows=nRow, ncols=nCol, figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor, squeeze=false)
    
    for row = 1:nRow, col = 1:nCol
        ax = axs[row, col]
        term = (row-1)*nCol + col
    
        ax.set_xlim(t[1]-1, t[end]+1);
        ax.set_facecolor(color_facecolor)
        ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
            color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
        for spine in ax.spines  # "left", "right", "bottom", "top"
            # ax.spines[spine].set_linewidth(linewidth)
            ax.spines[spine].set_visible(false)
        end
    
        ax.plot(t, bfield1[:, term], linewidth=linewidth, label="bfield 1")
        ax.plot(t, bfield2[:, term], linewidth=linewidth, label="bfield 2")
        ax.set_xlabel(L"Time \ [ms]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
        
        order = term == 1 ? 0 : term <= 4 ? 1 : term <= 9 ? 2 : 3
        ylabel = "$(spha_basis[term]) [$(unit_labels[order+1])]"
        ax.set_ylabel(ylabel, fontsize=fontsize_label, color=spha_color[term], labelpad=pad_label)
        ax.legend(bbox_to_anchor=(0.01, 1.05), fontsize=fontsize_legend, labelcolor=color_label, 
                scatteryoffsets=[0.5],
                ncols=5, loc="upper left", frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    end
    
    fig.align_ylabels()
    fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    return fig
end
