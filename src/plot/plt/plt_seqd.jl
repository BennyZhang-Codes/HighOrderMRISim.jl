

"""
    p = plt_seqd(hoseq::HO_Sequence; kwargs...)

Plots sampled nominal gradients and supplied high-order fields. Curves have
different units, stated in their legends. No order selection or field summation
is applied for display.

# Arguments
- `hoseqd`: (`::HO_DiscreteSequence`) HO_DiscreteSequence struct
- `hoseq`: (`::HO_Sequence`) HO_Sequence struct

# Keywords
- `sampling_params`: The sampling parameters for the sequence.
- `xlabel`: the label for the x-axis.
- `ylabel`: the label for the y-axis.
- `width`: the width of the figure in centimetres.
- `height`: the height of the figure in centimetres.
- `fontsize_label`: the font size for the labels.
- `fontsize_legend`: the font size for the legend.
- `fontsize_ticklabel`: the font size for the tick labels.
- `color_facecolor`: the background color of the figure.
- `color_label`: the color of the labels.
- `linewidth`: the width of the curves.
- `ticklength`: the length of the ticks.
- `pad_label`: the padding for the labels.
- `pad_labeltick`: the padding for the tick labels.
- `marker_size`: the size of the markers.

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_seqd(hoseq)
```
"""
function plt_seqd(
    hoseqd :: HO_DiscreteSequence                             ;
    xlabel             = "Time [ms]"                          ,
    ylabel             = "Amplitude (units in legend)"        ,
    width              = 20                                   ,      
    height             = 8                                    ,  
    fontsize_label     = 7                                    ,
    fontsize_legend    = 7                                    ,
    fontsize_ticklabel = 6                                    ,
    color_facecolor    = "#ffffff"                            ,
    color_label        = "#000000"                            ,
    linewidth          = 1.0                                  ,
    ticklength         = 1.5                                  ,
    pad_label          = 2                                    ,
    pad_labeltick      = 2                                    ,
    marker_size        = 4                                    ,
    )
    sh = SphericalHarmonics()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor, squeeze=false)
    ax = ax[1]
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    hide_plot_spines!(ax)

    ax.plot(hoseqd.seqd.t*1e3, hoseqd.seqd.Gx*1e3, label="Gx nominal [mT/m]", marker="o", color="#636efa", markersize=marker_size, linewidth=linewidth)
    ax.plot(hoseqd.seqd.t*1e3, hoseqd.seqd.Gy*1e3, label="Gy nominal [mT/m]", marker="o", color="#EF553B", markersize=marker_size, linewidth=linewidth)
    ax.plot(hoseqd.seqd.t*1e3, hoseqd.seqd.Gz*1e3, label="Gz nominal [mT/m]", marker="o", color="#00cc96", markersize=marker_size, linewidth=linewidth)
    ax.plot(hoseqd.seqd.t*1e3, abs.(hoseqd.seqd.B1*1e6), label="|B1| [μT]", marker="o", color="#AB63FA", markersize=marker_size, linewidth=linewidth)
    ax.plot(hoseqd.seqd.t[hoseqd.seqd.ADC]*1e3, zeros(sum(hoseqd.seqd.ADC)), label="ADC samples", marker="x", color="#FF6692", markersize=marker_size, linewidth=linewidth)

    for term in 0:15
        name = Symbol("h", term)
        field = getproperty(sh, name)
        unit = replace(field.unit, "T" => "mT")
        ax.plot(hoseqd.seqd.t*1e3, getproperty(hoseqd, name)*1e3, label="h$term field [$unit]",
            marker="o", color=field.color, markersize=marker_size, linewidth=linewidth)
    end

    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)

    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, scatteryoffsets=[0.5],
    loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=4,
    frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    return fig
end



function plt_seqd(
    hoseq ::HO_Sequence                                       ;
    sampling_params    = Dict{String,Any}(),
    xlabel             = "Time [ms]"                          ,
    ylabel             = "Amplitude (units in legend)"        ,
    width              = 20                                   ,      
    height             = 8                                    ,  
    fontsize_label     = 7                                    ,
    fontsize_legend    = 7                                    ,
    fontsize_ticklabel = 6                                    ,
    color_facecolor    = "#ffffff"                            ,
    color_label        = "#000000"                            ,
    linewidth          = 1.0                                  ,
    ticklength         = 1.5                                  ,
    pad_label          = 2                                    ,
    pad_labeltick      = 2                                    ,
    marker_size        = 4                                    ,
    )
    hoseqd = discretize(hoseq; sampling_params)
    return plt_seqd(hoseqd; xlabel, ylabel, width, height, fontsize_label, fontsize_legend, fontsize_ticklabel, 
    color_facecolor, color_label, linewidth, ticklength, pad_label, pad_labeltick, marker_size)
end
