

"""
    p = plt_seq(hoseq::HO_Sequence; kwargs...)

Plots nominal gradients and supplied high-order field waveforms. Curves have
different units, stated in their legends; the supplied fields are not an
order-selected effective field and are not added to the nominal gradients.

# Arguments
- `hoseq`: (`::HO_Sequence`) HO_Sequence struct

# Keywords
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

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_seq(hoseq)
```
"""
function plt_seq(
      hoseq ::HO_Sequence             ;
      xlabel             = "Time [ms]",
      ylabel             = "Amplitude (units in legend)",
      width              = 20         ,      
      height             = 8          ,  
      fontsize_label     = 7          ,
      fontsize_legend    = 7          ,
      fontsize_ticklabel = 6          ,
      color_facecolor    = "#ffffff"  ,
      color_label        = "#000000"  ,
      linewidth          = 1.0        ,
      ticklength         = 1.5        ,
      pad_label          = 2          ,
      pad_labeltick      = 2          ,
    )
    sh = SphericalHarmonics()
    # Get the samples of the events in the sequence
    samples = get_samples(hoseq; off_val=Inf, max_rf_samples=Inf)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor, squeeze=false)
    ax = ax[1]
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    hide_plot_spines!(ax)

    # For GRADs
    ax.plot(samples.gx.t*1e3, samples.gx.A*1e3, label="Gx nominal [mT/m]", color="#636EFA", linewidth=linewidth)
    ax.plot(samples.gy.t*1e3, samples.gy.A*1e3, label="Gy nominal [mT/m]", color="#EF553B", linewidth=linewidth)
    ax.plot(samples.gz.t*1e3, samples.gz.A*1e3, label="Gz nominal [mT/m]", color="#00CC96", linewidth=linewidth)


    # For RFs
    seq = simulation_sequence(hoseq)
    O = size(seq.RF, 1)
    for j in 1:O
        # Native sampling exposes the first transmit channel; sample each
        # channel separately without changing the simulation's RF model.
        rf = j == 1 ? samples.rf : get_samples(Sequence(
            seq.GR, seq.RF[j:j, :], seq.ADC, seq.DUR, seq.EXT, seq.DEF)).rf
        rf_phase = angle.(rf.A)
        rf_phase[rf.A .== Inf] .= Inf
        ax.plot(rf.t*1e3, abs.(rf.A)*1e6, label="|B1| channel $j [μT]", color="#AB63FA", linewidth=linewidth)
        ax.plot(rf.t*1e3, rf_phase, label="B1 phase $j [rad]", color="#FFA15A", linewidth=linewidth, alpha=0.5)
    end

    # For ADCs
    ax.plot(samples.adc.t*1e3, samples.adc.A*1.0, label="ADC [on/off]", color="#19D3F3", linewidth=linewidth)

    for term in 0:15
        name = Symbol("h", term)
        field, sample = getproperty(sh, name), getproperty(samples, name)
        unit = replace(field.unit, "T" => "mT")
        ax.plot(sample.t*1e3, sample.A*1e3, label="h$term field [$unit]", color=field.color, linewidth=linewidth)
    end

    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)

    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, scatteryoffsets=[0.5],
    loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=4,
    frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    return fig
end
