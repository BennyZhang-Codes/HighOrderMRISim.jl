

"""
    p = plt_kspace(hoseq::HO_Sequence; kwargs...)

Plot the first-order trajectory in cycles/m. `source=:nominal` preserves the
default nominal-gradient view; use `:measured` for attached h1–h3 coefficients
or `:both` to compare them. These waveform integrals follow `get_kspace` RF
conventions, not the order selection or RF model of `BlochHighOrder`.

# Arguments
- `hoseq`: (`::HO_Sequence`) HO_Sequence struct

# Keywords
- `source`: `:nominal`, `:measured`, or `:both`.
- `Δt`: maximum non-RF integration step in seconds, passed to `get_kspace`.
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
julia> fig = plt_kspace(hoseq; source=:both)
```
"""
function plt_kspace(
    hoseq ::HO_Sequence             ;
    source             = :nominal   ,
    Δt                 = 1          ,
    xlabel             = "kx [cycles/m]",
    ylabel             = "ky [cycles/m]",
    zlabel             = "kz [cycles/m]",
    width              = 20         ,      
    height             = 20         ,  
    fontsize_label     = 7          ,
    fontsize_legend    = 7          ,
    fontsize_ticklabel = 6          ,
    color_facecolor    = "#ffffff"  ,
    color_label        = "#000000"  ,
    color_traj         = "#555555"  ,
    linewidth          = 1.0        ,
    ticklength         = 1.5        ,
    pad_label          = 2          ,
    pad_labeltick      = 2          ,
    markersize         = 4          ,
    cmap               = "jet"      ,
    )
    source in (:nominal, :measured, :both) ||
        throw(ArgumentError("source must be :nominal, :measured, or :both."))
    K_nominal, K_nominal_adc, K_dfc, K_dfc_adc = get_kspace(hoseq; Δt)
    nominal = (K_nominal, K_nominal_adc, "Nominal", color_traj)
    measured = (K_dfc[:, 2:4], K_dfc_adc[:, 2:4], "Measured", "#1d6996")
    trajectories = source == :nominal ? (nominal,) : source == :measured ? (measured,) : (nominal, measured)

    # Include the full selected curves, even when a sequence has no ADC events.
    mink = vec(minimum(first(trajectories)[1]; dims=1))
    maxk = vec(maximum(first(trajectories)[1]; dims=1))
    for (k, _, _, _) in trajectories
        mink = min.(mink, vec(minimum(k; dims=1)))
        maxk = max.(maxk, vec(maximum(k; dims=1)))
    end
    dW = maximum(maxk .- mink) * 0.3
    iszero(dW) && (dW = 1.0)
    mink .-= dW
    maxk .+= dW
    t_adc = KomaMRIBase.get_adc_sampling_times(simulation_sequence(hoseq)) .* 1e3

	fig = plt.figure(figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor)
	ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_box_aspect((1,1,1)) 
    ax.grid(false) 
    ax.xaxis.set_pane_color(color_facecolor)
    ax.yaxis.set_pane_color(color_facecolor)
    ax.zaxis.set_pane_color(color_facecolor)
    ax.xaxis.line.set_color(color_label) 
    ax.yaxis.line.set_color(color_label) 
    ax.zaxis.line.set_color(color_label)
    ax.set_xlim(mink[1], maxk[1])
    ax.set_ylim(mink[2], maxk[2])
    ax.set_zlim(mink[3], maxk[3])
    ax.xaxis.set_tick_params("both", colors=color_label, length=ticklength, width=linewidth, pad=pad_labeltick, 
    color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    ax.yaxis.set_tick_params("both", colors=color_label, length=ticklength, width=linewidth, pad=pad_labeltick, 
    color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    ax.zaxis.set_tick_params("both", colors=color_label, length=ticklength, width=linewidth, pad=pad_labeltick, 
    color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)

    ax.set_facecolor(color_facecolor)
    # ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
    #     color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    # for spine in ax.spines  # "left", "right", "bottom", "top"
    #     # ax.spines[spine].set_linewidth(linewidth)
    #     ax.spines[spine].set_visible(false)
    # end
    for (k, k_adc, label, color) in trajectories
        ax.plot(k[:,1], k[:,2], k[:,3]; color, linewidth, label)
        isempty(t_adc) || ax.scatter(k_adc[:,1], k_adc[:,2], k_adc[:,3],
            marker=".", s=markersize, c=t_adc, cmap=cmap)
    end
    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, frameon=false)

    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_zlabel(zlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    return fig
end

