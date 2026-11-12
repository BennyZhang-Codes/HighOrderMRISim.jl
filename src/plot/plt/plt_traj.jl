function trajectory_projection(projection)
    projection == :xy && return (1, 2)
    projection == :xz && return (1, 3)
    projection == :yz && return (2, 3)
    projection == :xyz && return (1, 2, 3)
    throw(ArgumentError("projection must be :xy, :xz, :yz, or :xyz."))
end

"""
    fig = plt_traj(ktraj::AbstractArray{<:Real, 2}; kwargs...)

# Description
    Plots a first-order k-trajectory using matplotlib 

# Arguments
- `ktraj::AbstractArray{<:Real, 2}`: `[nPoint, 3]` trajectory in cycles/m.

# Keywords
- `projection`: `:xy` (default), `:xz`, `:yz`, or `:xyz`.
- `xlabel`, `ylabel`, `zlabel`: optional labels; defaults follow the projection.
- `width`: (`::Real`, `=5`) figure's width
- `height`: (`::Real`, `=5`) figure's height
- `markersize`: (`::Real`, `=0.5`) size of the markers
- `cmap`: (`::String`, `="gray"`) colormap to be used for plotting
- `fontsize_label`: (`::Integer`, `=6`) font size of the labels
- `fontsize_ticklabel`: (`::Integer`, `=4`) font size of the tick labels
- `color_facecolor`: (`::String`, `="#ffffff"`) background color of the figure
- `color_label`: (`::String`, `="#000000"`) color of the labels and tick labels
- `linewidth`: (`::Real`, `=0.5`) width of the lines
- `ticklength`: (`::Real`, `=1.5`) length of the ticks
- `pad_label`: (`::Real`, `=2`) padding of the labels
- `pad_labeltick`: (`::Real`, `=2`) padding of the tick labels

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_traj(rand(100, 3); projection=:xyz)
julia> fig.savefig("123.png",bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_traj(
    ktraj::AbstractArray{<:Real, 2};
    projection         = :xy      ,
    width              = 8        ,
    height             = 8        ,
    markersize         = 0.5      ,
    cmap               = "viridis",
    xlabel             = nothing  ,
    ylabel             = nothing  ,
    zlabel             = nothing  ,
    fontsize_label     = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#ffffff",
    color_label        = "#000000",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    )
    matplotlib.rc("mathtext", default="regular")
    nPoint, nTerm = size(ktraj)
    @assert nTerm == 3 "ktraj should have shape [nPoint, 3] with columns [kx, ky, kz]"
    axes = trajectory_projection(projection)
    labels = ("kx [cycles/m]", "ky [cycles/m]", "kz [cycles/m]")
    xlabel = something(xlabel, labels[axes[1]])
    ylabel = something(ylabel, labels[axes[2]])

    fig = plt.figure(figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor)
    if projection == :xyz
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
            axis.set_pane_color(color_facecolor)
        end
    else
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal", "box")
    end
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    hide_plot_spines!(ax)
    ax.scatter((ktraj[:, i] for i in axes)...; linewidths=linewidth,
        marker=".", s=markersize, c=collect(1:nPoint), cmap=cmap)

    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    if projection == :xyz
        ax.set_zlabel(something(zlabel, labels[3]), fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    end

    fig.tight_layout(pad=0.1, w_pad=0, h_pad=0)
    return fig
end

function plt_traj(
    trajs::AbstractVector{T};
    projection         = :xy      ,
    labels             = nothing  ,
    xlabel             = nothing  ,
    ylabel             = nothing  ,
    zlabel             = nothing  ,
    width              = 8        ,      
    height             = 8        ,  
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#ffffff",
    color_label        = "#000000",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
) where T<:AbstractArray{<:Real, 2}
    matplotlib.rc("mathtext", default="regular")
    axes = trajectory_projection(projection)
    axis_labels = ("kx [cycles/m]", "ky [cycles/m]", "kz [cycles/m]")
    xlabel = something(xlabel, axis_labels[axes[1]])
    ylabel = something(ylabel, axis_labels[axes[2]])
    @assert all(size(traj, 2) >= maximum(axes) for traj in trajs) "Each trajectory needs one column per projected coordinate"
    if !isnothing(labels)
        @assert length(trajs) == length(labels) "labels should have the same length as ys"
    else
        labels = ["traj $(idx)" for idx = 1:length(trajs)] 
    end
    fig = plt.figure(figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor)
    if projection == :xyz
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
            axis.set_pane_color(color_facecolor)
        end
    else
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal", "box")
    end
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    hide_plot_spines!(ax)

    for (traj, label) in zip(trajs, labels)
        ax.plot((traj[:, i] for i in axes)...; label, linewidth)
    end
    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    if projection == :xyz
        ax.set_zlabel(something(zlabel, axis_labels[3]), fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    end
    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, scatteryoffsets=[0.5],
            # loc="upper left", bbox_to_anchor=(0.01, 1.00), ncols=5, 
            frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    fig.tight_layout(pad=0.1, w_pad=0, h_pad=0)
    return fig
end
