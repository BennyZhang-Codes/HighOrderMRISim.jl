using Test
using HighOrderMRISim
import PyPlot

@testset "Plot backgrounds and preserved colors" begin
    seq = Sequence()
    @addblock seq += RF(1e-6, 1e-4)
    @addblock seq += ADC(4, 1e-3)
    hoseq = HO_Sequence(seq)
    hoseq.GR_dfc[2, 2] = Grad(1e-3, 1e-3)
    hoseq.GR_dfc[5, 2] = Grad(0.05, 1e-3)
    sampled = discretize(hoseq)
    coefficients = reshape(collect(1.0:48.0), 3, 16)
    trajectory = [0.0 0.0 0.0; 1.0 2.0 3.0; 2.0 1.0 4.0]
    ph = HO_Phantom(x=[-0.01, 0.01], y=[-0.02, 0.02], z=[-0.03, 0.03])
    mag = HighOrderMRISim.Mag(ComplexF64[1, im], [0.0, 1.0])
    grid = generate_grid([0.01, 0.01, 0.01], [3, 3, 3])
    image = reshape(collect(1.0:4.0), 2, 2)
    images = reshape(collect(1.0:8.0), 2, 2, 2)
    plots = (
        "curves" => kwargs -> plt_plot([[1.0, 2.0]]; labels=["one"], kwargs...),
        "scatter" => kwargs -> plt_scatter([[1.0, 2.0]]; labels=["one"], kwargs...),
        "sequence" => kwargs -> plt_seq(hoseq; kwargs...),
        "sampled sequence" => kwargs -> plt_seqd(hoseq; kwargs...),
        "sampled arrays" => kwargs -> plt_seqd(sampled; kwargs...),
        "fields" => kwargs -> plt_bfield(coefficients, 1e-4; kwargs...),
        "field comparison" => kwargs -> plt_bfield_com(coefficients, coefficients, 1e-4; kwargs...),
        "encoding" => kwargs -> plt_kspha(coefficients, 1e-4; kwargs...),
        "encoding comparison" => kwargs -> plt_ksphas([coefficients], 1e-4; kwargs...),
        "trajectory 2D" => kwargs -> plt_traj(trajectory; kwargs...),
        "trajectory 3D" => kwargs -> plt_traj(trajectory; projection=:xyz, kwargs...),
        "trajectories 2D" => kwargs -> plt_traj([trajectory]; kwargs...),
        "trajectories 3D" => kwargs -> plt_traj([trajectory]; projection=:xyz, kwargs...),
        "k-space" => kwargs -> plt_kspace(hoseq; source=:both, kwargs...),
        "phantom 2D" => kwargs -> plt_phantom(ph, :ρ; view_2d=true, kwargs...),
        "phantom 3D" => kwargs -> plt_phantom(ph, :ρ; kwargs...),
        "magnetization 2D" => kwargs -> plt_magnetization(ph, mag, :xy_mag; view_2d=true, kwargs...),
        "magnetization 3D" => kwargs -> plt_magnetization(ph, mag, :xy_mag; kwargs...),
        "grid" => kwargs -> plt_grid(grid; step=1, plot_reference=false, kwargs...),
        "B0" => kwargs -> plt_B0map(image; kwargs...),
        "image" => kwargs -> plt_image(image; kwargs...),
        "images" => kwargs -> plt_images(images; nRow=2, nCol=2, kwargs...),
    )
    for (name, plot) in plots
        @testset "$name" begin
            for (kwargs, expected) in (( (;), (1.0, 1.0, 1.0, 1.0)),
                    ((; color_facecolor="#203040", color_label="#abcdef"),
                        (32/255, 48/255, 64/255, 1.0)),
                    ((; color_facecolor="none"), (0.0, 0.0, 0.0, 0.0)))
                fig = plot(kwargs)
                try
                    @test collect(fig.get_facecolor()) ≈ collect(expected)
                    for ax in fig.axes
                        @test collect(ax.get_facecolor()) ≈ collect(expected)
                        if ax.name == "3d"
                            for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
                                @test collect(axis.pane.get_facecolor()) ≈ collect(expected)
                            end
                        end
                    end
                    fig.canvas.draw()
                finally
                    PyPlot.close(fig)
                end
            end
        end
    end

    for plot in (plt_bfield, plt_kspha)
        fig = plot(coefficients, 1e-4)
        try
            lines = [line for ax in fig.axes for line in ax.get_lines()]
            @test length(lines) == 16
            for (term, line) in enumerate(lines)
                @test line.get_color() == HighOrderMRISim.spha_color[term]
                @test collect(line.get_ydata()) == coefficients[:, term]
            end
        finally
            PyPlot.close(fig)
        end
    end
    fig = plt_seq(hoseq)
    try
        lines = Dict(line.get_label() => line for line in first(fig.axes).get_lines())
        for (label, color) in (("Gx nominal [mT/m]", "#636EFA"),
                ("Gy nominal [mT/m]", "#EF553B"), ("Gz nominal [mT/m]", "#00CC96"),
                ("|B1| channel 1 [μT]", "#AB63FA"), ("ADC [on/off]", "#19D3F3"))
            @test lines[label].get_color() == color
        end
        for term in 0:15
            field = getproperty(SphericalHarmonics(), Symbol("h", term))
            unit = replace(field.unit, "T" => "mT")
            @test lines["h$term field [$unit]"].get_color() == field.color
        end
        mktempdir() do directory
            filename = joinpath(directory, "transparent.png")
            fig.savefig(filename; transparent=true)
            pixels = PyPlot.imread(filename)
            @test size(pixels, 3) == 4
            @test pixels[1, 1, 4] == 0
            @test Tuple(fig.get_facecolor()) == (1.0, 1.0, 1.0, 1.0)
        end
    finally
        PyPlot.close(fig)
    end
end
