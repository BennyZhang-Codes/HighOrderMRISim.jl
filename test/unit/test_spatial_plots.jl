using Test
using HighOrderMRISim
import PyPlot

@testset "Spatial and trajectory plots" begin
    @testset "Nominal and measured k-space" begin
        τ = 1e-3
        gradients = [Grad(axis*1e-3, τ) for axis in 1:3, _ in 1:2]
        seq = Sequence(gradients, reshape([RF(0.0, 0.0), RF(0.0, 0.0)], 1, :),
            [ADC(0, 0.0), ADC(5, τ)], [τ, τ])
        hoseq = HO_Sequence(seq)
        hoseq.GR_dfc[2:4, :] .= 2 .* seq.GR
        # A longer attached event changes the next block's absolute ADC times.
        hoseq.GR_dfc[1, 1] = Grad(1e-6, 2τ)
        nominal, nominal_adc, measured, measured_adc = get_kspace(hoseq)
        @test measured[:, 2:4] ≈ 2nominal
        t_adc = HighOrderMRISim.get_adc_sampling_times(
            HighOrderMRISim.simulation_sequence(hoseq)) .* 1e3
        @test first(t_adc) >= 2τ*1e3
        for source in (:nominal, :measured, :both)
            fig = plt_kspace(hoseq; source)
            try
                ax = first(fig.axes)
                lines, samples = collect(ax.lines), collect(ax.collections)
                expected = source == :nominal ? ((nominal, nominal_adc, "Nominal"),) :
                    source == :measured ? ((measured[:, 2:4], measured_adc[:, 2:4], "Measured"),) :
                    ((nominal, nominal_adc, "Nominal"), (measured[:, 2:4], measured_adc[:, 2:4], "Measured"))
                @test length(lines) == length(samples) == length(expected)
                for (line, sample, (k, k_adc, label)) in zip(lines, samples, expected)
                    @test hcat(collect.(line.get_data_3d())...) ≈ k
                    @test hcat(collect.(sample._offsets3d)...) ≈ k_adc
                    @test collect(sample.get_array()) ≈ t_adc
                    @test line.get_label() == label
                    for (axis, limits) in enumerate((ax.get_xlim(), ax.get_ylim(), ax.get_zlim()))
                        @test limits[1] <= minimum(k[:, axis]) <= maximum(k[:, axis]) <= limits[2]
                    end
                end
                @test ax.get_xlabel() == "kx [cycles/m]"
                @test ax.get_ylabel() == "ky [cycles/m]"
                @test ax.get_zlabel() == "kz [cycles/m]"
                fig.canvas.draw()
            finally
                PyPlot.close(fig)
            end
        end
        fig = plt_kspace(hoseq)
        try
            @test only(collect(first(fig.axes).lines)).get_label() == "Nominal"
        finally
            PyPlot.close(fig)
        end
        fig = plt_kspace(HO_Sequence(gradients); source=:both)
        try
            @test isempty(collect(first(fig.axes).collections))
            @test length(collect(first(fig.axes).lines)) == 2
        finally
            PyPlot.close(fig)
        end
        @test_throws ArgumentError plt_kspace(hoseq; source=:effective)
    end

    @testset "Trajectory projections: $T" for T in (Float32, Float64)
        trajectory = T[1 2 3; 4 7 10; -2 3 8; 5 -3 2]
        for (projection, axes) in ((:xy, (1, 2)), (:xz, (1, 3)), (:yz, (2, 3)), (:xyz, (1, 2, 3)))
            fig = plt_traj(trajectory; projection)
            try
                ax = first(fig.axes)
                sample = only(collect(ax.collections))
                actual = projection == :xyz ? hcat(collect.(sample._offsets3d)...) : Array(sample.get_offsets())
                @test actual ≈ trajectory[:, collect(axes)]
                names = ("kx [cycles/m]", "ky [cycles/m]", "kz [cycles/m]")
                @test ax.get_xlabel() == names[axes[1]]
                @test ax.get_ylabel() == names[axes[2]]
                projection == :xyz && @test ax.get_zlabel() == names[3]
                fig.canvas.draw()
            finally
                PyPlot.close(fig)
            end
            fig = plt_traj([trajectory, 2trajectory]; projection, labels=["one", "two"])
            try
                ax = first(fig.axes)
                lines = collect(ax.lines)
                @test length(lines) == 2
                for (index, line) in enumerate(lines)
                    data = projection == :xyz ? line.get_data_3d() : line.get_data()
                    @test hcat(collect.(data)...) ≈ index .* trajectory[:, collect(axes)]
                end
            finally
                PyPlot.close(fig)
            end
        end
        fig = plt_traj(trajectory; xlabel="custom x", ylabel="custom y")
        try
            ax = first(fig.axes)
            @test Array(only(collect(ax.collections)).get_offsets()) ≈ trajectory[:, 1:2]
            @test ax.get_xlabel() == "custom x"
            @test ax.get_ylabel() == "custom y"
        finally
            PyPlot.close(fig)
        end
        @test_throws AssertionError plt_traj(zeros(T, 3, 5))
        @test_throws ArgumentError plt_traj(trajectory; projection=:invalid)
        fig = plt_traj([trajectory[:, 1:2]])
        try
            @test hcat(collect.(only(collect(first(fig.axes).lines)).get_data())...) ≈ trajectory[:, 1:2]
        finally
            PyPlot.close(fig)
        end
    end

    @testset "Phantom coordinates and display units: $T" for T in (Float32, Float64)
        ph = HO_Phantom(x=[0.01, -0.02], y=[0.03, 0.01], z=[-0.04, 0.02],
            ρ=[1.0, 0.5], Δw=2π .* [10.0, 20.0], csm=reshape(ComplexF64[1, im], :, 1),
            ux=(x, y, z, t) -> t, uy=(x, y, z, t) -> 2t, uz=(x, y, z, t) -> -3t)
        ph = T === Float32 ? HighOrderMRISim.f32(ph) : HighOrderMRISim.f64(ph)
        xyz = hcat(ph.x .+ 0.1, ph.y .+ 0.2, ph.z .- 0.3) .* 1e2
        for (axis, indices, labels) in (("axial", (1, 2), ("x [cm]", "y [cm]")),
                ("coronal", (1, 3), ("x [cm]", "z [cm]")),
                ("sagittal", (2, 3), ("y [cm]", "z [cm]")))
            fig = plt_phantom(ph, :Δw; view_2d=true, axis, t0=100)
            try
                ax = first(fig.axes)
                sample = only(collect(ax.collections))
                @test Array(sample.get_offsets()) ≈ xyz[:, collect(indices)]
                @test collect(sample.get_array()) ≈ [10.0, 20.0] rtol=8eps(T)
                @test ax.get_xlabel() == labels[1]
                @test ax.get_ylabel() == labels[2]
                @test last(fig.axes).get_title() == "[Δw] Hz"
                @test ax.get_xlim()[2] >= maximum(abs, xyz)
                fig.canvas.draw()
            finally
                PyPlot.close(fig)
            end
        end
        fig = plt_phantom(ph, :csm_pha; t0=100, xlabel="custom x", ylabel="custom y")
        try
            ax = first(fig.axes)
            @test hcat(collect.(only(collect(ax.collections))._offsets3d)...) ≈ xyz
            @test ax.get_xlabel() == "custom x"
            @test ax.get_ylabel() == "custom y"
            @test last(fig.axes).get_title() == "[csm_pha] rad"
        finally
            PyPlot.close(fig)
        end
    end

    @testset "Explicit image color limits" begin
        img = reshape(collect(1.0:6.0), 2, 3)
        imgs = reshape(collect(1.0:12.0), 2, 2, 3)
        for (plot, data) in ((plt_image, img), (plt_images, imgs))
            for (limits, expected) in (( (; vmin=-2.0), (-2.0, maximum(data))),
                    ((; vmax=20.0), (minimum(data), 20.0)),
                    ((; vmin=-3.0, vmax=30.0), (-3.0, 30.0)))
                fig = plot(data; limits...)
                try
                    for ax in fig.axes
                        @test only(collect(ax.images)).get_clim() == expected
                    end
                    fig.canvas.draw()
                finally
                    PyPlot.close(fig)
                end
            end
        end
    end
end
