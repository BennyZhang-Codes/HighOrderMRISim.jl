using Test
using HighOrderMRISim
import PyPlot

@testset "Field coefficient plots" begin
    # Matplotlib artists expose the plotted values directly; no pixel snapshots
    # or interactive display are needed to check channel order, units, and time.
    for T in (Float32, Float64), nTerm in (9, 16)
        values = T.([sample + 10term for sample in 1:3, term in 1:nTerm])
        original = copy(values)
        times = T[0, 2e-4, 7e-4]
        for (plot, units) in ((plt_bfield, :T), (plt_kspha, :cycles))
            fig = plot(values, T(1e-4); units, times)
            try
                axes = collect(fig.axes)
                @test length(axes) == (nTerm == 9 ? 3 : 4)
                @test [length(ax.get_lines()) for ax in axes] == (nTerm == 9 ? [1, 3, 5] : [1, 3, 5, 7])
                term = 1
                for (order, ax) in enumerate(axes)
                    suffix = order == 1 ? "" : order == 2 ? "/m" : order == 3 ? "/m²" : "/m³"
                    @test occursin("[$units$suffix]", ax.get_ylabel())
                    for line in ax.get_lines()
                        @test collect(line.get_ydata()) == values[:, term]
                        @test collect(line.get_xdata()) == times .* 1e3
                        term += 1
                    end
                end
                @test values == original
                fig.canvas.draw()
            finally
                PyPlot.close(fig)
            end
        end

        fig = plt_bfield_com(values, -values, T(1e-4); units=:T, times)
        try
            axes = collect(fig.axes)
            @test length(axes) == nTerm
            for (term, ax) in enumerate(axes)
                lines = collect(ax.get_lines())
                @test length(lines) == 2
                @test collect(lines[1].get_ydata()) == values[:, term]
                @test collect(lines[2].get_ydata()) == -values[:, term]
                @test collect(lines[1].get_xdata()) == times .* 1e3
            end
            @test occursin(nTerm == 9 ? "T/m²" : "T/m³", axes[end].get_ylabel())
            fig.canvas.draw()
        finally
            PyPlot.close(fig)
        end
    end

    # Real demo files may store dt as Float32. Both field plotters must accept
    # that interval without converting the coefficient values.
    for T in (Float32, Float64), comparison in (false, true)
        values = reshape(T.(1:48), 3, 16)
        original = copy(values)
        dt = T(1e-4)
        fig = comparison ? plt_bfield_com(values, -values, dt) : plt_bfield(values, dt)
        try
            line = first(first(fig.axes).get_lines())
            @test collect(line.get_xdata()) == (1:3) .* dt .* 1e3
            @test collect(line.get_ydata()) == values[:, 1]
            @test values == original
        finally
            PyPlot.close(fig)
        end
    end

    values = reshape(collect(1.0:48.0), 3, 16)
    # Legacy raw-array defaults remain mT and radians, without conversion.
    for (plot, unit) in ((plt_bfield, "mT"), (plt_kspha, "rad"))
        fig = plot(values, 1e-4)
        try
            ax = first(fig.axes)
            line = first(ax.get_lines())
            @test occursin("[$unit]", ax.get_ylabel())
            @test collect(line.get_ydata()) == values[:, 1]
            @test collect(line.get_xdata()) ≈ [0.1, 0.2, 0.3]
        finally
            PyPlot.close(fig)
        end
    end

    # Comparing different supported orders must not discard third-order data
    # or index columns that do not exist in the second-order input.
    fig = plt_ksphas([values[:, 1:9], values], 1e-4; units=:cycles)
    try
        axes = collect(fig.axes)
        @test length(axes) == 16
        @test [length(ax.get_lines()) for ax in axes] == [fill(2, 9); fill(1, 7)]
        @test collect(only(axes[16].get_lines()).get_ydata()) == values[:, 16]
        @test occursin("cycles/m³", axes[16].get_ylabel())
        fig.canvas.draw()
    finally
        PyPlot.close(fig)
    end

    @test_throws ArgumentError plt_kspha(values, 1e-4; units=:T)
    @test_throws ArgumentError plt_ksphas([values], 1e-4; units=:T)
    @test_throws ArgumentError plt_bfield(values, 1e-4; units=:rad)
    @test_throws ArgumentError plt_bfield_com(values, values, 1e-4; units=:rad)
    @test_throws ArgumentError plt_bfield(values, 0f0)
    @test_throws ArgumentError plt_bfield_com(values, values, -1f-4)
    @test_throws AssertionError plt_bfield(values[:, 1:10], 1e-4)
    @test_throws DimensionMismatch plt_bfield_com(values, values[:, 1:9], 1e-4)
    @test_throws DimensionMismatch plt_kspha(values, 1e-4; times=[0.0, 1e-4])
    @test_throws ArgumentError plt_bfield(values, 1e-4; times=[0.0, NaN, 1e-4])
    @test_throws ArgumentError plt_kspha(values, 1e-4; times=[0.0, 2e-4, 1e-4])
end

@testset "Sequence plot channels and units" begin
    nominal = Sequence(
        reshape(Grad[Grad(i*1e-3, 1e-3) for i in 1:3], :, 1),
        reshape(RF[RF(1e-6, 1e-3), RF(3im*1e-6, 1e-3)], :, 1),
        [ADC(2, 4e-4, 2e-4)],
    )
    hoseq = HO_Sequence(nominal, reshape(Grad[Grad(i*1e-6, 1e-3) for i in 1:16], :, 1))
    samples = get_samples(hoseq; off_val=Inf, max_rf_samples=Inf)
    fig = plt_seq(hoseq)
    try
        ax = first(fig.axes)
        lines = Dict(line.get_label() => line for line in ax.get_lines())
        for channel in 1:2
            line = lines["|B1| channel $channel [μT]"]
            amplitude = collect(line.get_ydata())
            @test maximum(amplitude) ≈ (channel == 1 ? 1.0 : 3.0)
            @test amplitude == (channel == 1 ? 1 : 3) .* abs.(samples.rf.A)*1e6
            @test collect(line.get_xdata()) == samples.rf.t*1e3
        end
        @test collect(lines["Gx nominal [mT/m]"].get_ydata()) == samples.gx.A*1e3
        @test collect(lines["h15 field [mT/m³]"].get_ydata()) == samples.h15.A*1e3
        @test haskey(lines, "B1 phase 2 [rad]")
        @test haskey(lines, "h0 field [mT]")
        @test occursin("units in legend", ax.get_ylabel())
        fig.canvas.draw()
    finally
        PyPlot.close(fig)
    end

    # The sampled path supports the currently implemented single transmit model.
    single = HO_Sequence(Sequence(nominal.GR, nominal.RF[1:1, :], nominal.ADC), hoseq.GR_dfc)
    sampled = discretize(single)
    for input in (single, sampled)
        fig = plt_seqd(input)
        try
            ax = first(fig.axes)
            lines = Dict(line.get_label() => line for line in ax.get_lines())
            @test collect(lines["h15 field [mT/m³]"].get_ydata()) == sampled.h15*1e3
            @test collect(lines["Gx nominal [mT/m]"].get_xdata()) == sampled.seqd.t*1e3
            @test haskey(lines, "|B1| [μT]")
            @test occursin("units in legend", ax.get_ylabel())
            fig.canvas.draw()
        finally
            PyPlot.close(fig)
        end
    end
end
