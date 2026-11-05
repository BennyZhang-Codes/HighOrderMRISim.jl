using Test, HighOrderMRISim

@testset "Simulation execution options" begin
    ranges = HighOrderMRISim.simulation_spin_ranges(11, 4)
    @test ranges == [1:2, 3:5, 6:8, 9:11]
    @test vcat(collect.(ranges)...) == collect(1:11)
    @test_throws ArgumentError HighOrderMRISim.simulation_spin_ranges(2, 3)
    obj, seq = compatibility_phantom(), HO_Sequence(compatibility_sequence())
    params = Dict{String,Any}("gpu" => true, "precision" => "f64", "Nthreads" => 1,
        "sim_method" => BlochHighOrder("0000"), "return_type" => "mat")
    original = copy(params)
    signal = simulate(obj, seq, Scanner(); sim_params=params, array_type=Array, verbose=false)
    @test size(signal) == (5, 2, 1)
    @test params == original
    @test_throws ArgumentError simulate(obj, seq, Scanner(); array_type=Array,
        sim_params=merge(params,Dict("synchronization"=>"invalid")),verbose=false)
    @test_throws ArgumentError prepare_simulation(obj, seq, Scanner();
        sim_params=params,array_type=Array)
    for (array_type, gpus) in ((Vector, nothing), (Array, [0]), (nothing, Int[]),
                                (nothing, [0, 0]), (nothing, [-1]), (nothing, [true]),
                                (nothing, [0.5]), (nothing, 0), (nothing, [0, 1, 2]))
        @test_throws ArgumentError simulate(obj, seq, Scanner(); sim_params=params,
                                            array_type, gpus, verbose=false)
    end
end
