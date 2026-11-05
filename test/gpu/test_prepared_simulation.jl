using Test, CUDA, HighOrderMRISim
isdefined(@__MODULE__, :rf_acquisition_sequence) ||
    include(joinpath(@__DIR__, "..", "support", "compatibility_workload.jl"))

function test_prepared_simulation(ids)
    obj = HO_Phantom(x=collect(range(-0.01,0.01;length=11)),
        ρ=collect(range(0.5,1.0;length=11)),T1=fill(0.9,11),T2=fill(0.07,11),
        Δw=collect(range(-20.0,30.0;length=11)),csm=[cis(i*j) for i in 1:11,j in 1:3])
    hoseq = HO_Sequence(rf_acquisition_sequence(3,2,11))
    @testset "Prepared simulation lifecycle" begin
        for precision in ("f32","f64"), mode in ("gemm","reduction","fused_warp","fused_tiled")
            params = Dict{String,Any}("sim_method"=>BlochHighOrder("1111"),"precision"=>precision,
                "return_type"=>"mat","implementation"=>"kernel","coil_combination"=>mode,
                "max_block_length"=>4,"max_rf_block_length"=>5)
            baseline = simulate(obj,hoseq,Scanner();sim_params=params,gpus=ids,verbose=false)
            fenced = simulate(obj,hoseq,Scanner();sim_params=merge(params,Dict("synchronization"=>"block")),
                              gpus=ids,verbose=false)
            @test baseline == fenced
            expected_state = simulate(obj,hoseq,Scanner();sim_params=merge(params,Dict("return_type"=>"state")),
                                      gpus=ids,verbose=false)
            plan = prepare_simulation(obj,hoseq,Scanner();sim_params=params,gpus=ids)
            try
                @test all(!istaskdone(w.task) for w in plan.workers)
                first_run = simulate(plan;verbose=false)
                @test first_run == baseline
                saved_density = copy(obj.ρ)
                obj.ρ .*= 2
                @test simulate(plan;verbose=false) == baseline
                obj.ρ .= saved_density
                @test simulate(plan;verbose=false) == first_run
                state = simulate(plan;return_type="state",verbose=false)
                @test state.xy == expected_state.xy
                @test state.z == expected_state.z
                raw = simulate(plan;return_type="raw",verbose=false)
                @test sum(size(p.data,1) for p in raw.profiles) == 16
                before = copy(first_run)
                @test simulate(plan;verbose=false) == before
                @test first_run == before # Returned arrays cannot alias reusable host buffers.
                @test_throws ArgumentError simulate(plan;return_type="invalid",verbose=false)
                fail = (_,_,data,_) -> begin
                    fill!(data.obj.ρ,zero(eltype(data.obj.ρ)))
                    fill!(data.Xt.xy,zero(eltype(data.Xt.xy)))
                    error("intentional prepared callback failure")
                end
                @test_throws CompositeException simulate(plan;callbacks=(fail,),verbose=false)
                @test simulate(plan;verbose=false) == baseline
                # A callback must observe completed data and may enqueue a
                # state edit that is consumed by the next block on this stream.
                observe = (_,_,data,_) -> begin
                    @test all(isfinite,Array(data.Xt.xy))
                    data.Xt.xy .*= eltype(data.Xt.xy)(0,1)
                    nothing
                end
                changed = simulate(plan;callbacks=(observe,),verbose=false)
                changed_reference = simulate(obj,hoseq,Scanner();
                    sim_params=merge(params,Dict("synchronization"=>"block")),
                    gpus=ids,callbacks=(observe,),verbose=false)
                @test changed == changed_reference
                recurse = (_,_,_,_) -> simulate(plan;verbose=false)
                @test_throws CompositeException simulate(plan;callbacks=(recurse,),verbose=false)
                @test simulate(plan;verbose=false) == baseline
            finally
                close(plan)
            end
            @test plan.closed && all(istaskdone(w.task) for w in plan.workers)
            @test isnothing(close(plan))
            @test_throws ArgumentError simulate(plan;verbose=false)
            result = prepare_simulation(obj,hoseq,Scanner();sim_params=params,gpus=ids) do p
                simulate(p;verbose=false)
            end
            @test result == baseline
        end
    end
    @testset "Prepared RF without acquisition" begin
        empty_sequence = HO_Sequence(rf_acquisition_sequence(0,0,0))
        for precision in ("f32","f64")
            params = Dict{String,Any}("sim_method"=>BlochHighOrder("1111"),
                "precision"=>precision,"return_type"=>"state","synchronization"=>"block")
            reference = simulate(obj,empty_sequence,Scanner();sim_params=params,gpus=ids,verbose=false)
            params["synchronization"] = "auto"
            prepare_simulation(obj,empty_sequence,Scanner();sim_params=params,gpus=ids) do plan
                @test isempty(simulate(plan;return_type="mat",verbose=false))
                state = simulate(plan;return_type="state",verbose=false)
                @test state.xy == reference.xy
                @test state.z == reference.z
            end
        end
    end
    return nothing
end
