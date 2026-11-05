using Test, HighOrderMRISim

@testset "Sampled high-order storage" begin
    h = discretize(HO_Sequence(compatibility_sequence()))
    fields = HighOrderMRISim.high_order_coefficients(h)
    @test fields == ntuple(i -> getfield(h, Symbol("h",i-1)), 16)
    @test isconcretetype(fieldtype(typeof(h), :seqd))
    @test isconcretetype(fieldtype(typeof(h), :h0))
    @test (@inferred HighOrderMRISim.high_order_coefficients(h)) === fields
    @test HO_DiscreteSequence{Float64}(h.seqd, fields...).h0 === h.h0
    nodes = 1:3
    block = @inferred view(h, nodes)
    @test block.h0 == h.h0[nodes]
    @test parent(block.h0) === h.h0
    @test block.seqd.Δt == h.seqd.Δt[1:2]
    @test all(eltype(v) === Float32 for v in HighOrderMRISim.high_order_coefficients(f32(h)))
    mixed_fields = (view(first(fields),:), Base.tail(fields)...)
    mixed = HO_DiscreteSequence(h.seqd, mixed_fields...)
    @test mixed.h0 === first(mixed_fields)
    @test mixed.h1 === fields[2]
    @test view(mixed,nodes).h0 == block.h0
end
