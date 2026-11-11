using Test
using HighOrderMRISim

@testset "README CPU example" begin
    readme = read(joinpath(@__DIR__, "..", "..", "README.md"), String)
    usage = split(readme, "## Usage"; limit=2)[2]
    example = match(r"```julia\n(.*?)\n```"s, usage)
    @test !isnothing(example)
    scope = Module(:ReadmeSimulationExample)
    Base.include_string(scope, only(example.captures), "README CPU example")
    signal = getfield(scope, :signal)
    @test size(signal) == (8, 1, 1)
    @test eltype(signal) == ComplexF64
    @test all(isfinite, signal)
    @test !isapprox(first(signal), last(signal))
end
