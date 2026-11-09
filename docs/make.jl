using Documenter
using HighOrderMRISim
using TOML

include("versioning.jl")

length(ARGS) <= 2 || error("Usage: make.jl [DOCS_ROOT [VERSION]]")
docs_root = isempty(ARGS) ? (@__DIR__) : abspath(ARGS[1])
docs_ref = length(ARGS) < 2 ? "dev" : ARGS[2]
docs_version = documentation_version(docs_ref)
realpath(pkgdir(HighOrderMRISim)) == realpath(dirname(docs_root)) ||
    error("The loaded package does not match the documentation source checkout")
source_sha = readchomp(`git -C $docs_root rev-parse HEAD`)

makedocs(
    root = docs_root,
    sitename = "HighOrderMRISim.jl",
    modules = [HighOrderMRISim],
    repo = Documenter.Remotes.GitHub("BennyZhang-Codes", "HighOrderMRISim.jl"),
    # The API page covers the simulation interface, not every legacy utility.
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = false,
        canonical = "https://bennyzhang-codes.github.io/HighOrderMRISim.jl/$(docs_version)/",
        edit_link = docs_ref == "dev" ? "dev" : nothing,
        inventory_version = docs_version == "dev" ? "dev" : docs_version[2:end],
        repolink = "https://github.com/BennyZhang-Codes/HighOrderMRISim.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Guides" => [
            "High-Order Simulation" => "guide/simulation.md",
            "2D and 3D Demos" => "guide/demos.md",
        ],
        "Migration" => "migration.md",
        "API Reference" => "api.md",
    ],
)

open(joinpath(docs_root, "build", "build-info.toml"), "w") do io
    TOML.print(io, Dict("ref" => docs_ref, "source_sha" => source_sha))
end
