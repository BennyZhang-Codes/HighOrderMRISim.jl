using Documenter
using HighOrderMRISim

makedocs(
    root = @__DIR__,
    sitename = "HighOrderMRISim.jl",
    modules = [HighOrderMRISim],
    remotes = nothing,
    # The API page covers the simulation interface, not every legacy utility.
    checkdocs = :none,
    format = Documenter.HTML(prettyurls = false, edit_link = nothing, repolink = nothing),
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
