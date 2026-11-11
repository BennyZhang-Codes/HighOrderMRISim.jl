# Build a new demo environment without changing the release lockfile or default project.
using Pkg
using TOML

function instantiate_checkout(environment)
    environment = abspath(environment)
    if ispath(environment)
        isdir(environment) && isempty(readdir(environment)) ||
            error("The destination must be a new or empty directory: $environment")
    end
    root = dirname(@__DIR__)
    project = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
    # Pkg.develop cannot combine a requested path with an existing URL source.
    project["sources"]["HighOrderMRISim"] = Dict("path" => root)
    mkpath(environment)
    open(joinpath(environment, "Project.toml"), "w") do io
        TOML.print(io, project)
    end
    manifest = TOML.parsefile(joinpath(@__DIR__, "Manifest.toml"))
    # Pkg 1.12 must resolve a source subdirectory from its pinned revision,
    # not descend into an already-resolved subtree hash a second time.
    source_trees = Dict{String,String}()
    for (name, source) in project["sources"]
        if haskey(source, "subdir")
            entry = only(manifest["deps"][name])
            source_trees[name] = pop!(entry, "git-tree-sha1")
        end
    end
    open(joinpath(environment, "Manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    Pkg.activate(environment)
    Pkg.develop(path=root; preserve=Pkg.PRESERVE_ALL)
    Pkg.instantiate()
    resolved = TOML.parsefile(joinpath(environment, "Manifest.toml"))
    for (name, tree) in source_trees
        only(resolved["deps"][name])["git-tree-sha1"] == tree ||
            error("The pinned source tree changed for $name")
    end
    Pkg.status(["HighOrderMRISim", "HighOrderMRI", "MRIGeometry"])
    println("CHECKOUT_DEMO_ENVIRONMENT: ", environment)
    return environment
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: instantiate_checkout.jl [NEW_ENVIRONMENT_DIRECTORY]")
    destination = isempty(ARGS) ? mktempdir(; cleanup=false) : only(ARGS)
    instantiate_checkout(destination)
end
