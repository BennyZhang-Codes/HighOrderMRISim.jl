# Share the same version validation between the builder and publisher.
function documentation_version(ref::AbstractString)
    ref == "dev" && return "dev"
    tag_pattern = r"""(?x)\A
        v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)
        (?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)
            (?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?
        (?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?
        \z"""
    occursin(tag_pattern, ref) ||
        throw(ArgumentError("Expected dev or a semantic version tag, got $(repr(ref))"))
    VersionNumber(ref[2:end])
    return first(split(ref, '+'; limit=2))
end
