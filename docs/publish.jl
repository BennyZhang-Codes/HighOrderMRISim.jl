using Documenter
using TOML

include("versioning.jl")

const DOCS_REPOSITORY = "BennyZhang-Codes/HighOrderMRISim.jl"

# An existing release can be built by a newer workflow without moving its tag.
struct ManualRelease <: Documenter.DeployConfig
    tag::String
end

function validate_deployment_context(doc_ref;
        repository=get(ENV, "GITHUB_REPOSITORY", ""),
        event=get(ENV, "GITHUB_EVENT_NAME", ""),
        ref=get(ENV, "GITHUB_REF", ""))
    documentation_version(doc_ref)
    repository == DOCS_REPOSITORY || error("Unexpected documentation repository")
    if doc_ref == "dev"
        event in ("push", "workflow_dispatch") && ref == "refs/heads/dev" ||
            error("Development documentation must be published from dev")
    elseif event == "workflow_dispatch"
        ref in ("refs/heads/main", "refs/heads/dev") ||
            error("Release backfills must run from main or dev")
    else
        event == "push" && ref == "refs/tags/$(doc_ref)" ||
            error("Release documentation must match the pushed tag")
    end
    return nothing
end

function Documenter.deploy_folder(cfg::ManualRelease; repo, branch, kwargs...)
    validate_deployment_context(cfg.tag)
    get(ENV, "GITHUB_EVENT_NAME", "") == "workflow_dispatch" ||
        error("ManualRelease requires workflow_dispatch")
    get(ENV, "RELEASE_TAG", "") == cfg.tag || error("The manual release tag does not match the input")
    repo == "github.com/$(DOCS_REPOSITORY).git" || error("Unexpected deployment repository")
    return Documenter.DeployDecision(
        all_ok=true, branch=branch, repo=repo,
        subfolder=documentation_version(cfg.tag), is_preview=false,
    )
end

Documenter.authentication_method(::ManualRelease) = Documenter.HTTPS
Documenter.authenticated_repo_url(::ManualRelease) =
    Documenter.authenticated_repo_url(Documenter.GitHubActions())

function validate_documentation_build(root, target, doc_ref, source_sha)
    documentation_version(doc_ref)
    occursin(r"\A[0-9a-f]{40}\z", source_sha) || error("Invalid source commit")
    readchomp(`git -C $root rev-parse HEAD`) == source_sha ||
        error("The source checkout does not match the built commit")
    success(`git -C $root diff --quiet HEAD --`) || error("The source checkout is modified")
    isfile(joinpath(target, "index.html")) || error("Missing documentation index")
    # Documenter requires the build directory to be inside the source docs root.
    relative = relpath(realpath(target), realpath(root))
    (relative == "." || first(splitpath(relative)) == "..") &&
        error("The artifact must be inside the source documentation directory")
    info = TOML.parsefile(joinpath(target, "build-info.toml"))
    info == Dict("ref" => doc_ref, "source_sha" => source_sha) ||
        error("The documentation artifact does not match its source")
    return nothing
end

function validate_remote_source(doc_ref, source_sha)
    repo = "https://github.com/$(DOCS_REPOSITORY).git"
    ref = doc_ref == "dev" ? "refs/heads/dev" : "refs/tags/$(doc_ref)"
    lines = split(readchomp(`git ls-remote $repo $ref $(ref * "^{}")`), '\n'; keepempty=false)
    refs = Dict(ref => sha for (sha, ref) in split.(lines))
    get(refs, ref * "^{}", get(refs, ref, nothing)) == source_sha ||
        error("The remote ref changed or does not match the built commit; rebuild before publishing")
    return nothing
end

function publish_documentation(root, target, doc_ref, source_sha)
    root, target = abspath(root), abspath(target)
    validate_deployment_context(doc_ref)
    validate_documentation_build(root, target, doc_ref, source_sha)
    validate_remote_source(doc_ref, source_sha)
    cfg = if doc_ref != "dev" && ENV["GITHUB_EVENT_NAME"] == "workflow_dispatch"
        ManualRelease(doc_ref)
    else
        Documenter.GitHubActions()
    end
    deploydocs(
        root=root, target=target,
        repo="github.com/$(DOCS_REPOSITORY).git",
        devbranch="dev", devurl="dev",
        versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
        deploy_config=cfg, push_preview=false, forcepush=false,
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 4 || error("Usage: publish.jl SOURCE_DOCS_DIR ARTIFACT_DIR VERSION SOURCE_SHA")
    publish_documentation(ARGS...)
end
