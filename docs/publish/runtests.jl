using Test

include(joinpath(@__DIR__, "..", "publish.jl"))

@testset "Documentation version names" begin
    for (ref, version) in (
            "dev" => "dev",
            "v0.2.0" => "v0.2.0",
            "v12.34.56" => "v12.34.56",
            "v1.2.3-rc.1" => "v1.2.3-rc.1",
            "v1.2.3+docs.01" => "v1.2.3",
            "v1.2.3-rc.1+docs.2" => "v1.2.3-rc.1",
        )
        @test documentation_version(ref) == version
    end
    for ref in (
            "", "stable", "main", "0.2.0", "v1", "v1.2", "v01.2.3",
            "v1.02.3", "v1.2.03", "v1.2.3-01", "v1.2.3-rc.01",
            "v1.2.3-rc..1", "v1.2.3-", "v1.2.3+", "v1.2.3/",
            "../v1.2.3", "v1.2.3/../../dev", "v1.2.3\\dev",
            "v1.2.3?query", "v1.2.3\n", "dev\n", " v1.2.3",
        )
        @test_throws ArgumentError documentation_version(ref)
    end
    @test_throws OverflowError documentation_version("v1.2.3-" * repeat("9", 80))
end

@testset "Deployment event and ref policy" begin
    for doc_ref in ("dev", "v0.2.0", "v0.3.0-rc.1"),
            repository in (DOCS_REPOSITORY, "another-owner/HighOrderMRISim.jl"),
            event in ("push", "workflow_dispatch", "pull_request", "workflow_run"),
            ref in (
                "refs/heads/main", "refs/heads/dev", "refs/heads/feature",
                "refs/tags/v0.2.0", "refs/tags/v0.3.0-rc.1", "refs/pull/5/merge",
            )
        allowed = repository == DOCS_REPOSITORY && if doc_ref == "dev"
            event in ("push", "workflow_dispatch") && ref == "refs/heads/dev"
        elseif event == "workflow_dispatch"
            ref in ("refs/heads/main", "refs/heads/dev")
        else
            event == "push" && ref == "refs/tags/$(doc_ref)"
        end
        if allowed
            @test validate_deployment_context(doc_ref; repository, event, ref) === nothing
        else
            @test_throws ErrorException validate_deployment_context(doc_ref; repository, event, ref)
        end
    end
    @test_throws ArgumentError validate_deployment_context("../stable";
        repository=DOCS_REPOSITORY, event="push", ref="refs/heads/dev")
end

@testset "Manual release deployment decision" begin
    for ref in ("refs/heads/main", "refs/heads/dev")
        withenv(
                "GITHUB_REPOSITORY" => DOCS_REPOSITORY,
                "GITHUB_EVENT_NAME" => "workflow_dispatch",
                "GITHUB_REF" => ref,
                "RELEASE_TAG" => "v0.2.0+docs.1",
            ) do
            cfg = ManualRelease("v0.2.0+docs.1")
            decision = Documenter.deploy_folder(cfg;
                repo="github.com/$(DOCS_REPOSITORY).git", branch="gh-pages",
                devbranch="dev", devurl="dev", push_preview=false)
            @test decision.all_ok
            @test !decision.is_preview
            @test decision.branch == "gh-pages"
            @test decision.repo == "github.com/$(DOCS_REPOSITORY).git"
            @test decision.subfolder == "v0.2.0"
            @test Documenter.authentication_method(cfg) == Documenter.HTTPS
            @test_throws ErrorException Documenter.deploy_folder(cfg;
                repo="github.com/another-owner/HighOrderMRISim.jl.git", branch="gh-pages")
            for release_tag in (nothing, "", "v0.2.0", "v0.2.1")
                withenv("RELEASE_TAG" => release_tag) do
                    @test_throws ErrorException Documenter.deploy_folder(cfg;
                        repo="github.com/$(DOCS_REPOSITORY).git", branch="gh-pages")
                end
            end
        end
    end
    withenv(
            "GITHUB_REPOSITORY" => DOCS_REPOSITORY,
            "GITHUB_EVENT_NAME" => "push",
            "GITHUB_REF" => "refs/tags/v0.2.0",
            "RELEASE_TAG" => "v0.2.0",
        ) do
        @test_throws ErrorException Documenter.deploy_folder(ManualRelease("v0.2.0");
            repo="github.com/$(DOCS_REPOSITORY).git", branch="gh-pages")
    end
end

function source_fixture(directory)
    root = joinpath(directory, "source")
    docs = joinpath(root, "docs")
    mkpath(joinpath(docs, "src"))
    write(joinpath(root, ".gitignore"), "docs/build/\n")
    write(joinpath(docs, "src", "index.md"), "# Test manual\n")
    run(`git init --quiet --initial-branch=dev $root`)
    run(`git -C $root add .gitignore docs/src/index.md`)
    run(`git -C $root -c user.name=DocumentationTests -c user.email=docs@example.invalid -c commit.gpgsign=false commit --quiet -m "Create documentation fixture"`)
    sha = readchomp(`git -C $root rev-parse HEAD`)
    return (; root, docs, sha)
end

function write_artifact(target, ref, sha; text="Test manual")
    mkpath(target)
    write(joinpath(target, "index.html"), "<!doctype html><title>$(text)</title>\n")
    open(joinpath(target, "build-info.toml"), "w") do io
        TOML.print(io, Dict("ref" => ref, "source_sha" => sha))
    end
    return nothing
end

@testset "Artifact source provenance" begin
    mktempdir() do directory
        source = source_fixture(directory)
        target = joinpath(source.docs, "build")
        write_artifact(target, "dev", source.sha)
        @test success(`git -C $(source.root) check-ignore --quiet $target`)
        @test validate_documentation_build(source.docs, target, "dev", source.sha) === nothing

        # Downloaded artifacts may be untracked rather than ignored.
        untracked = joinpath(source.docs, "downloaded-artifact")
        write_artifact(untracked, "dev", source.sha)
        @test !success(`git -C $(source.root) check-ignore --quiet $untracked`)
        @test validate_documentation_build(source.docs, untracked, "dev", source.sha) === nothing

        for sha in (repeat("0", 40), "short", repeat("g", 40), source.sha * "\n")
            @test_throws ErrorException validate_documentation_build(source.docs, target, "dev", sha)
        end
        @test_throws ErrorException validate_documentation_build(source.docs, target, "v0.2.0", source.sha)

        write_artifact(target, "dev", repeat("0", 40))
        @test_throws ErrorException validate_documentation_build(source.docs, target, "dev", source.sha)
        write_artifact(target, "dev", source.sha)
        open(joinpath(target, "build-info.toml"), "a") do io
            println(io, "\nunexpected = true")
        end
        @test_throws ErrorException validate_documentation_build(source.docs, target, "dev", source.sha)
        write_artifact(target, "dev", source.sha)

        tracked = joinpath(source.docs, "src", "index.md")
        original = read(tracked, String)
        write(tracked, original * "Uncommitted change\n")
        @test_throws ErrorException validate_documentation_build(source.docs, target, "dev", source.sha)
        write(tracked, original)
        @test validate_documentation_build(source.docs, target, "dev", source.sha) === nothing

        missing = joinpath(source.docs, "empty-artifact")
        mkpath(missing)
        @test_throws ErrorException validate_documentation_build(source.docs, missing, "dev", source.sha)
        outside = joinpath(directory, "outside-artifact")
        write_artifact(outside, "dev", source.sha)
        @test_throws ErrorException validate_documentation_build(source.docs, outside, "dev", source.sha)
        link = joinpath(source.docs, "external-artifact")
        symlink(outside, link)
        @test_throws ErrorException validate_documentation_build(source.docs, link, "dev", source.sha)
        write_artifact(source.docs, "dev", source.sha)
        @test_throws ErrorException validate_documentation_build(source.docs, source.docs, "dev", source.sha)
    end
end

# The real publisher validates GitHub refs. This test config instead exercises
# Documenter's archive management against a temporary local bare repository.
struct LocalDeployConfig <: Documenter.DeployConfig
    repository::String
    version::String
end

Documenter.authentication_method(::LocalDeployConfig) = Documenter.HTTPS
Documenter.authenticated_repo_url(cfg::LocalDeployConfig) = "file://$(abspath(cfg.repository))"

function Documenter.deploy_folder(cfg::LocalDeployConfig; repo, branch, kwargs...)
    return Documenter.DeployDecision(
        all_ok=true, branch=branch, repo=repo,
        subfolder=documentation_version(cfg.version), is_preview=false,
    )
end

@testset "Local version archive integration" begin
    mktempdir() do directory
        source = source_fixture(directory)
        target = joinpath(source.docs, "build")
        repository = joinpath(directory, "archive.git")
        run(`git init --bare --quiet --initial-branch=gh-pages $repository`)

        function publish(version, text)
            write_artifact(target, version, source.sha; text)
            @test validate_documentation_build(source.docs, target, version, source.sha) === nothing
            Documenter.deploydocs(
                root=source.docs, target=target, repo="local/documentation.git",
                devbranch="dev", devurl="dev",
                versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
                deploy_config=LocalDeployConfig(repository, version),
                push_preview=false, forcepush=false,
            )
            return nothing
        end

        function snapshot(name)
            destination = joinpath(directory, name)
            run(`git clone --quiet --branch gh-pages $repository $destination`)
            return destination
        end

        publish("dev", "Development first build")
        development = snapshot("development")
        @test occursin("Development first build", read(joinpath(development, "dev", "index.html"), String))
        @test occursin("url=./dev/", read(joinpath(development, "index.html"), String))
        @test occursin("DOCUMENTER_IS_DEV_VERSION = true", read(joinpath(development, "dev", "siteinfo.js"), String))

        publish("v0.2.0", "Release 0.2.0")
        release = snapshot("release")
        release_html = read(joinpath(release, "v0.2.0", "index.html"), String)
        @test occursin("Release 0.2.0", release_html)
        @test islink(joinpath(release, "stable"))
        @test readlink(joinpath(release, "stable")) == "v0.2.0"
        @test read(joinpath(release, "stable", "index.html"), String) == release_html
        @test occursin("url=./stable/", read(joinpath(release, "index.html"), String))
        @test occursin("DOCUMENTER_CURRENT_VERSION = \"v0.2.0\"", read(joinpath(release, "v0.2.0", "siteinfo.js"), String))
        @test occursin("DOCUMENTER_IS_DEV_VERSION = false", read(joinpath(release, "v0.2.0", "siteinfo.js"), String))
        for version in ("stable", "v0.2.0", "dev")
            @test occursin(version, read(joinpath(release, "versions.js"), String))
        end

        publish("dev", "Development second build")
        updated = snapshot("updated")
        @test occursin("Development second build", read(joinpath(updated, "dev", "index.html"), String))
        @test read(joinpath(updated, "v0.2.0", "index.html"), String) == release_html
        @test read(joinpath(updated, "stable", "index.html"), String) == release_html
        @test readlink(joinpath(updated, "stable")) == "v0.2.0"

        publish("v0.3.0-rc.1", "Release candidate")
        candidate = snapshot("candidate")
        @test occursin("Release candidate", read(joinpath(candidate, "v0.3.0-rc.1", "index.html"), String))
        @test readlink(joinpath(candidate, "stable")) == "v0.2.0"
        @test read(joinpath(candidate, "stable", "index.html"), String) == release_html
        @test occursin("url=./stable/", read(joinpath(candidate, "index.html"), String))
    end
end
