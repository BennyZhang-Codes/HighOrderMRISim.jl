# Documentation

The [documentation site](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/)
provides a version selector:

| Version | Source |
| --- | --- |
| [stable](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/stable/) | Highest non-prerelease version with published documentation |
| [dev](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/dev/) | Committed `dev` branch |
| [v0.2.0](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/v0.2.0/) | Archived manual built from the `v0.2.0` tag |

Release manuals use their tagged package and Markdown sources. Later changes
to the development manual do not change those release sources.

## Build locally

Use Julia 1.12. From the repository root, set up Python and Matplotlib as
described in [Getting Started](src/getting-started.md#install-from-a-checkout).
Then run:

```bash
export JULIA_LOAD_PATH='@:@stdlib'
export JULIA_DEPOT_PATH="$PWD/local/docs-depot:"
export CUDA_VISIBLE_DEVICES=''

julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=docs docs/make.jl
```

The documentation project loads HighOrderMRISim from the current checkout. It
does not install anything into the default Julia environment. You may reuse an
existing isolated depot instead of `local/docs-depot/` to avoid recompiling
dependencies.

Open `docs/build/index.html` in a browser. The build runs the CPU example and
checks documentation references without requiring a GPU. The generated site and
local documentation manifest are ignored by Git. The build does not deploy.

## Documentation CI and publishing

The [documentation workflow](../.github/workflows/documentation.yml) builds
the manual on pushes and pull requests to `main` and `dev`. Pull requests and
`main` pushes only run build checks. Pushes to `dev` update the development
manual; version tags (`v*`) publish versioned manuals. Only non-prerelease
versions can become `stable`.

To publish an existing release, run the workflow manually from `main` or `dev`
and set `release_tag`, for example `v0.2.0`. The workflow checks out that tag's
package and manual, using the current publishing configuration without moving
or changing the tag.

Documenter maintains the generated version archive on `gh-pages`. The workflow
then deploys the complete site as a GitHub Pages artifact. Keep Pages configured
with **GitHub Actions** as the source and allow the `github-pages` environment
to deploy from `dev`, `main`, and version tags. Pull requests never publish.
Publishing uses `GITHUB_TOKEN`; no SSH deploy key is needed.

The README badge reports the `dev` documentation workflow status. The separate
stable and development links open the published manuals.
