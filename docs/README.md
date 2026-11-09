# Build the documentation locally

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
`dev` only run build checks. A separate deployment job publishes `docs/build/`
from a successful `main` push or a manual run on `main`.

Before the first deployment, enable GitHub Pages with **GitHub Actions** as
the source and restrict the `github-pages` environment to `main`. These
repository settings must be configured separately; the workflow does not
enable Pages. No deploy key or `gh-pages` branch is needed.

The intended site is
[HighOrderMRISim.jl documentation](https://bennyzhang-codes.github.io/HighOrderMRISim.jl/).
It serves the current `main` manual, not a versioned release archive. Keep
the README badge on the local build until the first deployment is verified.
