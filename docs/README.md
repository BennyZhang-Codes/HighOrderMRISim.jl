# Build the documentation locally

Use Julia 1.12. From the repository root, set up Python and Matplotlib as
described in the [package README](../README.md#installation). Then run:

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
