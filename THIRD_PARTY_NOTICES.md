# Third-party notices

The copyright notices and [MIT terms](#mit-license) below apply only to the
identified incorporated third-party portions, not the package as a whole or
its datasets. The package's own terms remain in [LICENSE](LICENSE).

## KomaMRI.jl

Adapted legacy simulation, data-type, and plotting code.
[Upstream license (v0.8.0)](https://github.com/JuliaHealth/KomaMRI.jl/blob/v0.8.0/LICENSE).

## MRIReco.jl and MRISimulation

Adapted [image metrics](src/utils/ImageMetrics.jl) and
[birdcage coil sensitivities](src/phantom/csm/Birdcage.jl).
License references: [MRIReco](https://github.com/MagneticResonanceImaging/MRIReco.jl/blob/8307c24a6086d202fe8328f0727548de54ec8137/LICENSE.md)
and [MRISimulation](https://github.com/MagneticResonanceImaging/MRIReco.jl/blob/8307c24a6086d202fe8328f0727548de54ec8137/MRISimulation/LICENSE.md)
at v0.8.0 (`8307c24`). The exact original import revision is unconfirmed.

## MRFColormaps

[T1cm.mat](src/plot/assets/T1cm.mat) and [T2cm.mat](src/plot/assets/T2cm.mat)
match [MRFColormaps at `97df59a`](https://github.com/markgriswold/MRFColormaps/tree/97df59a499ef41fdb6791bc16a70cf2319afb69f).
[Upstream license](https://github.com/markgriswold/MRFColormaps/blob/97df59a499ef41fdb6791bc16a70cf2319afb69f/LICENSE).

## DataStitching

See the [project and paper attribution](README.md#citation). Its terms remain
in the package [LICENSE](LICENSE); pending data sources and permissions are
listed in [Data sources](DATA_SOURCES.md).

## MIT license

```text
KomaMRI.jl
Copyright (c) 2020 Carlos Castillo Passi

MRIReco.jl
Copyright (c) 2018: Tobias Knopp.

MRISimulation
Copyright (c) 2022: Tobias Knopp and Contributors

MRFColormaps
Copyright (c) 2019 markgriswold

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
