# Data sources

This inventory covers the ten `.mat` and `.seq` files tracked with
HighOrderMRISim.jl v0.2.0 (`5a073ed`). It separates verified file relationships
and documented uses from provenance and permissions that still need confirmation.
It is not a data license or a statement that all files are cleared for public
redistribution.

## Reference snapshots

The file comparison uses Git blob identities against these public snapshots:

- [DataStitching at `56a40df`](https://github.com/BennyZhang-Codes/DataStitching/tree/56a40dfea20e03cf114cfcfadbbfd3af0be41910).
- [MRFColormaps at `97df59a`](https://github.com/markgriswold/MRFColormaps/tree/97df59a499ef41fdb6791bc16a70cf2319afb69f).

The 2D sequence and both colormap files are byte-identical to files in that
DataStitching snapshot. The other seven files have no identical blob in that
snapshot. This does not establish that their underlying data are new: they may
be differently processed or exported versions of previously published data.
Public availability alone does not establish redistribution permission.

## 2D demo

The consumer is [demo2D/Sim2D.jl](demo2D/Sim2D.jl).

| File | Verified relationship or documented use | Pending confirmation |
| --- | --- | --- |
| [7T_1p0_200_r4.seq](demo2D/7T_1p0_200_r4.seq) | Byte-identical to [DataStitching's multi-channel simulation sequence](https://github.com/BennyZhang-Codes/DataStitching/blob/56a40dfea20e03cf114cfcfadbbfd3af0be41910/demo/Sim_MultiChannel/7T_1p0_200_r4.seq). Used as the nominal 2D spiral sequence. | Sequence-generation source and permission to redistribute it with this package. |
| [7T_1p0_200_r4.mat](demo2D/7T_1p0_200_r4.mat) | The demo identifies `ksphaMeasured` as dynamic field coefficients from the DataStitching method and reads sampling and synchronization parameters. Not byte-identical to the public file with the same name. | Original acquisition and owner; relationship to the public file; processing/export steps; redistribution permission and privacy review. |

## 3D demo

The three files share the basename
`Spiral_2p0x2p0x2p0_100x100x35_1x4_int1_9p5ms_tr500_te5p0`.
The consumer is [demo3D/1_load_seq.jl](demo3D/1_load_seq.jl).

| File | Verified relationship or documented use | Pending confirmation |
| --- | --- | --- |
| [.seq](demo3D/Spiral_2p0x2p0x2p0_100x100x35_1x4_int1_9p5ms_tr500_te5p0.seq) | Pulseq sequence. Its header identifies the MATLAB mr toolbox; definitions name Jinyuan Zhang as developer and `vdspiral` as the spiral design tool. | Generation script and version; provenance of any incorporated waveforms; redistribution permission. Tool names do not establish a license for this file. |
| [.mat](demo3D/Spiral_2p0x2p0x2p0_100x100x35_1x4_int1_9p5ms_tr500_te5p0.mat) | Used for ADC timing and sample counts from the `skope` parameter structure. | Original export and owner; complete contents and processing history; redistribution permission and privacy review. |
| [_skope.mat](demo3D/Spiral_2p0x2p0x2p0_100x100x35_1x4_int1_9p5ms_tr500_te5p0_skope.mat) | The demo describes measured trajectories acquired with Skope and reads `g_skope` and `t_g`. | Acquisition provenance and owner; processing/export steps; redistribution permission and privacy review. The measurement system name is not a copyright attribution. |

## Phantom and receive/field maps

| File | Verified relationship or documented use | Pending confirmation |
| --- | --- | --- |
| [brain3D_0.5_0.5_0.5.mat](src/phantom/brain3D_0.5_0.5_0.5.mat) | [load_phantom_mat.jl](src/phantom/load_phantom_mat.jl) reads `data` as tissue labels on a 0.5 mm source grid. [SpinProperty.jl](src/phantom/SpinProperty.jl) assigns simulation properties to the labels. | Original dataset, creator and version; segmentation/resampling history; redistribution terms; whether any human-derived anatomy requires a privacy review. No particular external phantom collection has been verified as the source. |
| [b0_4p0.mat](src/phantom/b0map/b0_4p0.mat) | [b0_real.jl](src/phantom/b0map/b0_real.jl) loads `b0` on a 4 mm source grid. The 2D demo describes the B0 and coil maps as derived from a real GRE acquisition. | Acquisition and owner; field-map estimation and resampling steps; redistribution permission and privacy review. |
| [csm_4p0.mat](src/phantom/csm/csm_4p0.mat) | [Real_32cha.jl](src/phantom/csm/Real_32cha.jl) loads `csm` on a 4 mm source grid for the real 32-channel receive model. The 2D demo attributes the maps to a real GRE acquisition. | Acquisition and owner; coil-map estimation and resampling steps; redistribution permission and privacy review. |

The sampling conventions above describe how this package interprets the files;
they do not establish the original acquisition geometry or preprocessing history.

## Colormaps

| File | Verified source | License status |
| --- | --- | --- |
| [T1cm.mat](src/plot/assets/T1cm.mat) | Byte-identical to [MRFColormaps/T1cm.mat](https://github.com/markgriswold/MRFColormaps/blob/97df59a499ef41fdb6791bc16a70cf2319afb69f/T1cm.mat) and DataStitching's `src/plot/assets/T1cm.mat`. | MIT; original notice preserved in [Third-party notices](THIRD_PARTY_NOTICES.md#mrfcolormaps). |
| [T2cm.mat](src/plot/assets/T2cm.mat) | Byte-identical to [MRFColormaps/T2cm.mat](https://github.com/markgriswold/MRFColormaps/blob/97df59a499ef41fdb6791bc16a70cf2319afb69f/T2cm.mat) and DataStitching's `src/plot/assets/T2cm.mat`. | MIT; original notice preserved in [Third-party notices](THIRD_PARTY_NOTICES.md#mrfcolormaps). |

These are visualization lookup tables, not phantom or acquisition data.

## Before public redistribution

For each pending entry, record the original source and version, creator or
data custodian, transformations used to produce the bundled file, and the
applicable redistribution terms or approval. Confirm that institutional
approvals cover this standalone package, not only the earlier DataStitching
repository. See the existing [LICENSE](LICENSE); no new data license is assigned
by this inventory.

This review used file identities, sequence text, and the consuming code. It did
not complete an inspection of all MAT contents or an assessment of potentially
identifiable anatomy. Those checks remain pending for acquisition-derived data
and the phantom. Unresolved entries must not be described as cleared for public
redistribution merely because they are bundled here.
