# AOSpectrum

AOSpectrum calculates sparse band structures and selected real-space orbitals
from localized atomic-orbital Hamiltonian and overlap blocks. The input
producer can be HamGNN, OpenMX, or any program that writes the
`aospectrum.operator-bundle/v1` directory contract.

## What It Does

```text
localized H(R), S(R), structure, AO layout
                        |
               operator bundle
                  /           \
       sparse H(k), S(k)     sparse H(Gamma), S(Gamma)
              |                        |
       CISS + cuDSS          cuDSS inertia + PRIMME
              |                        |
      bands.csv + PNG          selected coefficients
                                         |
                               OpenMX PAO expansion
                                         |
                              multi-state NGL viewer
```

| Application | Numerical path | Parallelism |
| --- | --- | --- |
| Band | complex64 CISS with cuDSS shifted solves | k points distributed over the GPUs listed in the TOML |
| Orbital | float32 Gamma-point inertia and directed PRIMME solves | one GPU |
| Orbital field | complex64 OpenMX PAO expansion | one GPU, chunked grid points |

`H(k)` and `S(k)` remain sparse. Only the small CISS projected subspace is
dense. AOSpectrum has no dense full-spectrum or alternate production backend.

## Build

The GPU extension requires CUDA, cuDSS, and a CUDA-enabled PRIMME build:

```bash
export CUDSS_ROOT=/path/to/cudss
export PRIMME_ROOT=/path/to/primme
PYTHON=python scripts/build-gpu.sh
python -m pip install torch
python -m pip install --force-reinstall dist/aospectrum-*.whl
```

Set `AOSPECTRUM_BUILD_DIR`, `AOSPECTRUM_WHEEL_DIR`, or
`AOSPECTRUM_CMAKE_ARGS` when a cluster needs custom build locations or CMake
options. Build the wheel on the CUDA stack where it will run.

## Input Bundle

```text
system.aobundle/
  manifest.json
  provenance.json
  filling.json
  structure/{cell,positions,atomic_numbers}.npy
  basis/layout.json
  basis/{orbital_atom,local_index}.npy
  basis/{angular_momentum,radial_index,real_harmonic_index}.npy
  operators/{row_atom,col_atom,lattice_shift}.npy
  operators/{block_offsets,block_shapes,h_values,s_values}.npy
```

The three basis descriptor arrays are needed only for Orbital. `filling.json`
is needed for `VBM`/`CBM` state expressions or `bundle_fermi` energy
referencing. H and eigenvalues use eV, positions and cell vectors use
Angstrom, and S is dimensionless.

For a block from reference-cell atom `i` to translated atom `j + R`,

```text
H_ij(k) += H_i0,jR exp(+2 pi i R dot k)
S_ij(k) += S_i0,jR exp(+2 pi i R dot k)
```

The supplied overlap is used in `Hc = ESc`; it is never replaced by the
identity. Python producers can create the directory with
`aospectrum.write_bundle`.

## Band

Start from [examples/band/band.toml](examples/band/band.toml) and the
121-point Y-G-X-M path beside it:

```bash
aospectrum band band.toml
```

The energy interval is relative to `input_zero`, a fixed reference, or the
bundle Fermi energy. Optional CISS controls are:

```toml
[solver]
integration_points = 32
contour_batch_size = 1
block_size = 32
moment_size = 4
```

`block_size * moment_size` is the maximum search-subspace dimension. Increase
it when the requested energy interval contains too many states. The output
contains:

```text
band-result/
  band.png
  bands.csv
  summary.json
```

`summary.json` records the maximum generalized-eigenpair residual in eV,
stage timings, peak GPU-memory evidence reported by cuDSS, and warnings.

## Orbital

Map each atomic number to its OpenMX `.pao` file as shown in
[examples/orbital/openmx-basis.toml](examples/orbital/openmx-basis.toml), then
run:

```bash
aospectrum orbital orbital.toml
aospectrum view orbital-result
```

State expressions can be absolute or relative to the band edges:

```toml
states = "absolute:5285:5301"
states = "VBM-5:CBM+10"
```

Orbital is currently a single-frame Gamma-point application. The grid is set
by either `grid_shape = [nx, ny, nz]` or
`grid_spacing_angstrom = spacing`. The result is one integrated NGL viewer
with a state selector, phase/density switch, enclosed-probability control,
atoms, and the periodic cell:

```text
orbital-result/
  index.html
  viewer-manifest.json
  assets/
  states/
```

Use `aospectrum view` rather than opening `index.html` directly because the
browser loads the local volume files through HTTP.

## Resume

An interrupted run leaves one lightweight work directory beside its requested
output. Continue it explicitly:

```bash
aospectrum band band.toml --resume
aospectrum orbital orbital.toml --resume
```

Band resumes completed k points. Orbital resumes the selected eigensystem and
completed state fields. A successful run removes the work directory.

## Current Scope

The current release targets periodic Band calculations and scalar,
Gamma-point Orbital visualization. Nonperiodic metadata and near-degenerate
state boundaries are reported in result warnings because individual orbital
interpretation may be inaccurate. Complex spinor H/S lies outside the scalar
Orbital input scope. SOC-aware orbitals and time-dependent orbital views are
planned extensions.

## Related Projects

- [NVIDIA cuDSS](https://developer.nvidia.com/cudss)
- [PRIMME](https://github.com/primme/primme)
- [OpenMX](https://www.openmx-square.org/)
- [NGL Viewer](https://github.com/nglviewer/ngl)

## License

Apache-2.0. Bundled third-party notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
