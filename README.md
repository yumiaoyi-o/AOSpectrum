# AOSpectrum

AOSpectrum is a sparse Band and real-space Orbital toolkit for periodic
localized atomic-orbital models. It consumes translation-indexed Hamiltonian
and overlap blocks from any producer that writes the
`aospectrum.operator-bundle/v1` contract.

The package has two peer applications:

- **Band** returns every eigenvalue in a requested energy interval along a
  k path.
- **Orbital** resolves an absolute or VBM/CBM-relative state range at Gamma,
  reconstructs the selected wavefunctions from OpenMX pseudo-atomic orbitals,
  and writes one multi-state NGL viewer.

Both applications keep `H(k)` and `S(k)` sparse. The production package does
not expose a dense eigensolver path.

## Calculation Flow

```text
localized H(R), S(R), structure, AO layout
                        |
               operator bundle
                  /           \
       sparse H(k), S(k)     sparse H(Gamma), S(Gamma)
              |                        |
       CISS + cuDSS          cuDSS inertia + PRIMME
              |                        |
     Band arrays and plot       selected coefficients
                                         |
                               OpenMX PAO reconstruction
                                         |
                                  NGL index.html
```

The production numerical paths are fixed:

| Application | Solver | Scalar path | Parallelism |
| --- | --- | --- | --- |
| Band | CISS with cuDSS shifted solves | float32/complex64 | one worker per listed GPU, k points sharded across workers |
| Orbital | cuDSS inertia with directed PRIMME solves | real float32 at Gamma, complex64 coefficients | one GPU |

There is no hidden dense or CPU fallback. Every result records generalized
eigenproblem residuals, stage timings, available GPU-memory evidence, and the
effective solver settings.

## Installation

Install the Python package and Orbital field dependency:

```bash
python -m pip install ".[orbital]"
```

Production solving also requires CUDA, cuDSS, and a GPU-enabled PRIMME build.
The repository includes a single build entry:

```bash
export CUDSS_ROOT=/path/to/cudss
export PRIMME_ROOT=/path/to/primme
PYTHON=python scripts/build-gpu.sh
python -m pip install --force-reinstall dist/aospectrum-*.whl
```

`AOSPECTRUM_BUILD_DIR`, `AOSPECTRUM_WHEEL_DIR`, and
`AOSPECTRUM_CMAKE_ARGS` can be set when a cluster needs custom build paths or
CMake options.

GitHub releases provide the target-neutral source archive. Build the native
wheel on the CUDA stack where it will run so that cuDSS, PRIMME, BLAS, and
LAPACK are linked against that site.

## Input Contract

The canonical input is an `aospectrum.operator-bundle/v1` directory:

```text
system.aobundle/
  manifest.json
  provenance.json
  filling.json
  structure/{cell,positions,atomic_numbers}.npy
  basis/layout.json
  basis/{orbital_atom,local_index}.npy
  operators/{row_atom,col_atom,lattice_shift}.npy
  operators/{block_offsets,block_shapes,h_values,s_values}.npy
```

Orbital reconstruction additionally needs the angular-momentum, radial-index,
and real-harmonic-index arrays in `basis/`. Electronic filling is required for
VBM/CBM-relative state selection or a bundle-derived energy reference.
Hamiltonian values and eigenvalues use eV; cell vectors and positions use
Angstrom; overlap values are dimensionless.

For a block from reference-cell atom `i` to translated atom `j + R`,
AOSpectrum assembles

```text
H_ij(k) += H_i0,jR exp(+2 pi i R dot k)
S_ij(k) += S_i0,jR exp(+2 pi i R dot k)
```

where `k` is fractional reciprocal coordinate. The producer supplies the
Hermitian mirror records and one shared global AO order. AOSpectrum solves
`Hc = ESc`; the supplied overlap is never replaced by the identity.

Use `aospectrum inspect system.aobundle` to validate and summarize a bundle.
The public `write_bundle` function is the supported producer entry point.
`examples/band/` contains a 121-point Y-G-X-M path and Band configuration;
`examples/orbital/` contains the Orbital and OpenMX PAO mapping templates.
All paths in a TOML configuration are resolved relative to that file.

## Quick Start

Prepare an operator bundle and copy the relevant example configuration beside
its input files before running these commands. The repository examples are
templates; they do not bundle a material Hamiltonian or OpenMX PAO dataset.

Inspect an operator bundle:

```bash
aospectrum inspect system.aobundle
```

Run a Band calculation:

```bash
aospectrum band --config band.toml
```

Run selected Gamma-point orbitals and serve the result:

```bash
aospectrum orbital --config orbital.toml
aospectrum view orbital-result
```

An interrupted calculation is continued only when requested explicitly:

```bash
aospectrum band --config band.toml --resume
aospectrum orbital --config orbital.toml --resume
```

Band resumes completed k points. Orbital resumes the selected eigensystem and
completed per-state real-space fields. A successful run removes its checkpoint
directory; an interrupted or failed run keeps it beside the requested output.

Compare a Band result with a same-path reference:

```bash
aospectrum compare-band \
  --new band-result \
  --reference reference-band-result \
  --output band-comparison
```

The comparison reports pointwise absolute energy differences in eV. Its
logarithmic red-white-blue plot uses `1e-5 eV` as the default white midpoint.

## Version Scope

The `1.0.0rc1` line targets periodic full-k-path Band calculations and
single-frame, Gamma-point scalar Orbital visualization. Near-degenerate
selection boundaries and nonperiodic structure metadata are reported as
warnings because individual orbital images may depend on conventions outside
this validated scope.

SOC-aware spinor orbitals and time-dependent orbital views are the next
development goals. Their data flow extends Orbital semantics without merging
the Band and Orbital applications.

## Related Projects

- [NVIDIA cuDSS](https://developer.nvidia.com/cudss)
- [PRIMME](https://github.com/primme/primme)
- [OpenMX](https://www.openmx-square.org/)
- [NGL Viewer](https://github.com/nglviewer/ngl)

## License

Apache-2.0. Bundled third-party notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
