"""Public exception hierarchy."""


class AOSpectrumError(RuntimeError):
    """Base error raised by AOSpectrum."""


class ContractError(AOSpectrumError):
    """A public data contract is invalid."""


class BundleIOError(AOSpectrumError):
    """An operator bundle cannot be read or written."""


class AssemblyError(AOSpectrumError):
    """Sparse Bloch assembly failed."""


class SolverError(AOSpectrumError):
    """A numerical backend failed to satisfy its contract."""


class BackendUnavailableError(SolverError):
    """A selected numerical backend is not installed in this environment."""


class FieldError(AOSpectrumError):
    """A real-space AO field cannot be evaluated."""


class ArtifactError(AOSpectrumError):
    """A result artifact cannot be produced or read."""
