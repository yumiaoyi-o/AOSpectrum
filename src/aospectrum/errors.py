"""Small user-facing exception set."""


class AOSpectrumError(RuntimeError):
    pass


class InputError(AOSpectrumError):
    pass


class SolverError(AOSpectrumError):
    pass
