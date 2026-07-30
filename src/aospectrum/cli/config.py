"""TOML and path helpers shared by command-line applications."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from aospectrum.errors import ContractError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class ConfigDocument:
    path: Path
    values: Mapping[str, Any]

    @property
    def base_directory(self) -> Path:
        return self.path.parent

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ContractError(f"configuration requires [{name}]")
        return value

    def require_sections(self, allowed: set[str]) -> None:
        reject_unknown_keys(self.values, allowed, context="configuration")

    def resolve_path(self, value: Any, *, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{name} must be a nonempty path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.base_directory / path
        return path.resolve()


def load_config(path: str | Path) -> ConfigDocument:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("rb") as handle:
            values = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"cannot read TOML configuration {source}: {exc}") from exc
    if not isinstance(values, dict):
        raise ContractError("TOML configuration root must be a table")
    return ConfigDocument(source, values)


def required_string(
    section: Mapping[str, Any],
    name: str,
    *,
    context: str,
) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context}.{name} must be a nonempty string")
    return value.strip()


def required_number_pair(
    section: Mapping[str, Any],
    name: str,
    *,
    context: str,
) -> tuple[float, float]:
    value = section.get(name)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) for item in value)
    ):
        raise ContractError(f"{context}.{name} must contain two numbers")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{context}.{name} must contain two numbers"
        ) from exc


def reject_unknown_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        rendered = ", ".join(unknown)
        raise ContractError(f"{context} contains unknown field(s): {rendered}")


def cuda_visible_token(
    device: str,
    inherited: str | None = None,
) -> str:
    """Resolve one logical CUDA slot against an inherited allocation."""

    if not device.startswith("cuda:"):
        raise ContractError(f"cannot bind non-CUDA device {device!r}")
    try:
        logical_index = int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ContractError(f"invalid CUDA device {device!r}") from exc
    inherited = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited is None
        else inherited
    )
    if inherited:
        visible = tuple(
            token.strip()
            for token in inherited.split(",")
            if token.strip()
        )
        if logical_index >= len(visible):
            raise ContractError(
                f"{device} exceeds the inherited CUDA allocation "
                f"of {len(visible)} device(s)"
            )
        return visible[logical_index]
    return str(logical_index)


def bind_cuda_visible_device(device: str) -> None:
    """Bind one process to a logical slot in its inherited allocation."""

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_token(device)
