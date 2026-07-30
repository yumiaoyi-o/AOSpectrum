"""Parse absolute and VBM/CBM-relative Orbital selections."""

from __future__ import annotations

import re

from aospectrum.errors import ContractError
from aospectrum.model.operators import ElectronicFilling
from aospectrum.model.spectra import AbsoluteStateRange
from aospectrum.orbital.model import ResolvedStateSelection


_EDGE = re.compile(r"^(VBM|CBM)(?:([+-])(\d+))?$", re.IGNORECASE)


def _edge_state(token: str, filling: ElectronicFilling) -> int:
    match = _EDGE.fullmatch(token.strip())
    if match is None:
        raise ContractError(f"invalid band-edge state token: {token!r}")
    edge, sign, magnitude = match.groups()
    base = (
        filling.vbm_state_number
        if edge.upper() == "VBM"
        else filling.cbm_state_number
    )
    offset = int(magnitude or 0)
    if sign == "-":
        offset = -offset
    return base + offset


def resolve_state_selection(
    expression: str,
    filling: ElectronicFilling | None,
) -> ResolvedStateSelection:
    """Resolve one user expression to one-based inclusive absolute states."""

    normalized = expression.strip()
    lower = normalized.lower()
    if lower.startswith("absolute:"):
        payload = normalized.split(":", maxsplit=1)[1]
        parts = payload.split(":")
        if len(parts) not in {1, 2}:
            raise ContractError("absolute selection must be N or N:M")
        try:
            first = int(parts[0])
            last = first if len(parts) == 1 else int(parts[1])
        except ValueError as exc:
            raise ContractError("absolute state numbers must be integers") from exc
        return ResolvedStateSelection(
            expression=normalized,
            absolute=AbsoluteStateRange(first, last),
            semantics="absolute",
        )

    if filling is None:
        raise ContractError(
            "VBM/CBM-relative selection requires ElectronicFilling"
        )
    parts = normalized.split(":")
    if len(parts) not in {1, 2}:
        raise ContractError("band-edge selection must be STATE or STATE:STATE")
    first = _edge_state(parts[0], filling)
    last = first if len(parts) == 1 else _edge_state(parts[1], filling)
    return ResolvedStateSelection(
        expression=normalized,
        absolute=AbsoluteStateRange(first, last),
        semantics="band_edge",
    )
