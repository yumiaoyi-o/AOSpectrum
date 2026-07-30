"""Versioned JSON input for reciprocal-space paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aospectrum.band.model import KPath
from aospectrum.errors import ContractError


SCHEMA = "aospectrum.kpath/v1"


def read_kpath(path: str | Path) -> KPath:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read k path {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ContractError(f"k path must use schema {SCHEMA}")
    labels = value.get("labels")
    if not isinstance(labels, list):
        raise ContractError("k path labels must be a list")
    if "nodes_fractional" in value:
        points_per_segment = value.get("points_per_segment")
        if (
            isinstance(points_per_segment, bool)
            or not isinstance(points_per_segment, int)
        ):
            raise ContractError("interpolated k path requires points_per_segment")
        return KPath.interpolate(
            value["nodes_fractional"],
            tuple(str(label) for label in labels),
            points_per_segment=points_per_segment,
        )
    try:
        return KPath(
            fractional_points=np.asarray(
                value["fractional_points"],
                dtype=np.float64,
            ),
            node_indices=np.asarray(value["node_indices"], dtype=np.int64),
            node_labels=tuple(str(label) for label in labels),
        )
    except KeyError as exc:
        raise ContractError(
            "k path requires nodes_fractional or explicit fractional_points"
        ) from exc

