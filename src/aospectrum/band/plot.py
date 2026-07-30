"""Publication-style line rendering for ragged energy-window bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from aospectrum.band.model import BandResult
from aospectrum.band.tracking import tracked_line_segments


def plot_band_result(
    result: BandResult,
    destination: str | Path,
    *,
    title: str | None = None,
    dpi: int = 240,
) -> Path:
    """Render connected band lines without changing the numerical result."""

    path = Path(destination)
    distances = result.kpath.distances(
        np.asarray(result.metadata["cell_angstrom"], dtype=np.float64)
        if "cell_angstrom" in result.metadata
        else np.eye(3, dtype=np.float64)
    )
    segments = tracked_line_segments(distances, result)
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    if segments.size:
        axis.add_collection(
            LineCollection(
                segments,
                colors="#17213a",
                linewidths=1.05,
                antialiased=True,
            )
        )
    axis.set_xlim(float(distances[0]), float(distances[-1]))
    axis.set_ylim(
        result.display_interval_ev.lower_ev,
        result.display_interval_ev.upper_ev,
    )
    node_x = distances[result.kpath.node_indices]
    for position in node_x[1:-1]:
        axis.axvline(position, color="#9aa1ad", linewidth=0.7, zorder=0)
    axis.axhline(0.0, color="#b73b3b", linewidth=0.8, linestyle="--")
    axis.set_xticks(node_x, result.kpath.node_labels)
    axis.set_xlabel("K path")
    axis.set_ylabel("Energy relative to reference (eV)")
    if title:
        axis.set_title(title)
    axis.tick_params(direction="in")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path
