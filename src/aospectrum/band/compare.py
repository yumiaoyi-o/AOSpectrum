"""Same-path Band energy comparison and residual rendering."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from aospectrum.band.model import BandResult
from aospectrum.band.tracking import assign_track_ids
from aospectrum.errors import ArtifactError, ContractError


@dataclass(frozen=True, slots=True, eq=False)
class BandComparison:
    residuals_ev: tuple[np.ndarray, ...]

    @property
    def flattened_ev(self) -> np.ndarray:
        if not self.residuals_ev:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self.residuals_ev)


class MidpointLogNorm(Normalize):
    """Log-like nonnegative map with one explicit white midpoint."""

    def __init__(
        self,
        *,
        midpoint: float = 1.0e-5,
        vmin: float = 1.0e-8,
        vmax: float = 1.0e-2,
    ) -> None:
        if not 0.0 < vmin < midpoint < vmax:
            raise ContractError("residual color scale requires vmin < midpoint < vmax")
        super().__init__(vmin=vmin, vmax=vmax, clip=True)
        self.midpoint = float(midpoint)

    def __call__(self, value, clip=None):
        values = np.ma.asarray(value)
        clipped = np.clip(values, self.vmin, self.vmax)
        logarithm = np.log10(clipped)
        lower = np.log10(self.vmin)
        middle = np.log10(self.midpoint)
        upper = np.log10(self.vmax)
        mapped = np.where(
            logarithm <= middle,
            0.5 * (logarithm - lower) / (middle - lower),
            0.5 + 0.5 * (logarithm - middle) / (upper - middle),
        )
        return np.ma.array(mapped, mask=np.ma.getmask(values))


def compare_band_results(
    new: BandResult,
    reference: BandResult,
) -> BandComparison:
    if (
        not np.array_equal(
            new.kpath.fractional_points,
            reference.kpath.fractional_points,
        )
        or not np.array_equal(
            new.kpath.node_indices,
            reference.kpath.node_indices,
        )
    ):
        raise ContractError("Band comparison requires the same k path")
    residuals: list[np.ndarray] = []
    for index, (actual, expected) in enumerate(
        zip(new.display_energies_ev, reference.display_energies_ev)
    ):
        if actual.shape != expected.shape:
            raise ContractError(
                f"Band comparison state count differs at k index {index}"
            )
        residuals.append(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    return BandComparison(tuple(residuals))


def write_band_comparison(
    comparison: BandComparison,
    new: BandResult,
    destination: str | Path,
    *,
    midpoint_ev: float = 1.0e-5,
) -> Path:
    root = Path(destination)
    if root.exists():
        if not root.is_dir():
            raise ArtifactError(
                f"Band comparison destination is not a directory: {root}"
            )
        if any(root.iterdir()):
            raise ArtifactError(
                f"Band comparison destination is not empty: {root}"
            )
    root.mkdir(parents=True, exist_ok=True)
    flattened = comparison.flattened_ev
    np.save(
        root / "absolute_energy_residual_ev.npy",
        flattened,
        allow_pickle=False,
    )
    summary = {
        "schema": "aospectrum.band-comparison/v1",
        "residual_unit": "eV",
        "matching": "same-k-point-sorted-local-order",
        "count": int(flattened.size),
        "mean_absolute_energy_residual_ev": (
            float(np.mean(flattened)) if flattened.size else 0.0
        ),
        "maximum_absolute_energy_residual_ev": (
            float(np.max(flattened)) if flattened.size else 0.0
        ),
        "color_midpoint_ev": float(midpoint_ev),
    }
    (root / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_comparison(
        comparison,
        new,
        root / "band-energy-residual.png",
        midpoint_ev=midpoint_ev,
    )
    return root


def _plot_comparison(
    comparison: BandComparison,
    result: BandResult,
    destination: Path,
    *,
    midpoint_ev: float,
) -> None:
    cell = np.asarray(result.metadata.get("cell_angstrom", np.eye(3)))
    distances = result.kpath.distances(cell)
    energies = result.display_energies_ev
    track_ids = assign_track_ids(result)
    values = comparison.flattened_ev
    positive = values[values > 0.0]
    vmin = min(
        midpoint_ev / 1_000.0,
        float(np.min(positive)) if positive.size else midpoint_ev / 1_000.0,
    )
    vmax = max(
        midpoint_ev * 1_000.0,
        float(np.max(values)) if values.size else midpoint_ev * 1_000.0,
    )
    norm = MidpointLogNorm(
        midpoint=midpoint_ev,
        vmin=max(vmin, np.finfo(np.float64).tiny),
        vmax=vmax,
    )
    colormap = LinearSegmentedColormap.from_list(
        "aospectrum-residual",
        ("#2457c5", "#ffffff", "#d62728"),
    )
    figure, axis = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    by_track: dict[int, list[tuple[float, float]]] = {}
    for x, point_energies, identifiers in zip(
        distances,
        energies,
        track_ids,
    ):
        for energy, identifier in zip(point_energies, identifiers):
            by_track.setdefault(int(identifier), []).append(
                (float(x), float(energy))
            )
    for coordinates in by_track.values():
        if len(coordinates) > 1:
            axis.plot(
                [item[0] for item in coordinates],
                [item[1] for item in coordinates],
                color="#a8adb5",
                linewidth=0.65,
                zorder=1,
            )
    scatter = axis.scatter(
        np.concatenate(
            [
                np.full(point.absolute_energies_ev.size, distances[index])
                for index, point in enumerate(result.points)
            ]
        ),
        np.concatenate(energies),
        c=values,
        cmap=colormap,
        norm=norm,
        s=8,
        linewidths=0,
        zorder=2,
    )
    nodes = distances[result.kpath.node_indices]
    for position in nodes[1:-1]:
        axis.axvline(position, color="#c5c8ce", linewidth=0.6, zorder=0)
    axis.axhline(0.0, color="#555b65", linewidth=0.7, linestyle="--")
    axis.set_xlim(float(distances[0]), float(distances[-1]))
    axis.set_ylim(
        result.display_interval_ev.lower_ev,
        result.display_interval_ev.upper_ev,
    )
    axis.set_xticks(nodes, result.kpath.node_labels)
    axis.set_xlabel("K path")
    axis.set_ylabel("Energy relative to reference (eV)")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Absolute energy residual (eV)")
    colorbar.set_ticks((norm.vmin, midpoint_ev, norm.vmax))
    colorbar.set_ticklabels(
        (f"{norm.vmin:.0e}", f"{midpoint_ev:.0e}", f"{norm.vmax:.0e}")
    )
    figure.savefig(destination, dpi=240)
    plt.close(figure)
