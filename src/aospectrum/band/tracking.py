"""Deterministic energy-continuity tracks for ragged Band windows."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from aospectrum.band.model import BandResult


def assign_track_ids(result: BandResult) -> tuple[np.ndarray, ...]:
    """Match adjacent k points by minimum total energy displacement.

    Band is an energy-window application and therefore has no absolute state
    numbers. Track identifiers only encode continuity within this result.
    """

    energies = result.display_energies_ev
    first = np.arange(energies[0].size, dtype=np.int64)
    tracks = [first]
    next_identifier = int(first.size)
    for index in range(1, len(energies)):
        previous = energies[index - 1]
        current = energies[index]
        previous_ids = tracks[index - 1]
        current_ids = np.full(current.size, -1, dtype=np.int64)
        if previous.size and current.size:
            cost = np.abs(previous[:, None] - current[None, :])
            previous_rows, current_rows = linear_sum_assignment(cost)
            current_ids[current_rows] = previous_ids[previous_rows]
        unmatched = np.flatnonzero(current_ids < 0)
        current_ids[unmatched] = np.arange(
            next_identifier,
            next_identifier + unmatched.size,
            dtype=np.int64,
        )
        next_identifier += int(unmatched.size)
        current_ids.setflags(write=False)
        tracks.append(current_ids)
    return tuple(tracks)


def tracked_line_segments(
    x: np.ndarray,
    result: BandResult,
) -> np.ndarray:
    """Return adjacent line segments joined by continuity track."""

    energies = result.display_energies_ev
    identifiers = assign_track_ids(result)
    segments: list[np.ndarray] = []
    for index in range(x.size - 1):
        right_by_track = {
            int(track): local
            for local, track in enumerate(identifiers[index + 1])
        }
        for left_local, track in enumerate(identifiers[index]):
            right_local = right_by_track.get(int(track))
            if right_local is None:
                continue
            segments.append(
                np.asarray(
                    [
                        [x[index], energies[index][left_local]],
                        [x[index + 1], energies[index + 1][right_local]],
                    ],
                    dtype=np.float64,
                )
            )
    if not segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack(segments)
