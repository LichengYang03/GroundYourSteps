"""Ten-tap fractional differences over the executed arm-joint horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HorizonMap", "binomial_taps", "frac_diff", "resolve_columns", "select_columns"]


def binomial_taps(v: float, terms: int) -> np.ndarray:
    """Compute (-1)^k binom(v, k), k=0,...,terms-1, by recurrence."""
    if terms < 1:
        raise ValueError(f"need at least one tap, got {terms}")
    if v < -1.0:
        raise ValueError(f"order {v} is below the supported calibration interval [-1, 2]")
    taps = np.empty(int(terms), dtype=np.float64)
    coefficient = 1.0
    for k in range(int(terms)):
        taps[k] = ((-1.0) ** k) * coefficient
        coefficient *= (v - k) / (k + 1.0)
    return taps


def frac_diff(array: np.ndarray, v: float, terms: int) -> np.ndarray:
    """Apply fractional differences along axis -2.

    Input shape: (..., H, d). Output shape: (..., H - terms + 1, d).
    Only positions with a complete filter window are retained.
    """
    array = np.asarray(array, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"array must be (..., H, d), got shape {array.shape}")
    horizon = array.shape[-2]
    if horizon < terms:
        raise ValueError(f"a {terms}-tap filter needs a horizon of at least {terms}, got {horizon}")
    taps = binomial_taps(v, terms)
    width = horizon - terms + 1
    out = np.zeros((*array.shape[:-2], width, array.shape[-1]), dtype=np.float64)
    for k, tap in enumerate(taps):
        if tap == 0.0:
            continue
        start = terms - 1 - k
        out += tap * array[..., start : start + width, :]
    return out


def select_columns(chunks: np.ndarray, columns: np.ndarray | None) -> np.ndarray:
    """Keep ``columns`` of ``(..., H, D)``; float64.  ``None`` keeps everything."""
    array = np.asarray(chunks, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"expected at least (H, D), got shape {array.shape}")
    if columns is None:
        return array
    index = np.asarray(columns, dtype=np.int64)
    if index.ndim != 1 or index.size == 0:
        raise ValueError("columns must be a non-empty 1-D index array")
    if index.min() < 0 or index.max() >= array.shape[-1]:
        raise ValueError(
            f"columns {index.min()}..{index.max()} out of range for width {array.shape[-1]}"
        )
    if np.unique(index).size != index.size:
        raise ValueError("column indices must be unique")
    return array[..., index]


def resolve_columns(
    score_slice: tuple[int, int] | None,
    exclude: tuple[int, ...],
    width: int | None = None,
) -> np.ndarray | None:
    """Select robot coordinates and omit grippers.

    Exclusion indices are relative to the robot action slice, not the padded latent.
    """
    if score_slice is None:
        if not exclude:
            return None
        if width is None:
            raise ValueError("excluding columns without a score_slice needs the latent width")
        base = np.arange(int(width), dtype=np.int64)
    else:
        lo, hi = int(score_slice[0]), int(score_slice[1])
        if hi <= lo:
            raise ValueError(f"score_slice must be a non-empty [start, stop), got {score_slice}")
        base = np.arange(lo, hi, dtype=np.int64)
    if not exclude:
        return base
    drop = np.asarray(sorted(int(c) for c in exclude), dtype=np.int64)
    if drop.min() < 0 or drop.max() >= base.size:
        raise ValueError(
            f"excluded columns {drop.tolist()} are indexed within the {base.size} scored "
            f"columns, so they must lie in [0, {base.size})"
        )
    keep = np.setdiff1d(np.arange(base.size, dtype=np.int64), drop)
    if keep.size == 0:
        raise ValueError("the gripper switch would remove every scored column")
    return base[keep]


@dataclass(frozen=True)
class HorizonMap:
    """Select the executed action coordinates and apply fractional differencing."""

    terms: int
    columns: np.ndarray | None = None
    horizon: int = 0

    def trim(self, array: np.ndarray) -> np.ndarray:
        """Select the scored action horizon and coordinates."""
        array = np.asarray(array, dtype=np.float64)
        if self.horizon:
            if array.shape[-2] < self.horizon:
                raise ValueError(
                    f"horizon={self.horizon} exceeds the chunk length {array.shape[-2]}"
                )
            array = array[..., : self.horizon, :]
        return select_columns(array, self.columns)

    def filter(self, trimmed: np.ndarray, v: float) -> np.ndarray:
        """``Delta^v`` along the horizon of an already-trimmed array."""
        return frac_diff(trimmed, float(v), int(self.terms))

    def apply(self, array: np.ndarray, v: float) -> np.ndarray:
        """Transform the selected action coordinates."""
        return self.filter(self.trim(array), v)

    @property
    def n_columns(self) -> int | None:
        return None if self.columns is None else int(np.size(self.columns))
