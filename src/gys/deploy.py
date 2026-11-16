"""Serialize an argmin Euler schedule on canonical and native policy time axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core import step_widths, validate_nodes

__all__ = ["KNOT_AXES", "knot_axis_from_metadata", "parse_wire", "schedule_document", "wire_string"]

KNOT_AXES = ("canonical", "flipped")


def _check_axis(knot_axis: str) -> str:
    if knot_axis not in KNOT_AXES:
        raise ValueError(f"unknown knot_axis {knot_axis!r}; known: {list(KNOT_AXES)}")
    return knot_axis


def knot_axis_from_metadata(metadata: dict[str, Any] | None) -> str:
    """Read the native time direction from the policy metadata."""
    axis = (metadata or {}).get("wire_knot_axis")
    if axis is None:
        raise ValueError("policy metadata must specify wire_knot_axis")
    return _check_axis(str(axis))


def wire_string(nodes: Sequence[float], *, knot_axis: str) -> str:
    """Canonical nodes -> the comma-separated token a serving config carries."""
    _check_axis(knot_axis)
    grid = validate_nodes(np.asarray(nodes, dtype=np.float64))
    values = grid if knot_axis == "canonical" else 1.0 - grid
    return ",".join(f"{float(v):.17g}" for v in values)


def parse_wire(token: str, *, knot_axis: str) -> np.ndarray:
    """Inverse of :func:`wire_string`; returns canonical nodes."""
    _check_axis(knot_axis)
    text = token.strip()
    if not text:
        raise ValueError("empty wire token")
    try:
        native = np.array([float(part) for part in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"malformed wire token {token!r}: {exc}") from exc
    nodes = native if knot_axis == "canonical" else 1.0 - native
    return validate_nodes(nodes)


def schedule_document(
    nodes: Sequence[float],
    *,
    knot_axis: str,
    fingerprint: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The deployable artifact: one schedule, stated on both axes, with its provenance."""
    grid = validate_nodes(np.asarray(nodes, dtype=np.float64))
    return {
        "schema": "gys-schedule-v1",
        "fingerprint": fingerprint,
        "steps": int(grid.size - 1),
        "time_convention": {
            "canonical": "0 = noise, 1 = action; `nodes` is stated on this axis",
            "knot_axis": knot_axis,
            "sampler": "forward Euler",
        },
        "nodes": [float(v) for v in grid],
        "native_nodes": [float(v) for v in (grid if knot_axis == "canonical" else 1.0 - grid)],
        "step_widths": [float(v) for v in step_widths(grid)],
        "wire": wire_string(grid, knot_axis=knot_axis),
        "provenance": dict(provenance),
    }
