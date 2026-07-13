from __future__ import annotations

from datetime import UTC, datetime

import numpy as np


def _smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sun_vector(timestamp: datetime) -> np.ndarray:
    timestamp = timestamp.astimezone(UTC)
    day = timestamp.timetuple().tm_yday
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    gamma = 2.0 * np.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    equation = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    subsolar_lon = np.deg2rad((720.0 - 4.0 * 0.0 - equation) / 4.0 - hour * 15.0)
    subsolar_lon = (subsolar_lon + np.pi) % (2 * np.pi) - np.pi
    return np.array(
        [
            np.cos(declination) * np.cos(subsolar_lon),
            np.cos(declination) * np.sin(subsolar_lon),
            np.sin(declination),
        ],
        dtype=np.float32,
    )


def daylight(surface_vectors: np.ndarray, timestamp: datetime) -> np.ndarray:
    dot = np.sum(surface_vectors * sun_vector(timestamp), axis=-1)
    return _smoothstep(-0.16, 0.12, dot).astype(np.float32)

