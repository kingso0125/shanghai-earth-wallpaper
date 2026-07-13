from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import SHANGHAI


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    name: str = "Current location"
    updated_utc: str | None = None

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")
        if not self.name.strip() or len(self.name) > 48:
            raise ValueError("location name must contain 1 to 48 characters")


DEFAULT_LOCATION = Location(SHANGHAI[0], SHANGHAI[1], "Shanghai")


def haversine_km(first: Location, second: Location) -> float:
    lat1, lat2 = math.radians(first.latitude), math.radians(second.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second.longitude - first.longitude)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.asin(min(1.0, math.sqrt(value)))


class LocationStore:
    def __init__(self, path: Path, threshold_km: float = 80.0):
        if threshold_km <= 0:
            raise ValueError("threshold_km must be positive")
        self.path = path
        self.threshold_km = threshold_km

    def load(self) -> Location:
        if not self.path.exists():
            return DEFAULT_LOCATION
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return Location(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            name=str(payload.get("name") or "Current location"),
            updated_utc=payload.get("updated_utc"),
        )

    def update(self, candidate: Location) -> tuple[Location, float, bool]:
        current = self.load()
        distance = haversine_km(current, candidate)
        if distance <= self.threshold_km:
            return current, distance, False

        accepted = Location(
            candidate.latitude,
            candidate.longitude,
            candidate.name,
            datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(asdict(accepted), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return accepted, distance, True
