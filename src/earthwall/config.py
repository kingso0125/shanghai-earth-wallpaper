from dataclasses import dataclass, replace

SHANGHAI = (31.2304, 121.4737)
LOCK_LATITUDE_OFFSET = -6.0


@dataclass(frozen=True)
class RenderPreset:
    name: str
    size: tuple[int, int]
    center_px: tuple[float, float]
    globe_radius_px: float
    target_lat: float = SHANGHAI[0]
    target_lon: float = SHANGHAI[1]


LOCK = RenderPreset(
    "lock",
    (1320, 2868),
    (660.0, 1470.0),
    620.0,
    SHANGHAI[0] + LOCK_LATITUDE_OFFSET,
    SHANGHAI[1],
)
HOME = RenderPreset(
    "home", (1320, 2868), (660.0, 2440.0), 1500.0, 0.0, SHANGHAI[1]
)
PRESETS = (LOCK, HOME)


def presets_for_location(latitude: float, longitude: float) -> tuple[RenderPreset, ...]:
    """Center Lock on location; keep Home on the clearer equatorial camera angle."""
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be between -90 and 90")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be between -180 and 180")
    return tuple(
        replace(
            preset,
            target_lat=(
                max(-90.0, min(90.0, latitude + LOCK_LATITUDE_OFFSET))
                if preset.name == "lock"
                else 0.0
            ),
            target_lon=longitude,
        )
        for preset in PRESETS
    )

GIBS_ENDPOINT = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_CAPABILITIES = (
    GIBS_ENDPOINT + "?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"
)
VISIBLE_LAYER = "Himawari_AHI_Band3_Red_Visible_1km"
IR_LAYER = "Himawari_AHI_Band13_Clean_Infrared"
BASE_LAYER = "BlueMarble_NextGeneration"
LIGHTS_LAYER = "VIIRS_CityLights_2012"
