from dataclasses import dataclass

SHANGHAI = (31.2304, 121.4737)


@dataclass(frozen=True)
class RenderPreset:
    name: str
    size: tuple[int, int]
    center_px: tuple[float, float]
    globe_radius_px: float
    target_lat: float = SHANGHAI[0]
    target_lon: float = SHANGHAI[1]


LOCK = RenderPreset("lock", (1320, 2868), (660.0, 1470.0), 620.0, 0.0, SHANGHAI[1])
HOME = RenderPreset("home", (1320, 2868), (660.0, 2440.0), 1500.0, 0.0, SHANGHAI[1])
PRESETS = (LOCK, HOME)

GIBS_ENDPOINT = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_CAPABILITIES = (
    GIBS_ENDPOINT + "?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"
)
VISIBLE_LAYER = "Himawari_AHI_Band3_Red_Visible_1km"
IR_LAYER = "Himawari_AHI_Band13_Clean_Infrared"
BASE_LAYER = "BlueMarble_NextGeneration"
LIGHTS_LAYER = "VIIRS_CityLights_2012"
