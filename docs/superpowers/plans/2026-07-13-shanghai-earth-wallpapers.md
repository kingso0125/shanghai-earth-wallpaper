# Shanghai Earth Wallpaper Renderer Implementation Plan

> 实际实现已按视觉验收调整：输出统一为 1320×2868，优先采用 GK2A 云地一体全圆盘，并沿上海经线构图。运行时行为以 README 与 `src/earthwall/` 为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate matching Lock and Home wallpaper images centered on Shanghai, using the newest available Himawari cloud observation and time-correct Earth lighting.

**Architecture:** A Python CLI downloads the newest valid JMA Himawari full-disk observation, reprojects it from geostationary coordinates into a Shanghai-centered globe, and renders two fixed camera presets. Static land/ocean, night-light, and procedural-star layers provide consistent art direction; only clouds, sunlight, and night lights vary with observation time.

**Tech Stack:** Python 3.12, NumPy, Pillow, pyproj, requests, pytest

---

## File structure

- `pyproject.toml`: package metadata, dependencies, CLI entry point, pytest settings.
- `src/earthwall/config.py`: Shanghai coordinates, source projection, output sizes, and camera presets.
- `src/earthwall/himawari.py`: source discovery, freshness checks, downloads, and cache fallback.
- `src/earthwall/projection.py`: output-pixel to Earth-coordinate and Himawari-pixel transforms.
- `src/earthwall/lighting.py`: UTC sun vector, day/night mask, and city-light mask.
- `src/earthwall/style.py`: tone curve, atmosphere, bloom, vignette, and procedural stars.
- `src/earthwall/render.py`: shared render pipeline and Lock/Home composition.
- `src/earthwall/cli.py`: `earthwall render` command and output manifest.
- `tests/`: source-selection, projection, lighting, and output regression tests.
- `assets/`: independently licensed static maps; no Apple wallpaper assets.
- `output/current/`: generated `lock.png`, `home.png`, and `manifest.json`.

## Fixed visual specification

- Target: Shanghai, `31.2304 N, 121.4737 E`.
- Lock: `1290 x 2796`; full globe; center approximately `(645, 1420)`; radius approximately `575 px`; top hemisphere begins below the lock-screen time safe area.
- Home: `1179 x 2556`; near-orbit crop; curved horizon begins around `y=850`; Shanghai/eastern China remains readable above the dock-safe area.
- Both outputs use the same observation timestamp, cloud field, subsolar point, and color grade.
- The renderer outputs wallpaper pixels only. Carrier, clock, icons, search pill, flashlight, camera, and dock are not baked into the images.
- Weather-app conditions are metadata/QA only. Visible cloud structure comes from satellite observations, not an invented weather effect.

### Task 1: Scaffold the renderer and lock the presets

**Files:**
- Create: `pyproject.toml`
- Create: `src/earthwall/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Add failing preset tests**

```python
from earthwall.config import HOME, LOCK, SHANGHAI

def test_shanghai_and_output_presets():
    assert SHANGHAI == (31.2304, 121.4737)
    assert LOCK.size == (1290, 2796)
    assert HOME.size == (1179, 2556)
    assert LOCK.globe_radius_px < HOME.globe_radius_px
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_config.py -q`

Expected: FAIL because `earthwall.config` does not exist.

- [ ] **Step 3: Add the package and immutable presets**

```python
from dataclasses import dataclass

SHANGHAI = (31.2304, 121.4737)

@dataclass(frozen=True)
class RenderPreset:
    size: tuple[int, int]
    center_px: tuple[float, float]
    globe_radius_px: float
    target_lat: float = SHANGHAI[0]
    target_lon: float = SHANGHAI[1]

LOCK = RenderPreset((1290, 2796), (645.0, 1420.0), 575.0)
HOME = RenderPreset((1179, 2556), (589.5, 2040.0), 1325.0)
```

- [ ] **Step 4: Run `python -m pytest tests/test_config.py -q`**

Expected: PASS.

### Task 2: Fetch one coherent Himawari observation

**Files:**
- Create: `src/earthwall/himawari.py`
- Create: `tests/test_himawari.py`

- [ ] **Step 1: Test newest-common-time selection and stale-data rejection**

```python
from datetime import UTC, datetime, timedelta
from earthwall.himawari import choose_observation

def test_choose_observation_requires_matching_true_color_and_ir():
    now = datetime(2026, 7, 13, 3, 30, tzinfo=UTC)
    files = {
        "trm": {"0320": "true.jpg"},
        "b13": {"0310": "old-ir.jpg", "0320": "ir.jpg"},
    }
    selected = choose_observation(files, now, max_age=timedelta(minutes=90))
    assert selected.time_code == "0320"
    assert selected.urls == ("true.jpg", "ir.jpg")
```

- [ ] **Step 2: Implement discovery against JMA's Full Disk index**

Use `https://www.data.jma.go.jp/mscweb/data/himawari/list_fd_.html`. Parse `fd__trm_HHMM.jpg` and `fd__b13_HHMM.jpg`, require a matching timestamp, verify HTTP `Last-Modified`, and reject observations older than 90 minutes.

- [ ] **Step 3: Implement atomic cache writes and bounded fallback**

Download to `cache/<timestamp>.tmp`, validate JPEG dimensions, then rename. If JMA is unavailable, use the newest cached pair only when it is no more than six hours old; mark `source_status: cached` in metadata.

- [ ] **Step 4: Run `python -m pytest tests/test_himawari.py -q`**

Expected: PASS with network calls mocked.

### Task 3: Acquire independent static Earth textures

**Files:**
- Create: `src/earthwall/assets.py`
- Create: `tests/test_assets.py`

- [ ] **Step 1: Test deterministic GIBS requests**

```python
from earthwall.assets import gibs_params

def test_gibs_global_texture_request():
    params = gibs_params("BlueMarble_ShadedRelief_Bathymetry")
    assert params["bbox"] == "-180,-90,180,90"
    assert params["width"] == "4096"
    assert params["height"] == "2048"
    assert params["format"] == "image/jpeg"
```

- [ ] **Step 2: Download the static NASA GIBS layers**

Request `BlueMarble_ShadedRelief_Bathymetry` and `VIIRS_CityLights_2012` from `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi` with WMS 1.1.1, EPSG:4326, `bbox=-180,-90,180,90`, and `4096x2048` output. Store source URL, retrieval time, and SHA-256 beside each asset.

- [ ] **Step 3: Validate dimensions and cache immutably**

Reject non-image responses or incorrect dimensions. Reuse a valid cached static texture because these two layers are not hourly inputs.

- [ ] **Step 4: Run `python -m pytest tests/test_assets.py -q`**

Expected: PASS with HTTP calls mocked.

### Task 4: Reproject the satellite disk into the Shanghai camera

**Files:**
- Create: `src/earthwall/projection.py`
- Create: `tests/test_projection.py`

- [ ] **Step 1: Add geometry invariants**

```python
import numpy as np
from earthwall.projection import camera_latlon_grid

def test_camera_center_is_shanghai_and_outside_is_masked():
    lat, lon, visible = camera_latlon_grid(
        size=(101, 101), center_px=(50, 50), radius_px=45,
        target_lat=31.2304, target_lon=121.4737,
    )
    assert np.isclose(lat[50, 50], 31.2304, atol=0.05)
    assert np.isclose(lon[50, 50], 121.4737, atol=0.05)
    assert not visible[0, 0]
```

- [ ] **Step 2: Implement inverse orthographic camera mapping**

For each visible output pixel, form a unit sphere normal, rotate it so the camera forward vector points at Shanghai, and convert the rotated vector to latitude/longitude.

- [ ] **Step 3: Implement Himawari geostationary sampling**

Use `pyproj.Proj(proj="geos", h=35785863, lon_0=140.7, sweep="y")` to map latitude/longitude into the source disk. Bilinearly sample TCR and B13; mask coordinates beyond the observed disk.

- [ ] **Step 4: Run `python -m pytest tests/test_projection.py -q`**

Expected: PASS.

### Task 5: Add time-correct lighting and cloud compositing

**Files:**
- Create: `src/earthwall/lighting.py`
- Create: `src/earthwall/render.py`
- Create: `tests/test_lighting.py`

- [ ] **Step 1: Test day/night continuity**

```python
import numpy as np
from earthwall.lighting import daylight

def test_daylight_is_soft_near_terminator():
    value = daylight(np.array([-0.05, 0.0, 0.05]))
    assert 0.0 < value[1] < 1.0
    assert np.all(np.diff(value) > 0)
```

- [ ] **Step 2: Calculate the solar vector from UTC**

Implement NOAA-style Julian-day solar declination and Greenwich hour angle equations. Compute `dot(surface_normal, sun_vector)` and pass it through a smoothstep band so the terminator has no hard seam.

- [ ] **Step 3: Build the cloud alpha layer**

Use visible TCR luminance on the daylight side and normalized B13 cloud-top temperature on the night side. Feather low-confidence values, preserve thin clouds, cap dense-cloud opacity, and composite clouds above the separately color-graded land/ocean layer.

- [ ] **Step 4: Run `python -m pytest tests/test_lighting.py -q`**

Expected: PASS.

### Task 6: Match the Apple-like composition and finish

**Files:**
- Create: `src/earthwall/style.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: Add output contract tests**

```python
from PIL import Image
from earthwall.render import render_pair

def test_render_pair_has_expected_dimensions(sample_observation, tmp_path):
    lock, home, _ = render_pair(sample_observation, tmp_path)
    assert Image.open(lock).size == (1290, 2796)
    assert Image.open(home).size == (1179, 2556)
```

- [ ] **Step 2: Implement shared finishing passes**

Apply a subdued cyan-blue ocean grade, lower land saturation, restrained cloud whites, a narrow cyan atmospheric rim, wide low-opacity bloom, soft global vignette, and deterministic sparse stars. Do not copy pixels from either Apple reference image.

- [ ] **Step 3: Add perceptual regression fixtures**

Render a fixed observation into `tests/golden/`; compare a 64x64 perceptual thumbnail, Earth bounding box, upper safe-area luminance, and edge-halo width. Fail if mean absolute thumbnail error exceeds the recorded tolerance.

- [ ] **Step 4: Run `python -m pytest -q`**

Expected: all tests PASS.

### Task 7: Expose repeatable generation and provenance

**Files:**
- Create: `src/earthwall/cli.py`
- Create: `README.md`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Test CLI outputs**

Run: `earthwall render --output output/current`

Expected files: `output/current/lock.png`, `output/current/home.png`, and `output/current/manifest.json`.

- [ ] **Step 2: Write the manifest**

Include observation UTC, render UTC, source URLs, cache/fresh status, center coordinates, sun coordinates, image dimensions, and SHA-256 hashes. Include the JMA/NOAA/NESDIS/CSU-CIRA acknowledgement required by the source terms.

- [ ] **Step 3: Document local generation**

Document environment setup, one-command rendering, freshness behavior, cache fallback, output locations, and the fact that hourly scheduling/iPhone wallpaper switching is a separate deployment step.

- [ ] **Step 4: Run the full verification**

Run: `python -m pytest -q && earthwall render --output output/current`

Expected: tests PASS; both PNGs open without warnings; manifest reports one matching observation timestamp.

## Acceptance criteria

- Shanghai is the camera target in both images.
- Lock and Home are visibly the same Earth at the same moment.
- Cloud locations come from a fresh or explicitly marked cached Himawari observation.
- Solar lighting matches the observation UTC; night lights appear only on the dark side.
- No phone UI or Apple-owned wallpaper pixels are included.
- A source outage never silently produces a supposedly live image.
