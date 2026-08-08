# Cinematic Earth V2 Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce isolated iPhone Lock and Home preview images with more natural clouds, warmer filmic color, softer light, and physically coherent limb treatment while preserving the current observed geography and cloud field.

**Architecture:** Add an experimental renderer beside the production renderer. It reuses the existing acquisition and lighting contracts, but owns its projection, material shading, cloud optical-depth treatment, atmosphere, and tone mapping. Outputs go only to `output/preview-v2`, so existing phone and Mac automation paths remain unchanged.

**Tech Stack:** Python 3.11+, NumPy, Pillow, existing Earthwall acquisition modules.

---

### Task 1: Lock the V2 preview contract

**Files:**
- Create: `tests/test_preview_v2.py`
- Create: `src/earthwall/preview_v2.py`

- [ ] **Step 1: Write the failing projection and output-isolation tests**

```python
def test_preview_presets_do_not_reuse_production_paths():
    assert preview_output("lock").as_posix() == "output/preview-v2/lock.jpg"

def test_perspective_center_points_at_requested_location():
    lat, lon, visible, _, _ = perspective_camera_grid(V2_LOCK)
    y, x = map(round, (V2_LOCK.center_px[1], V2_LOCK.center_px[0]))
    assert visible[y, x]
    assert np.rad2deg(lat[y, x]) == pytest.approx(V2_LOCK.target_lat, abs=0.05)
    assert np.rad2deg(lon[y, x]) == pytest.approx(V2_LOCK.target_lon, abs=0.05)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: import failure because `earthwall.preview_v2` does not exist.

- [ ] **Step 3: Add independent V2 presets, output paths, and perspective ray-sphere projection**

```python
V2_LOCK = V2Preset("lock", (1320, 2868), (660.0, 1480.0), 0.82, 25.23, 121.47)
V2_HOME = V2Preset("home", (1320, 2868), (660.0, 2470.0), 0.62, 8.0, 121.47)

def preview_output(name: str) -> Path:
    return Path("output/preview-v2") / f"{name}.jpg"
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: PASS.

### Task 2: Build physically coherent surface, ocean, and lighting

**Files:**
- Modify: `src/earthwall/preview_v2.py`
- Modify: `tests/test_preview_v2.py`

- [ ] **Step 1: Add regression tests for material separation**

```python
def test_ocean_specular_is_daylight_only():
    day = shade_surface(sample_surface(), sample_normals(), sample_sun(), np.ones((8, 8)))
    night = shade_surface(sample_surface(), sample_normals(), -sample_sun(), np.ones((8, 8)))
    assert day[..., 2].max() > night[..., 2].max()

def test_filmic_curve_preserves_highlight_order():
    values = np.array([0.25, 1.0, 4.0], dtype=np.float32)
    mapped = aces_tonemap(values)
    assert 0 < mapped[0] < mapped[1] < mapped[2] <= 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: missing material and tone-map functions.

- [ ] **Step 3: Implement linear-light Lambert shading, DEM-derived normals, ocean Fresnel/specular, coastal cyan, and ACES tone mapping**

```python
def aces_tonemap(rgb):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e), 0.0, 1.0)
```

The surface must use the static map only as albedo. Relief contributes normal perturbation, never baked color or a second lighting direction.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: PASS.

### Task 3: Reconstruct observed clouds as a layered optical medium

**Files:**
- Modify: `src/earthwall/preview_v2.py`
- Modify: `tests/test_preview_v2.py`

- [ ] **Step 1: Add cloud invariance and depth tests**

```python
def test_zero_cloud_signal_does_not_change_surface():
    result = composite_clouds(SURFACE, np.zeros((9, 9)), DAY, SUN_DIR)
    np.testing.assert_allclose(result, SURFACE, atol=1e-6)

def test_dense_cloud_is_brighter_and_more_opaque_than_thin_cloud():
    alpha = np.array([[0.2, 0.9]], dtype=np.float32)
    result = cloud_optical_properties(alpha, np.ones_like(alpha))
    assert result.opacity[0, 1] > result.opacity[0, 0]
    assert result.radiance[0, 1].mean() > result.radiance[0, 0].mean()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: missing cloud functions.

- [ ] **Step 3: Implement multi-scale optical depth, directional cloud-top light, soft displaced shadows, and cloud-occluded night lights**

Cloud structure must come only from the current visible/infrared observation. Processing may change opacity and lighting but must not generate, move, or erase weather systems.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: PASS.

### Task 4: Add asymmetric atmosphere and preview CLI

**Files:**
- Modify: `src/earthwall/preview_v2.py`
- Create: `src/earthwall/preview_cli.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_preview_v2.py`

- [ ] **Step 1: Add atmosphere and manifest tests**

```python
def test_atmosphere_is_thin_and_brighter_on_sunward_limb():
    rim, halo = atmosphere_scattering(MASK, NORMALS, SUN_DIR)
    assert rim[MASK].mean() > 0
    assert halo[~MASK].max() > 0

def test_preview_manifest_marks_nonproduction_output():
    assert build_manifest(OBSERVATION)["preview_only"] is True
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python3 -m pytest tests/test_preview_v2.py -q`

Expected: missing atmosphere and manifest functions.

- [ ] **Step 3: Implement sunward Rayleigh-style rim, restrained bloom, sparse background, render pair, and CLI**

```toml
[project.scripts]
earthwall-preview-v2 = "earthwall.preview_cli:main"
```

- [ ] **Step 4: Run all automated tests**

Run: `python3 -m pytest -q`

Expected: all tests pass.

### Task 5: Render and visually review the preview pair

**Files:**
- Create: `output/preview-v2/lock.jpg`
- Create: `output/preview-v2/home.jpg`
- Create: `output/preview-v2/manifest.json`

- [ ] **Step 1: Acquire the latest observation and render isolated previews**

Run: `earthwall-preview-v2 --cache cache --output output/preview-v2`

Expected: two 1320x2868 JPEGs and one preview manifest; no writes to `output/current` or cloud publishing paths.

- [ ] **Step 2: Check image dimensions, timestamps, and source freshness**

Run: `python3 -m earthwall.qa output/preview-v2`

Expected: dimensions match and the preview observation is current or explicitly marked cached.

- [ ] **Step 3: Perform visual comparison against the supplied Apple Lock/Home references**

Acceptance: no paper-white cloud sheet, no neon rim, no uniform cyan ocean, no clipped highlights, visible cloud thickness, soft cloud shadows, Shanghai composition retained, and clear Lock/Home safe areas.

- [ ] **Step 4: Present the preview pair for user confirmation**

Do not deploy or modify phone/Mac automation until the user approves the V2 direction.
