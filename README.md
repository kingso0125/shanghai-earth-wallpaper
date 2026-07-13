# 实时地球壁纸

每小时生成一组参考 Apple Astronomy Earth 质感的 iPhone Lock/Home 壁纸。主路径优先使用韩国 GK2A 原生全圆盘 GeoColor；镜头沿上海经线 121.4737°E 对准赤道，使上海位于画面水平中心、地球上半部，与 Apple 参考构图一致，同时保持云层和地表为同一张卫星观测。

在线预览：<https://kingso0125.github.io/shanghai-earth-wallpaper/>

iPhone 快捷指令安装文件由 `shortcuts/更新上海实时地球.plist` 生成，显式区分 Lock/Home 目标；在 macOS 上运行 `scripts/build-shortcut.sh` 可重新签名并写入 `web/`。

## 本地生成

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
earthwall --cache cache --output output/current
earthwall-qa output/current
```

产物：

- `output/current/lock.jpg`：1320×2868
- `output/current/home.jpg`：1320×2868
- `output/current/manifest.json`：观测时间、数据状态、来源与文件哈希

## 每小时运行

本机可执行 `scripts/hourly.sh`。云端使用 `.github/workflows/hourly-wallpaper.yml`，每小时渲染、质量检查并发布到 GitHub Pages；`.github/workflows/hourly-watchdog.yml` 在每小时 `:35` 检查线上发布时间，超过 50 分钟未更新时自动补跑，为手机 `:45` 换图留出时间。首次使用时需要在仓库 Settings → Pages 中将 Source 设为 GitHub Actions。

GitHub 原生 cron 之外，`cloudflare/` 提供独立 Cron Trigger 备援，用于在 GitHub 调度延迟或漏跑时触发同一 watchdog。部署和最小权限说明见 [docs/cloudflare-scheduler.md](docs/cloudflare-scheduler.md)。

`.github/workflows/daily-storage-cleanup.yml` 每天上海时间 01:30 删除前一天的 Earth fallback caches 与构建 artifacts；始终保留最新一份卫星缓存，Pages artifacts 另有 1 天自动过期保护。

iPhone 端配置见 [docs/iphone-shortcut.md](docs/iphone-shortcut.md)。

## 数据降级

主源为 CIRA SLIDER 的 KMA GK2A GeoColor。若主源不可用，程序依次尝试 JMA Himawari-9 与 NASA GIBS Himawari 可见光/红外；仍不可用时使用最近缓存，并在 `manifest.json` 标记 `source_status: cached`。缓存图只用于维持小时级壁纸输出，不会被标记为最新卫星观测。

GK2A 不可用时依次降级到 Himawari-9、NASA GIBS 与缓存数据；`manifest.json` 的 `render_mode: fused_geostationary_plate_shanghai_meridian`、观测源和文件哈希用于验证实际路径。

`assets/space-background.jpg` 是可选的生成式美术层，只影响星空与极弱的银河尘埃；真实地球、云层和昼夜状态不会经过生成模型。没有该文件时自动使用确定性的程序化星空。

## Attribution

Satellite imagery: Korea Meteorological Administration (KMA), Japan Meteorological Agency (JMA), NOAA/NESDIS, and Colorado State University/CIRA. Static Earth and city-light imagery: NASA Earth Observatory/GIBS.
