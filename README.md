# 实时地球壁纸

每小时生成一组参考 Apple Astronomy Earth 质感的 iPhone Lock/Home 壁纸。主路径优先使用韩国 GK2A 原生全圆盘 GeoColor；默认镜头沿上海经线 121.4737°E 对准赤道，启用手机定位后则沿当前位置经线构图，同时保持云层和地表为同一张卫星观测。

在线预览：<https://kingso0125.github.io/shanghai-earth-wallpaper/>

iPhone 快捷指令安装文件由 `shortcuts/更新上海实时地球.plist` 生成，显式区分 Lock/Home 目标；在 macOS 上运行 `scripts/build-shortcut.sh` 可重新签名并写入 `web/`。构建结束会解包校验签名，并确认发布包内的动作与源文件完全一致。

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

本机可执行 `scripts/hourly.sh`。云端使用 `.github/workflows/hourly-wallpaper.yml`，每小时渲染、质量检查并发布到 GitHub Pages；`.github/workflows/hourly-watchdog.yml` 在每小时 `:28` 检查线上发布时间，超过 50 分钟未更新时自动补跑，为手机 `:45` 换图及 CDN 缓存失效留出时间。首次使用时需要在仓库 Settings → Pages 中将 Source 设为 GitHub Actions。

作为 GitHub 原生 cron 的独立备援，阿里云服务器在每小时 `:25` 向仓库专用 `scheduler` 分支推送心跳，触发同一 watchdog。服务器只使用本仓库 Deploy Key，不保存 GitHub Token。详见 [docs/aliyun-scheduler.md](docs/aliyun-scheduler.md)。

`.github/workflows/daily-storage-cleanup.yml` 每天上海时间 01:30 删除前一天的 Earth fallback caches 与构建 artifacts；始终保留最新一份卫星缓存，Pages artifacts 另有 1 天自动过期保护。

iPhone 端配置见 [docs/iphone-shortcut.md](docs/iphone-shortcut.md)。

## 位置自适应

阿里云 HTTPS 接口接收快捷指令单次运行时取得的经纬度，不持续追踪手机。新位置与当前中心距离**超过 80 公里**才更新并立即生成；80 公里以内沿用原构图，避免 GPS 漂移。服务只保存最新位置，不保存轨迹，定位无效时继续使用上次位置；首次部署的默认位置仍为上海。

动态壁纸由 `https://earthwall.47-116-45-167.nip.io/earthwall/lock.jpg` 与 `home.jpg` 提供，并返回 `Cache-Control: no-store`，避免 GitHub Pages 固定 URL 的十分钟缓存。阿里云上的 `earthwall-render.timer` 同样在每小时 `:17` 生成，Mac 无需开机。

## 数据降级

主源为 CIRA SLIDER 的 KMA GK2A GeoColor。若主源不可用，程序依次尝试 JMA Himawari-9 与 NASA GIBS Himawari 可见光/红外；仍不可用时使用最近缓存，并在 `manifest.json` 标记 `source_status: cached`。缓存图只用于维持小时级壁纸输出，不会被标记为最新卫星观测。

GK2A 不可用时依次降级到 Himawari-9、NASA GIBS 与缓存数据；`manifest.json` 的 `render_mode: fused_geostationary_plate_shanghai_meridian`、观测源和文件哈希用于验证实际路径。

夜间画面使用 NASA GIBS `VIIRS_CityLights_2012` 的真实观测基线。灯光位置和相对分布不生成、不移动；程序只依据当前太阳位置决定夜半球，并使用当小时 GK2A 云层降低被云覆盖区域的灯光。夜光底图不是小时级电力监控数据，`manifest.json` 会明确标注这一时间属性。

`assets/space-background.jpg` 是可选的生成式美术层，只影响星空与极弱的银河尘埃；真实地球、云层和昼夜状态不会经过生成模型。没有该文件时自动使用确定性的程序化星空。

## Attribution

Satellite imagery: Korea Meteorological Administration (KMA), Japan Meteorological Agency (JMA), NOAA/NESDIS, and Colorado State University/CIRA. Static Earth and city-light imagery: NASA Earth Observatory/GIBS.
