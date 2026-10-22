# Astronomy Matte V26 · 2026-10-22

## 已发布

- 服务器代码：`faa1e0f`，当前代码目录 `/opt/earthwall/releases-code/faa1e0f`。
- 手机发布工作流：`943b88f`；[正式运行](https://github.com/kingso0125/shanghai-earth-wallpaper/actions/runs/39808408937)成功。
- 材质标记：`astronomy-matte-v26` / `observed-volume-r3`。
- 上海中心、手机和 Mac 原构图及分辨率保持原值。

## 改动

- 降低海面镜面反光，减少陆地过度暖化；夜景保留深色海洋，收敛金色灯斑和夜云亮度，缩窄大气外晕。
- 保留同一观测的云系、地理投影与云下城市灯光；显示材质没有生成新云系或修改城市位置。
- JPEG 继续使用 quality=97 和 4:4:4，增加渐进编码及体积优化。同条件 A/B 双图总量减少：白天 11.1%、黄昏 14.6%、夜间 19.5%。这些差异包含材质和编码调整的共同影响。
- 下载对临时 TLS、服务端限流与 HTTP 数据中断做有限重试。`IncompleteRead` 归入既有网络回退路径，8K 请求失败时可保留有效的同观测 4K 双通道；缓存文件也检查完整性。
- Mac 优先使用服务器已验证、未过期且覆盖当前中心的原始观测，避免先等待一次独立外网请求；可同步同观测的 8K 云图。
- Pages 在每小时 `:23` 优先复制服务器不可变版本的 Lock/Home，检查年龄、SHA-256、尺寸和画面质量；失败才独立渲染。`:25` 服务器心跳与 `:28` watchdog 保留。新的补跑不再取消正在执行的发布，单次发布上限 12 分钟。

## 验证证据

- 本地、正式服务器、GitHub Actions 各 92 项测试通过。
- 使用同一历史卫星观测，对白天、黄昏、夜间做同条件 A/B；共 12 张原生手机图通过质量检查。这些是光照校准样图，不是当前天气实况。
- 服务器在 1450 MB 限制下预演和正式生成成功，位置服务与小时定时器 active。
- 正式服务器双图生成于 `2026-10-22T15:48:31.820887Z`，卫星观测 `2026-10-22T14:30:00Z`。手机固定地址取回双图，与服务器 manifest 的 SHA-256 完全相同。
- GitHub 发布日志明确采用 `verified-server-release`，并非独立补图。
- Mac 双图通过质量与哈希检查，实际桌面已指向新生成的 `mac-home-20261022T234535.jpg`。
- 手机端证据为固定 URL 的实际图片，尚未在 iPhone 上执行快捷指令；这与云端生成/发布成功是不同的验收层。

本地前后对比：`output/upgrade-20261022/index.html`。
完整核验：`output/upgrade-20261022/verification.json`；运行日志：`publication.log`、`mac-refresh.log`。
视觉方向参照 [Apple Astronomy 官方介绍](https://www.apple.com/newsroom/2022/06/apple-unveils-new-ways-to-share-and-communicate-in-ios-16/)。

## 服务器回退

原代码指向和两个 systemd runtime 配置备份在 `/opt/earthwall/backups/v26-20261022/`。需要回退时恢复 `original-current-code` 记录的代码软链接及 `earthwall-render-runtime.conf`、`earthwall-location-runtime.conf`，执行 `systemctl daemon-reload`、重启位置服务并重新运行小时渲染。Pages 将继续同步恢复后的服务器成图。原图片 URL 和位置数据无需修改。
