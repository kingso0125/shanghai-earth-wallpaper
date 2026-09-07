# V25 / observed-volume-r2 发布记录

2026-09-07，用户确认上线云层加厚版。

## 发布范围

- 阿里云手机图片渲染、GitHub Pages 每小时出图、Mac 主屏/锁屏图的同一套材质规则。
- 保持原图片 URL、手机快捷指令、每小时运行时点和城市中心不变。
- 自动定位客户端及 `--follow-server` 不在本次启用范围；不上传手机 GPS。
- 保留每日卫星缓存清理，并纳入新增的 8K 云图缓存。

## 上线前验证

- 本机与阿里云：75 项测试全部通过。
- 内存优化前后手机/Mac 四张确认样图像素差均为 0。
- 阿里云独立预演：1450 MB 内存限制，约 41 秒完成手机双图，图片校验通过。

## 发布状态

发布完成，渲染版本 `a58e05a`，云材质标记 `observed-volume-r2`。

- [GitHub Actions 发布成功](https://github.com/kingso0125/shanghai-earth-wallpaper/actions/runs/34121710317)，执行的提交为 `a58e05a85820efac06a0dd4992b540670667a514`。曾有一次网络失败后的旧版手动运行，未作为新版成功证据；成功重推后已重新触发并核对提交。
- GitHub Pages 原有固定地址（没有查询参数）取回 Lock/Home，与同一发布 manifest 的 SHA-256 均一致。观测 11:30 UTC，光照 12:27 UTC。
- 阿里云发布 `/srv/earthwall/releases/2026-09-07T122636-062557Z`，HTTPS 下双图哈希均一致，观测 11:50 UTC，光照 12:26 UTC。`earthwall-location.service` 与 `earthwall-render.timer` 均处于 active；单次渲染成功退出。
- Mac 双图哈希一致，观测 11:50 UTC，光照 12:28 UTC。实际桌面已指向 EarthwallMac 生成的新版文件；Idle 配置仍指向 `EarthwallMac/current/mac-lock.jpg`，没有更改用户桌面/锁屏构图。
- 本次没有读取、修改或重建 iPhone 快捷指令；手机将在原有自动化触发时获取新版。

## 服务器运行与回退

- 新源码：`/opt/earthwall/releases-code/a58e05a`，`/opt/earthwall/current-code` 指向该目录。
- 两个服务的 `runtime.conf` 指向 current-code，设置 1450 MB 内存限制。已安装包已升级；保留原 `/opt/earthwall/repo` 工作目录及其此前改动，不强制清理或覆盖。
- 旧模块与原服务/计划任务配置保存在 `/opt/earthwall/backups/cloud-depth-r2-20260907/`。需要回退时可恢复旧模块、移除本次服务 drop-in、恢复计划任务后重新生成图片。用户授权删除旧图片，因此旧成品图不作为回退保留项。
- 每日 01:35 的缓存清理改用 current-code 下的脚本，覆盖 8K 云图；原每小时渲染和手机执行时点不变。

## 用户追加的空间清理

- 删除 53 项旧成品图、过期云图及独立测试目录，文件总量 135,214,658 字节（约 135 MB）。
- 清理后文件系统可用 27,210,956,800 字节，`df -h` 显示约 26 GB 可用。
- 仅保留当前成品及与当前观测对应的 4K/8K 原始云图、静态海陆/灯光/地形素材；其他项目未动。
- 完整清理清单：`/opt/earthwall/backups/cloud-depth-r2-20260907/cleanup-report.json`。

本机验证记录：原项目 `output/reference-refresh-20260907/release-verification.json`。

先前的 `reference-v25-review.md` 是候选阶段记录；当前上线状态以本文件和线上 manifest 为准。自动定位仍未启用。
