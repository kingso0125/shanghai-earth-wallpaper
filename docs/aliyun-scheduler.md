# 阿里云独立调度器

阿里云服务器作为 GitHub 原生 cron 之外的独立时钟。每小时 `:25`，服务器向仓库专用 `scheduler` 分支强制推送一个基于最新 `main` 的空心跳提交。`.github/workflows/hourly-watchdog.yml` 监听该分支的 push：

- GitHub 主任务 `:17` 已成功时，watchdog 判定壁纸新鲜并结束。
- 主任务漏跑时，watchdog 会在 iPhone `:45` 换图前重新渲染、QA 并发布。
- 心跳每次都从最新 `main` 建立，`scheduler` 分支只保留一个额外提交，不会无限增长历史。

## 安全边界

服务器不保存 GitHub Token。`/etc/shanghai-earth-wallpaper/deploy_key` 是仅能读写该仓库的 GitHub Deploy Key，权限为 `0600`；它无法访问用户的其他仓库。

GitHub `github-pages` Environment 的 deployment branch policy 只允许 `main` 和 `scheduler`；其他分支不能使用该环境发布 Pages。

## 服务器文件

- `/usr/local/sbin/shanghai-earth-wallpaper-trigger`：受版本控制的心跳脚本。
- `/etc/cron.d/shanghai-earth-wallpaper`：每小时 `:25` 运行，使用 `flock` 防止重入。
- `/var/log/shanghai-earth-wallpaper-trigger.log`：心跳结果。
- `/etc/logrotate.d/shanghai-earth-wallpaper`：每日轮转，保留 7 份并压缩。
