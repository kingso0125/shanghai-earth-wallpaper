# iPhone 自动换壁纸

云端发布后，固定地址为：

- Lock：`https://kingso0125.github.io/shanghai-earth-wallpaper/lock.jpg`
- Home：`https://kingso0125.github.io/shanghai-earth-wallpaper/home.jpg`
- 状态：`https://kingso0125.github.io/shanghai-earth-wallpaper/manifest.json`

## 快捷指令“更新上海实时地球”

该快捷指令已在同一 Apple ID 的 macOS 快捷指令资料库中建立，包含两组“URL → 获取 URL 内容 → 设定墙纸照片”。等待 iCloud 同步后，在 iPhone 上打开它：

1. 将第一组“设定墙纸照片”指定为锁定屏幕。
2. 将第二组“设定墙纸照片”指定为主屏幕，并关闭模糊与“运行时显示”。
3. 首次运行时允许联网、照片与墙纸权限。

固定图片地址的 CDN 缓存约 10 分钟，远短于 1 小时更新间隔；手机在每小时 `:25` 运行时，上一小时的缓存已经失效。

## 每小时自动运行

iOS 的个人自动化没有一个“每小时”循环开关。稳定做法是在“快捷指令 → 自动化”中建立 24 个“特定时间”自动化，从 00:25 到 23:25，每个调用同一个“更新上海实时地球”快捷指令，并关闭“运行前询问”和“运行时通知”。只需建立一次，图片地址之后保持不变。

云端任务安排在每小时第 17 分钟生成。如果手机自动化使用 `:25`，通常能取得本小时的新图片；若卫星源暂时中断，服务器仍生成壁纸，但 `manifest.json` 会标记为 `cached`，不会伪装为新观测。
