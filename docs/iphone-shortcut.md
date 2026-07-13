# iPhone 自动换壁纸

云端发布后，固定地址为：

- Lock：`https://kingso0125.github.io/shanghai-earth-wallpaper/lock.jpg`
- Home：`https://kingso0125.github.io/shanghai-earth-wallpaper/home.jpg`
- 状态：`https://kingso0125.github.io/shanghai-earth-wallpaper/manifest.json`

## 快捷指令“更新上海实时地球”

该快捷指令已在同一 Apple ID 的 macOS 快捷指令资料库中建立，包含两组“URL → 获取 URL 内容 → 设定墙纸照片”。等待 iCloud 同步后，在 iPhone 上打开它：

若 iCloud 尚未同步，也可以从[在线预览页](https://kingso0125.github.io/shanghai-earth-wallpaper/)直接安装已经签名的“更新上海实时地球”快捷指令。安装文件已经显式指定第一张为 `Lock Screen`、第二张为 `Home Screen`，并关闭预览、主体裁切和主屏模糊。

1. 将第一组“设定墙纸照片”指定为锁定屏幕。
2. 将第二组“设定墙纸照片”指定为主屏幕，并关闭模糊与“运行时显示”。
3. 首次运行时允许联网、照片与墙纸权限。

固定图片地址的 CDN 缓存约 10 分钟。主任务漏跑时，独立 watchdog 也会尽量在 `:26` 左右完成发布，使手机在每小时 `:45` 运行前仍有约 19 分钟用于缓存失效。

## 位置自适应版本

启用后，每次运行先执行“获取当前位置”，将纬度、经度、定位精度和城市名通过 HTTPS POST 到 `https://2026.mtomorrow.com/earthwall/api/location`。请求使用设备私有 Bearer Token；Token 不写入公开仓库和安装页。

- 新位置距离当前中心不超过 80 公里：不改变视角，直接换图。
- 超过 80 公里：服务器保存新位置、立即重新生成，再返回结果。
- 定位失败或精度差于 20 公里：保留上次位置。
- 服务只保存一条最新位置，不记录历史轨迹。

位置自适应图片地址不经过 GitHub CDN：

- Lock：`https://2026.mtomorrow.com/earthwall/lock.jpg`
- Home：`https://2026.mtomorrow.com/earthwall/home.jpg`

## 每小时自动运行

iOS 没有“每小时”循环开关，因此需要 24 个每天触发一次的时间点。

在 iOS 27 Beta 中，触发器直接附加在快捷指令顶部：

1. 打开“更新上海实时地球”，点击“编辑”。
2. 点击底部“搜索操作” → “自动化” → “特定时间（Time of Day）”。
3. 将时间设为 `00:45`，重复设为“每天”，开启“自动化”，关闭“通知”。
4. 从 00:45 到 23:45 共添加 24 个时间点。所有时间点之间显示为“或”，共用下方同一组 Lock/Home 换图动作。

若使用旧版 iOS，则在“快捷指令 → 自动化”中建立相同的 24 个“特定时间”个人自动化，每个调用“更新上海实时地球”，选择立即运行并关闭通知。两种方式都只需配置一次，图片地址之后保持不变。

云端主任务安排在每小时第 17 分钟生成；阿里云在 `:25` 触发独立 watchdog，GitHub 原生 watchdog 在 `:28` 再次检查，超过 50 分钟未更新时自动补跑。手机在 `:45` 执行：正常任务有约 28 分钟生成与刷新，主任务漏跑时备援仍可为 CDN 缓存留出至少约 17 分钟。若卫星源暂时中断，服务器仍生成壁纸，但 `manifest.json` 会标记为 `cached`，不会伪装为新观测。
