# Cloudflare 独立调度器

GitHub 原生 cron 保留为主调度。Cloudflare Cron Trigger 作为独立时钟，每小时 `:33` 调用 `hourly-watchdog.yml`：

- 主任务 `:17` 已发布时，watchdog 判定为新鲜并结束。
- 主任务漏跑时，watchdog 在手机 `:45` 换图前自动重新渲染和发布。
- Worker 不保存或处理卫星图，只负责发出 GitHub Actions 调度请求。

## 权限

在 GitHub 为仓库 `kingso0125/shanghai-earth-wallpaper` 创建 Fine-grained personal access token，仅授予：

- Repository access: 仅本仓库
- Actions: Read and write
- Metadata: Read-only

令牌只能通过 `wrangler secret put GITHUB_TOKEN` 写入 Cloudflare Secret，不得写入仓库、Worker 源码或快捷指令。

## 部署

```bash
cd cloudflare
wrangler login
wrangler secret put GITHUB_TOKEN
wrangler deploy
```

部署后使用 `wrangler triggers list`、Worker 日志和 GitHub Actions 中 `event: workflow_dispatch` 的 watchdog 运行三方交叉验证。
