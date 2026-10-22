# V27 — 云层与地形修订

## 修复

- 地形使用 NOAA ETOPO 2022 的米制高程，按地理距离计算东西/南北坡度，转成世界坐标法线。坡度视觉放大 12 倍，加入局部谷地遮蔽；水面与原球体轮廓保留。
- 云层过滤改成浮点运算，保留小于 1/255 的密度变化。日间保留实际可见光云丝，去掉重复锐化，加入多尺度云顶起伏与太阳方向明暗。
- 夜间优先直接解码 Himawari-9 HSD Band 13 计数与逐帧校准系数，按 JMA 的 WGS84 静止卫星投影取样。替代有歧义的红外增强彩图反解，避免极冷云顶被误判为暖地表。
- 云顶材质与地表/城市灯光独立合成。灯光穿过云的透射率后才可见；厚云遮光，薄云渐透，灯光纹理不会复制到云顶。
- 校准原始数据下载并行且有时间预算，缺帧时保留同一观测时刻的有效彩图备用，并在 manifest 的 `night_cloud_input.calibrated_temperature` 中明确标识。下载中断、数据缺失、校验不通过时不会发布半套图片。
- Mac 同步服务器同一时刻的温度数据；服务器与 Pages 仍使用现有成对发布流程，每小时计划保留。

## 数据与限制

高程源：[NOAA ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model)。由 60 arc-second 数据每 4 点取样，制作成 5400 × 2700、uint16 米制陆地高程；低于海平面部分置零。生成工具为 `scripts/prepare-elevation.py`，仅离线转换需 scipy，运行时无新增依赖。

夜云源：[NOAA 公开 Himawari 归档](https://registry.opendata.aws/noaa-himawari/)，数据归属 JMA；格式与辐射校准依据 [JMA HSD User's Guide](https://ds.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v12.pdf)。保留全部十个原始分段，验证时间、波段、分段顺序、尺寸、有效覆盖后才写入缓存；错误码与非物理温度被遮罩。

本作品对 JMA/NOAA 数据进行了重投影和视觉处理。单红外通道提供的是亮温，不能独立测量云高、光学厚度或区分所有冷地表。云的厚薄、环境补光和地形夸张是视觉表现；不作为气象产品。原始温度缺失时的彩图备用仍受灰阶歧义限制，manifest 会如实标识。

## 对比与验证

- `output/cloud-relief-v27/index.html`：V26/V27 同时刻、同光照的真实昼/夜观测；含云丝、地形、夜云和灯光的原始像素裁切。
- 白天观测 2026-10-22 02:50 UTC，夜间观测 14:20 UTC；为历史对比，不能当作当前天气。
- 覆盖温度校准、极冷云顶、地理投影、缺失数据、浮点过滤、地形法线、灯光遮蔽、同一时刻备用和下载时间预算。
- 正式发布及外部读回记录另见该目录 `verification.json`。
