# 卫星过境观测 API（Satellite Pass Observation API）

给定一个历元时刻的卫星 ECI 状态与地面观测站坐标，预测指定 UTC 时间区间内
卫星相对观测站的可见窗口（升/落时刻、峰值仰角及其时刻、方位角等）。

- **语言/框架**：Python 3.11+、FastAPI、NumPy、Pydantic v2
- **动力学**：纯两体开普勒传播（WGS84 引力常数），牛顿迭代解开普勒方程
  （带回退二分法），机器精度守恒
- **坐标系统**：固定 WGS84 椭球与固定地球自转率
  （ω = 7.2921151467×10⁻⁵ rad/s），IAU 1982 GMST 模型做 ECI↔ECEF 旋转；
  观测站按 WGS84 大地坐标转换为 ECEF，再到站心地平坐标（东-北-天，ENU）
- **求交策略**：两级自适应扫描 + 二分精确求仰角穿越 + 黄金分割精化峰值，
  搜索区间按**半开区间 [start, end)** 处理。双端低于门限的采样区间只有在
  **仰角一阶变化率界（K1 帐篷证明）或二阶曲率界（解析端点速率 + 严格的
  |仰角″| 上界，Taylor 包络证明）**排除内部穿越后才计为已覆盖；否则持续
  对分细分直至 1e-7 s 数值时间分辨率，仍无法证明的残差（及原因）进入
  `uncovered_intervals`，绝不因步长下限而宣称 complete。

---

## 1. 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`：

| 用途 | 地址 |
| --- | --- |
| 健康检查 | `GET http://localhost:8000/health` |
| 过境预测 | `POST http://localhost:8000/passes` |
| 交互式文档（Swagger UI） | `http://localhost:8000/docs` |
| OpenAPI 模式 | `http://localhost:8000/openapi.json` |

停止服务：`docker compose down`。

### 不使用 Docker（本地运行）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

运行测试：

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```

---

## 2. 请求格式（POST /passes）

Content-Type: `application/json`。所有时间必须是**带时区的 UTC**
（ISO-8601，推荐以 `Z` 结尾）。

| 字段 | 类型 | 单位 / 范围 | 说明 |
| --- | --- | --- | --- |
| `epoch` | string | UTC 时间 | 卫星状态向量的历元 |
| `start` / `end` | string | UTC 时间 | 搜索区间，`end > start`，跨度 **≤ 7 天** |
| `min_elevation_deg` | number | 度，(-90, 90) | 最低仰角（含该仰角即为可见） |
| `satellite.position_km` | number[3] | **km**，有限数 | 历元 ECI 位置 [x, y, z] |
| `satellite.velocity_km_s` | number[3] | **km/s**，有限数 | 历元 ECI 速度 [vx, vy, vz] |
| `station.latitude_deg` | number | 度，[-90, 90] | WGS84 大地纬度 |
| `station.longitude_deg` | number | 度，[-180, 180] | WGS84 经度 |
| `station.altitude_km` | number | **km**，[-12, 100000] | WGS84 椭球高 |

### 422 校验规则

以下情况返回 **HTTP 422**（FastAPI/Pydantic 标准错误体）：

- 时间顺序非法（`end <= start`）或跨度超过 7 天；时间不带时区
- 经纬高、最低仰角超出物理范围；向量长度不是 3 或含 NaN/Infinity
- 轨道退化：位置/速度为零、r 与 v 共线（角动量为零）
- 轨道非束缚：抛物线/双曲线（比机械能 ≥ 0）或离心率 ≥ 1
- 近地点位于地表以下（两体传播无物理意义）
- 未知字段（请求体禁止额外字段）

> 单位提示：位置必须用 **km**、速度必须用 **km/s**。若误把位置按米
> （放大 1000 倍）提交，近地点将落到地球内部，服务会以 422
> `orbit_impact_earth` 拒绝；速度按 m/s 提交则会因双曲线轨道被拒绝。

---

## 3. 实际请求示例

以下是一次真实请求（420 km 圆轨道、倾角 51.6° 的类 ISS 卫星，
北京附近观测站，最低仰角 10°，搜索 6 小时；完整文件见
[`examples/request.json`](examples/request.json)）：

```bash
curl -X POST http://localhost:8000/passes \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

请求体（即 `examples/request.json`）：

```json
{
  "epoch": "2025-06-01T00:00:00Z",
  "start": "2025-06-01T00:00:00Z",
  "end": "2025-06-01T06:00:00.000Z",
  "min_elevation_deg": 10.0,
  "satellite": {
    "position_km": [4270.307144917519, -1323.5276198332833, 5121.271150639637],
    "velocity_km_s": [3.950957000514434, 6.347260232247907, -1.6540865405752112]
  },
  "station": {
    "latitude_deg": 39.9,
    "longitude_deg": 116.4,
    "altitude_km": 0.05
  }
}
```

### 实际响应（HTTP 200）

```json
{
  "status": "complete",
  "count": 1,
  "windows": [
    {
      "start": "2025-06-01T00:01:40.294556Z",
      "end": "2025-06-01T00:08:29.127934Z",
      "peak_time": "2025-06-01T00:05:05.711370Z",
      "peak_elevation_deg": 85.67588450928231,
      "peak_azimuth_deg": 218.31383088183551,
      "rise_azimuth_deg": 306.5090306738104,
      "set_azimuth_deg": 130.01647921521067,
      "max_range_km": 1521.86280352021,
      "duration_s": 408.83337833246924,
      "start_open": false,
      "end_open": false,
      "truncated": false
    }
  ],
  "uncovered_intervals": [],
  "search_start": "2025-06-01T00:00:00Z",
  "search_end": "2025-06-01T06:00:00Z",
  "min_elevation_deg": 10.0,
  "diagnostics": {
    "orbit": {
      "semi_major_axis_km": 6798.137,
      "eccentricity": 3.6e-16,
      "period_s": 5578.22,
      "perigee_radius_km": 6798.137,
      "apogee_radius_km": 6798.137,
      "perigee_altitude_km": 420.0,
      "apogee_altitude_km": 420.0
    },
    "coarse_samples": 297,
    "dense_samples": 1107,
    "search_span_s": 21600.0,
    "uncovered_seconds": 0,
    "coverage_fraction": 1.0,
    "notes": "Elevation crossings refined by bisection; peaks by golden-section search. Half-open search interval [start, end). Below-threshold brackets are certified by a first-order elevation-rate bound and a second-order curvature bound using the analytic endpoint rates; brackets that cannot be proven are subdivided down to the 1e-7 s numerical time resolution, and anything still unprovable is listed in uncovered_intervals rather than assumed clear."
  }
}
```

字段含义：

- `windows[]` 按 **`start` 升序**排列；每个窗口含起止时刻、峰值时刻、
  峰值仰角（度）、峰值/升/落方位角（度，正北顺时针）、最大斜距（km）、
  持续时间（秒）。
- `start_open` / `end_open`：窗口在搜索区间起点已可见 / 到区间终点仍可见
  （半开区间被边界截断）。
- `truncated`：窗口因数值失败被提前截断（此时 `status` 为 `partial`）。
- `uncovered_intervals`：因传播或求根数值失败而**未能覆盖的时间区间及
  原因**，不为空时 `status` 为 `partial`（有结果但不完整）或 `failed`
  （没有任何可信窗口）。
- `diagnostics`：轨道根数、粗/细扫描采样数、未覆盖秒数与覆盖率。

### 422 响应示例（搜索区间 8 天 > 7 天上限）

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, search span (end - start) must not exceed 7 days"
    }
  ]
}
```

轨道退化/非束缚错误示例：

```json
{
  "detail": [
    {
      "type": "orbit_degenerate_or_invalid",
      "loc": ["body", "satellite"],
      "msg": "orbit is not bound (parabolic/hyperbolic): only elliptic orbits are supported"
    }
  ]
}
```

---

## 4. 算法说明

### 4.1 两体传播

由历元状态计算比机械能 ε = v²/2 − μ/r 与离心率矢量，要求 ε < 0、
e < 1。任意时刻 t = t₀ + Δt 用开普勒方程 M = E − e·sin E
（牛顿迭代，容差 1e-10，失败时二分回退），再在由 h、e 矢量构造的
近焦点坐标系中还原 r、v。圆轨道（e≈0）使用 r 方向作为面内基准，
无近点奇点。自测：一个周期后状态误差为 0，3 天内能量/角动量
相对漂移 < 1e-15。

### 4.2 坐标与时角

- GMST 使用 IAU 1982 模型（J2000 锚点 18h41m50.543s）；
- ECI→ECEF：`r_ECEF = Rz(−GMST)·r_ECI`，速度另减 ω×r 项；
- ENU 基由测站纬经度固定，仰角 `asin(ρ_up/|ρ|)`、
  方位角 `atan2(ρ_E, ρ_N)`；并给出仰角/方位角解析变化率供自适应用。

### 4.3 两级自适应扫描（不漏窗口）

1. **粗扫（几何门限）**：只算卫星—测站的**地心角** α。可见要求
   cos α ≥ R站/r星，因此用远地点构造最松门限 ψ_flag（地平角 + 8° 缓冲）。
   由近地点速度给出严格的地心角变化率上界
   Kα = v_p/r_p + ω_E，粗扫步长固定为 5°/Kα（LEO 约 40–75 s），
   **任何可能可见的几何都不可能被跨过**。标记点 ±一个粗步长合并为候选段。
2. **细扫（仰角门限）**：在候选段内按仰角距阈值的余量和全局 Lipschitz 界
   K1 = (v_p + ω_E·r_a)/(r_p − R站) 自适应取步（下限 0.02 s 仅影响
   性能，不构成完整性保证）。对每个**两端都低于门限**的采样区间执行
   认证，只有通过下述任一时段性证明才记为已覆盖：
   - **一阶证明（帐篷界）**：K1·w ≤ ½(margin_a + margin_b)，端点仰角的
     一阶 Lipschitz 锥在整个区间内不可能到达门限；
   - **二阶证明（曲率界）**：由解析的端点仰角变化率与区间内严格的
     |仰角″| ≤ K2 上界（由 ECEF 重力 + 离心 + 科氏加速度、站心距下界与
     cos(仰角) 下界推出，仅在区间内可达仰角 < 89° 时使用），构造两条
     Taylor 上包络抛物线并检查其最大值：包络处处低于门限即证明无穿越。
   证明不成立则**对分细分**：中点仰角仍低于门限则对两个子区间继续认证；
   中点达到/高于门限且高出量超过 1e-6° 仰角求值分辨率，说明内部藏有
   过境（即使其宽不足 0.02 s），分别二分精化升、落穿越，再回头认证
   两侧肩部。细分在以下任一数值分辨率处停止并把锥间隙残差登记为
   `uncovered_intervals`（附原因）：1e-7 s 浮点时间分辨率；或区间内采样
   最高仰角高出门限不足 1e-6°（GMST/开普勒求值在亚微秒尺度的噪声地板
   约 3e-7°，此时穿越存在与否无法与噪声区分，既不输出噪声窗口也不
   宣称已覆盖）；认证采样超过 5 万节点的预算保护同样登记为未覆盖。
3. **精确求交**：符号变化的相邻采样对做 45 次二分（时间精度 1e-8 s，
   端点仰角误差 < 1e-7°）；认证内部发现的鼓包同样二分。
4. **峰值**：以最佳细扫采样及其相邻采样为支架做 32 次黄金分割，
   且结果永不低于实际观测到的最大采样值。

**为什么短擦边过境不再漏**：旧实现把步长下限 0.02 s 当成完整性论据——
峰值仅略高于门限的短窗口夹在两个低值采样点之间会被整段跳过，却仍返回
空窗口、空 `uncovered_intervals` 与 `complete`。现在区间的“已覆盖”状态
只来源于可验证的数学证明（一阶/二阶界）或细分到数值分辨率后的显式
结论；凡是证明不了的时间都如实出现在 `uncovered_intervals` 中，
`status` 降级为 `partial`/`failed`，`coverage_fraction` 同步下降。
对真实两体动力学，曲率界使典型切触（峰值恰好不超过门限）在 O(10²)
次求值内被严格排除；物理上真实存在的亚 0.02 s 窗口（实测
~0.014 s 宽、峰高出 5e-6°）可被检出，升落时刻与 0.5 ms 独立网格吻合
至 0.002 s；峰高出量低于 1e-6° 求值分辨率的接触则如实登记为
`uncovered_intervals`、`status=failed`。与 0.25 s 均匀网格真值对比
（1 天类 ISS 轨道）：窗口数完全一致，常规过境升/落时刻误差 < 0.05 s，
峰值仰角误差 < 0.001°。

### 4.4 半开区间与失败报告

- 搜索域为 **[start, end)**：恰在 `end` 发生的降落穿越按区间内处理
  （窗口闭合于该时刻）；恰在 `start` 已可见则标记 `start_open=true`，
  到 `end` 仍可见标记 `end_open=true`。
- 零时长切触（峰值恰好等于阈值）不作为窗口返回；持续不足 1e-6 s 的
  亚时间分辨率接触，或高出门限不足 1e-6°（仰角求值噪声地板）的接触，
  登记为 `uncovered_intervals`（原因注明低于时间/仰角求值分辨率、
  可见性无法分类），而不是静默丢弃或返回噪声驱动的假窗口。
- 任何传播/求值/二分失败，或区间细分到数值分辨率（1e-7 s 时间 /
  1e-6° 仰角）后仍无法由一/二阶变化率界排除内部穿越：受影响时段进入
  `uncovered_intervals`（含起止时刻与原因），打开中的窗口标记
  `truncated`，`status` 降级为 `partial`/`failed`，
  `diagnostics.coverage_fraction` 与 `uncovered_seconds` 同步反映未覆盖
  时长，绝不静默丢弃或凭步长下限宣称 complete。

### 4.5 性能

典型开销（单机纯 Python）：7 天类 ISS 任务约 8000 次粗扫 + 7.5 万次
细扫求值，耗时约 8 秒；6 小时任务约 0.1–0.5 秒。高轨（GEO、闪电轨道）
更省或相当。

---

## 5. 项目结构

```
app/
  main.py        # FastAPI 路由、422 映射、响应组装
  schemas.py     # Pydantic 请求/响应模型与范围校验
  mechanics.py   # WGS84 常数、两体开普勒传播
  geometry.py    # 大地坐标、GMST、ECI/ECEF、ENU、仰角/方位角
  passes.py      # 两级自适应扫描、二分穿越、黄金分割峰值
  time_utils.py  # UTC/儒略日/GMST
examples/        # 真实请求、成功响应与 422 响应
tests/           # pytest：27 项（含独立网格真值交叉验证、亚 0.02 s 擦边
                 # 过境回归、未覆盖登记、422 全分支）
Dockerfile, docker-compose.yml, requirements.txt
```
