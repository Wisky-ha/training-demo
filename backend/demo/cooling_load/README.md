# 冷负荷 LightGBM 演示（EnergyPlus 模拟）

本目录包含两种数据集构建方式：

1. **多站点气候一致气象年（推荐）**：从 `forAgent/simulation_all` 的 253 份预计算模拟中，
   按冷负荷曲线一致性筛选 9 个站点拼成面板，留一站点验证 R²=0.9920。
2. **单站点安庆**：只用安庆一份气象文件，见下文“单站点安庆数据集”。

两者的建筑模型、目标定义、脚本契约完全一致，只是数据范围不同。

## 数据来源与事实

本演示不使用来历不明的“冷负荷”标签，而是用 EnergyPlus 26.1 在中国气象条件下**实际模拟**得到冷负荷：

| 项目 | 值 |
|---|---|
| 气象文件 | `CHN_AH_Anqing.584240_TMYx.2011-2025.epw`（中国安徽安庆，WMO 584240，TMYx 2011–2025） |
| 建筑模型 | `ASHRAE901_OfficeLarge_STD2019_Denver_Chiller205_Detailed_v1.0.idf`（大型办公建筑 + 详细冷水机组） |
| 模拟工具 | EnergyPlus 26.1.0，全年 8,760 小时 |
| 冷负荷定义 | 全部 `Air System Total Cooling Energy [J](Hourly)` 之和 ÷ 3600 s，单位 W |
| 原始文件 | `.../data/processed/cooling_load_raw_with_lags.csv` |
| 派生文件 | `.../data/processed/cooling_load_platform.csv` |
| 数据汇总 | `.../data/reports/cooling_dataset_summary.json` |

原始气象为安庆 TMYx，建筑模型原本附带 Denver 气象。仅替换气象文件、不改建筑与系统设定，因此得到的是**安庆气象驱动下该办公建筑的冷负荷**。

### 冷负荷构成

目标 `target_total_cooling_energy_w` 汇总 8 个空气环路：

- 建筑侧 4 个：`CAV_BAS`、`VAV_BOT/MID/TOP WITH REHEAT`
- 数据中心侧 4 个：`AIRLOOP DATACENTER BASEMENT/BOT/MID/TOP`

原始表中另存拆分列用于核对：

- `building_cooling_energy_w`：建筑侧冷负荷，均值约 426,185 W（占总量 63.6%）
- `datacenter_cooling_energy_w`：数据中心冷负荷，均值约 243,751 W，接近常开的基载

如需只预测“建筑冷负荷”，可用 `building_cooling_energy_w` 作为目标重新训练；该列有零值，MAPE 需额外处理。

## 生成派生文件

脚本不覆盖输入文件，也不修改 EnergyPlus 原始输出：

```bash
python backend/demo/cooling_load/prepare_dataset.py \
  --epw  <Anqing.epw> \
  --sim-csv <EnergyPlus 输出目录>/eplusout.csv \
  --raw-output "C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/processed/cooling_load_raw_with_lags.csv" \
  --platform-output "C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/processed/cooling_load_platform.csv" \
  --summary "C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/reports/cooling_dataset_summary.json"
```

字段结构与电力负荷完全一致：首列 `timestamp`、末列目标、`sequence_index`/`scenario_code`/`scenario_row` 加天气、日历与 `lag_1`/`lag_24`，共 8,736 行、23 个特征。`timestamp` 是**平台排序时间轴**，不是真实历史年份。

## 平台上传与执行顺序

1. 上传 `cooling_load_platform.csv` 为数据集。
2. 上传并启用 `../scripts/cooling_load_lightgbm_preprocessor.py`，脚本类型 `preprocessor`，支持模型类型任选（当前平台无冷负荷类型）。
3. 上传并启用 `../scripts/cooling_load_lightgbm_trainer.py`，脚本类型 `trainer`。
4. 提交训练任务，算法明确为 LightGBM。

预处理脚本只有一个 `Preprocessor` 类，实现 `fit(df, config)` / `transform(df, config)`；训练脚本实现 `train(X_train, y_train, X_test, y_test, config)` 并返回带 `predict(X)` 的 LightGBM 模型。执行环境必须安装 `lightgbm>=4.5,<5.0`，未安装时明确失败，不会替换为其它算法。

## 多站点气候一致气象年（推荐）

### 数据来源

`C:\EnergyPlusV26-1-0\forAgent` 已经包含完整产物，**不需要重新模拟**：

| 目录 | 内容 |
|---|---|
| `epw_flat/` | 252 份 EPW 气象文件 |
| `simulation_all/` | 253 份对应的 EnergyPlus 年模拟结果（含 8 个冷却能量列） |
| `energy consumption/*.idf` | 共享建筑模型，`Site:Location` 与设计日均为 **Shanghai**（“Denver”只是文件名） |

安庆的预计算结果与本目录自己跑出的结果逐小时完全一致（最大差 0.000000 W），
所以直接复用 `simulation_all`。

### 站点筛选（数据驱动，非人工挑选）

对全部 252 份模拟计算每小时冷负荷，用与安庆的差异排序：

```
combined = 归一化月度曲线差异 + 年度冷负荷相对差异
```

取 `combined <= 0.34` 的 9 个站点（脚本内 `SITES` 常量，含距离可复现）：

| code | 站点 | 距离 | 年冷负荷 |
|---|---|---|---|
| ANQ | Anqing, AH, CHN | 0.000 | 5,853 MWh |
| MIL | Milos, GRC | 0.229 | 5,917 MWh |
| IRA | Irako, JPN | 0.234 | 5,692 MWh |
| BST | Bust AP, AFG | 0.296 | 5,815 MWh |
| ADA | Adana Incirlik, TUR | 0.310 | 6,465 MWh |
| BEJ | Beja, TUN | 0.328 | 5,127 MWh |
| BSV | Beer Sheva, ISR | 0.334 | 6,187 MWh |
| BEN | Benghazi, LBY | 0.336 | 6,254 MWh |
| CEU | Ceuta, ESP | 0.338 | 5,652 MWh |

年度冷负荷差异 ≤ 13%，月度曲线形状差异 ≤ 0.23，属于同一冷负荷气候带。
东亚范围内只有 Irako 与 Gunsan 接近，热带站点（Taichung、HKG、Macau 等）年冷负荷
高出 57%–80%，已排除。

### 生成数据

```bash
python backend/demo/cooling_load/build_climate_years.py \
  --raw-output    ".../data/processed/cooling_load_climate_years_raw.csv" \
  --platform-output ".../data/processed/cooling_load_climate_years_platform.csv" \
  --summary       ".../data/reports/cooling_climate_years_summary.json"
```

输出 78,624 行、25 列、9 个 scenario，与电力负荷数据集结构一致。

**滞后特征在站点内生成**（`groupby(scenario).shift`），每个站点前 24 小时丢弃。
跨站点生成滞后是物理上错误的：安庆 12 月 31 日的负荷与 Milos 1 月 1 日无关。

### 训练与评估

```bash
python backend/demo/cooling_load/train_climate_years.py
```

| 评估 | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **留一站点（主指标）** | **0.9920** | **39,646 W** | 62,478 W | **15.64%** |
| 同折 lag_1 持续性基线 | 0.9244 | 101,688 W | — | 26.56% |
| 时序 80/20（平台默认） | 0.9894 | 49,111 W | — | 19.29% |
| 时序 80/20 持续性基线 | 0.9265 | 103,474 W | — | — |

9 折留一站点结果全部稳定在 R²=0.990–0.994、MAE 33,850–43,629 W。

与单站点安庆对比，多站点面板同时改善了主指标和时序切分：

| 方案 | 主指标 R² / MAE | 时序 80/20 R² / MAE |
|---|---|---|
| 单站点安庆（季节均衡切分） | 0.9860 / 52,861 W | 0.8104 / 68,073 W（劣于基线） |
| 多站点面板（留一站点） | 0.9920 / 39,646 W | 0.9894 / 49,111 W（优于基线） |

关键结论：**多站点面板让时序 80/20 重新成立**。单站点时训练期不含冬季、测试期恰是冬季，
模型必然失效；拼入气候一致的多个气象年后，训练与测试都覆盖完整年度循环，
时序切分不再是分布外外推。

产物：

```
models/lightgbm_cooling_load_climate_years_v1.joblib
results/lightgbm_cooling_climate_years/leave_one_site_out_predictions.csv
results/lightgbm_cooling_climate_years/feature_importance.csv
data/reports/lightgbm_cooling_climate_years_metrics.json
```

特征重要性：`lag_1` > `hour` > 干球温度 > `lag_24` > 星期 > 露点温度 > 直射辐射。

## 单站点安庆数据集

训练与评估脚本：

```bash
python backend/demo/cooling_load/train_and_evaluate.py
```

它会重新读取 `cooling_load_platform.csv`，按下列协议训练并覆盖导出模型与报告。

评估协议对**单年 TMY 数据**做了明确区分：

| 评估 | 说明 | R² | MAE (W) | MAPE |
|---|---|---|---|---|
| 季节均衡切分（主指标） | 每 5 个日历日抽 1 天作测试，训练/测试都含四季 | 0.9860 | 52,861 | 19.43% |
| 仅 `lag_1` 持续性基线 | 同一测试集，直接用上一小时实测值 | 0.9278 | 87,399 | 24.95% |
| 时序 80/20（压力测试） | 平台默认切分，测试期为 10 月下旬至 12 月 | 0.8104 | 68,073 | 34.98% |
| 时序 80/20 的持续性基线 | 同上 | 0.8173 | 45,625 | 22.74% |

关键结论：

- 主指标下模型明显优于持续性基线，说明模型有效。
- 时序 80/20 不是有效评估：单年数据下训练期完全不含冬季，而测试期恰好是冬季，存在结构性分布偏移；此时模型甚至弱于持续性基线，**不代表模型能力**。
- 产物：`models/lightgbm_cooling_load_v1.joblib`、`results/lightgbm_cooling/test_predictions.csv`、`results/lightgbm_cooling/feature_importance.csv`、`data/reports/lightgbm_cooling_metrics.json`。

特征重要性排序（gain）：`lag_1` > `hour` > `lag_24` > 露点温度 > 干球温度 > 水平总辐射，符合冷负荷物理规律。

## 已知限制

- 建筑模型为美国 ASHRAE 90.1 大型办公原型，但 `Site:Location` 与设计日是按上海配置的；
  只有气象逐站点替换。HVAC 自动选型仍固定在上海设计工况，对冷热差异大的站点会引入偏差。
- 目标包含数据中心冷负荷基载，纯天气驱动的建筑冷负荷只占约 63.6%。
- 多站点面板是**不同地点的相似气候**，不是同一站点的历史多年；它是合成面板，
  不能解释为真实年际变化。
- 站点筛选基于固定建筑模型下的冷负荷相似度，因此天然偏向温度驱动型气候；
  湿度差异未单独约束。
- 派生 CSV 约 13–14 MB，仅用于上传/运行，不纳入 Git。
