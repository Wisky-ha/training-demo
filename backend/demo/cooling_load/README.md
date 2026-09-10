# 冷负荷 LightGBM 演示（中国安庆气象）

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

## 模型与评估

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

- 建筑模型为美国 ASHRAE 90.1 大型办公原型，不是中国本地建筑；只有气象是中国的。
- 目标包含数据中心冷负荷基载，纯天气驱动的建筑冷负荷只占约 63.6%。
- 单年 TMY 数据无法支撑“未来年份”外推，时序切分不成立。
- 派生 CSV 约 1.4 MB，仅用于上传/运行，不纳入 Git。
