# 电力负荷 LightGBM 演示

## 文件与数据事实

- 原始文件：`C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/processed/merged_energy_weather_with_lags.csv`
- 派生文件：`C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/processed/electric_load_platform.csv`
- 原始数据为 87,576 行、25 列：没有 timestamp 首列，首列是唯一的 `sequence_index`；`scenario` 是字符串；`liquid_precipitation_depth_mm` 全为空；目标是末列 `target_total_purchased_electricity_w`。

用准备脚本生成派生文件（脚本不覆盖原始文件）：

```bash
python backend/demo/electric_load/prepare_dataset.py \
  --input "C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/processed/merged_energy_weather_with_lags.csv" \
  --output "C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720/data/processed/electric_load_platform.csv"
```

派生的首列 `timestamp` 是由 `scenario + month/day/hour` 组成的**平台排序时间轴**，不是原始数据真实年份。源数据的 1-based `hour` 中，`hour=24` 被映射为下一日 00:00；不同 scenario 使用不重叠的时间块。脚本会按该轴排序、检查时间唯一递增、目标有限，并报告 scenario 编码。

## 字段选择

输出首列为 `timestamp`，末列保留 `target_total_purchased_electricity_w`。中间字段只保留可转为数值且至少有一个有限值的特征：原始数值天气、日历、滞后和 `sequence_index`，以及由字符串 `scenario` 编出的数值 `scenario_code`。`scenario` 本身和全空的 `liquid_precipitation_depth_mm` 不上传；其它部分缺失值交给预处理器按训练集统计量填补。

## 平台上传与执行顺序

1. 先运行上面的准备脚本，把 `electric_load_platform.csv` 上传为数据集；平台按首列/末列识别 time/target，并按固定时间顺序 80/20 划分。
2. 上传并启用 `../scripts/electric_load_lightgbm_preprocessor.py`，脚本类型选 `preprocessor`，支持模型类型选 `electric_load`；选择它执行预处理并完成数据集划分。
3. 上传并启用 `../scripts/electric_load_lightgbm_trainer.py`，脚本类型选 `trainer`，支持模型类型选 `electric_load`；在训练步骤选择该脚本。
4. 提交训练任务。算法明确为 **LightGBM 回归**，默认使用固定随机种子、验证集 early stopping 和低频日志；可在任务 `config` 中覆盖支持的 LightGBM 参数。

预处理脚本只有一个 `Preprocessor` 类，严格实现 `fit(df, config)` / `transform(df, config)`；训练脚本严格实现 `train(X_train, y_train, X_test, y_test, config)` 并返回带 `predict(X)` 的 LightGBM 模型。执行平台的 Python 环境必须先安装 `backend/requirements.txt` 或 `backend/pyproject.toml` 中约束的 `lightgbm>=4.5,<5.0`；未安装时不会改用其它算法。

派生 CSV 约 13 MB，仅用于上传/运行，不纳入 Git。
