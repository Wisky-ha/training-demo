# 模型训练可视化平台后端（内部演示版）

这是当前 FastAPI/SQLite 后端的可运行说明和 API 合同。本文以代码中的路由、Pydantic schema 和状态枚举为准；未列出的接口不是本版本合同。

## 1. 安装与启动

从仓库根目录执行：

```bash
python -m pip install -r backend/requirements-dev.txt
python -m uvicorn backend.app.main:app --reload
```

或者进入 `backend/` 后执行：

```bash
cd backend
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
# 兼容路径：/api/health（不在 OpenAPI schema 中）
```

`backend/requirements.txt` 是运行时依赖，`requirements-dev.txt` 另外包含 pytest/httpx。上传接口需要 `python-multipart`，训练模型持久化需要 `cloudpickle`；两种安装方式都已覆盖。不要执行前端构建来验证本后端步骤。

### 配置

复制 `backend/.env.example` 为 `backend/.env`，或直接设置环境变量。环境变量优先级由 pydantic-settings 处理；`APP_` 名称是推荐写法，数据库也兼容 `DATABASE_URL`。

| 变量 | 默认值/说明 |
|---|---|
| `APP_ENVIRONMENT` | `development`、`testing`、`production`，默认 `development` |
| `APP_DEBUG` | 默认 `false` |
| `APP_DATABASE_URL` / `DATABASE_URL` | `sqlite:///./data/app.db`；相对路径相对于启动工作目录 |
| `APP_MODEL_STORAGE_DIR` | `data/models`，旧配置兼容项 |
| `APP_UPLOAD_STORAGE_DIR` | `data/uploads` |
| `APP_STORAGE_ROOT` | 可选的统一制品根目录；未配置时回退到模型目录 |
| `APP_SCRIPT_STORAGE_DIR` | 可选的脚本源目录；未配置时回退到制品根目录 |
| `APP_MAX_SCRIPT_SIZE_BYTES` | 默认 5 MiB |
| `APP_MAX_DATASET_SIZE_BYTES` | 默认 50 MiB |
| `APP_ALLOWED_ORIGINS` | JSON 数组，默认 localhost/127.0.0.1:5173 |

启动生命周期会创建存储目录、执行 `create_all`/兼容升级，并幂等创建三种模型类型和 `v0-baseline`。显式初始化命令如下（在仓库根目录中）：

```bash
python -m backend.app.cli init-db
# 临时库示例：
python -m backend.app.cli init-db --database-url sqlite:///./data/demo.db
```

命令输出 `model_types=3 baselines=3` 即表示三类基线均已检查；重复执行不会新增基线。

## 2. 数据库初始化与升级行为

`init-db` 执行两件事：

1. `Base.metadata.create_all` 创建不存在的表、约束和索引；
2. 运行当前 SQLite 的兼容升级（为旧库补齐已知字段），随后由 `ModelBaselineService` 插入缺少的模型类型和每类 `v0-baseline`。

当前表为 `model_types`、`scripts`、`script_model_types`、`datasets`、`dataset_splits`、`preprocessing_tasks`、`file_artifacts`、`training_jobs`、`model_versions`、`publish_records`、`model_alerts`、`rollback_records`。升级会为旧 SQLite 库补齐：

- `training_jobs.preprocessing_task_id`、`started_at`、`current_stage`、`error_code`、`error_details`、`config`、`config_summary`、`model_version_id`；
- `preprocessing_tasks.error_code`、`error_details`；
- `model_versions.health_status`、`model_artifact_id`、`preprocessor_artifact_id`；
- `model_alerts.acknowledged_at`；
- `publish_records.idempotency_key` 及其非空唯一索引。

这是**无迁移版本号的 SQLite 兼容升级**，只做增加字段/索引，不删除表、行或制品，不会覆盖已有业务数据。它不是通用 Alembic 迁移器；升级生产副本前仍应备份数据库和 `data/`。若某模型类型已经存在一个非系统的 `v0-baseline`，初始化会报 `BASELINE_VERSION_CONFLICT` 并回滚，不会把用户版本转换为基线。

运行时目录、数据库、模型制品、上传文件、缓存和备份均由 `.gitignore` 排除；演示提交只包含 `backend/demo/` 中的小型源码/CSV。

## 3. API 合同

### 3.1 共同规则和错误响应

成功响应直接是各 endpoint 的 schema 对象（不是额外的 `data` 包装）。除上传 CSV 的历史兼容字段外，错误的规范顶层结构为：

```json
{
  "success": false,
  "error_code": "DATASET_NOT_FOUND",
  "message": "数据集不存在",
  "details": {},
  "detail": {"code": "DATASET_NOT_FOUND", "message": "数据集不存在"}
}
```

`detail` 是兼容别名，新客户端应使用 `error_code/message/details`。参数校验通常为 HTTP 422，资源不存在为 404，重复/状态冲突为 409，业务校验为 400；服务器未处理异常统一为 500 `INTERNAL_ERROR`。CSV 上传在同一规范字段之外保留 `error` 和 `errors` 兼容字段。MCP 错误也使用同一规范字段，但不附加 `detail`。

统一支持的模型类型只有：`electric_load`、`heating_cooling_load`、`integrated_energy`。本版本没有 `GET /api/model-types` 路由；模型类型由初始化命令写入。脚本类型为 `preprocessor`、`trainer`。

### 3.2 数据集、脚本、预处理和固定划分

| 方法和路径 | 请求 | 成功响应/状态 |
|---|---|---|
| `POST /api/datasets/upload` | multipart `file`，仅 `.csv` | `201`，返回 `id/file_name/row_count/columns/time_column/feature_columns/target_column/column_types/missing_value_counts/preview_rows/numeric_columns/time_parse/time_range/summary/file_storage` 等解析元数据（另有 `file_path/file_size_bytes/checksum_sha256` 平铺字段） |
| `POST /api/datasets/{dataset_id}/split` | 可选 JSON `{"preprocessing_task_id":"..."}`；不接受比例/策略 | `201 DatasetSplitResponse`，固定升序时间 80/20；重复划分 `409` |
| `GET /api/datasets/{dataset_id}/split` | 无 | `200`，持久化划分元数据；未划分 `404` |
| `POST /api/scripts/upload` | multipart `file`、`name`、`script_type`、`supported_model_types`（JSON 数组），可选 `version` | `201 ScriptResponse`；只接受 UTF-8、语法有效 `.py`；同名/类型/版本重复 `409` |
| `GET /api/scripts` | `model_type`、`script_type`、`status`、`page`、`page_size` | `200 {items,pagination}`；默认 `status=ENABLED`，`status=disabled` 可查停用项 |
| `GET /api/scripts/{id}` | 无 | `200 ScriptResponse`，含源码；不存在 `404` |
| `POST /api/scripts/{id}/enable` / `/disable` | 无 | `200 ScriptResponse`，只改变启用状态，不删除版本 |
| `POST /api/preprocessing-tasks` | JSON：`model_type`、`dataset_id`、可选 `preprocess_script_id`、`mode`=`use/skip`、`skip`、`config` | `201 PreprocessingTaskResponse`；接口同步执行并返回最终状态 |
| `POST /api/preprocessing-tasks/{id}/execute` | 无 | `200`，重新执行任务；兼容创建路径 `/api/preprocessing-tasks/execute` 不作为推荐入口 |
| `GET /api/preprocessing-tasks/{id}` | 无 | `200`，阶段、日志、前后摘要、错误和制品状态 |
| `POST /api/preprocessing-tasks/{id}/transform` | JSON：必填 `dataset_id`，可选 `config` | `200 {task_id,preprocess_used,data_source,row_count,columns,summary}`；仅复用已 fit 状态，不再次 fit |

CSV 约定：第一列是可解析且不重复的时间，最后一列是有限数值目标，至少 2 行；特征可有缺失但不可整列为空。原始 CSV 不被改写。划分响应还包括 `split_strategy=time_ordered`、`split_ratio=0.8`、`test_ratio=0.2`、两侧行数/时间范围、`sort_order=ascending`、`rounding_rule=floor(total_row_count * 0.8)` 和 `sorted_before_split`。

预处理状态为 `WAITING/RUNNING/SUCCEEDED/SKIPPED/FAILED`，阶段为 `waiting/data_reading/preprocessing/validating/completed/failed`。脚本必须定义 `Preprocessor.fit(df, config)`（返回 self）和 `transform(df, config)`（返回 DataFrame），结果必须保留时间/目标并至少有一个特征。

### 3.3 训练任务与评估

| 方法和路径 | 请求 | 成功响应/状态 |
|---|---|---|
| `POST /api/training-jobs` | JSON：`model_type`、`dataset_id`、`train_script_id`，可选 `preprocess_script_id`、`preprocessing_task_id`、`config` | `201 TrainingJobResponse`，初始 `PENDING`，随后后台执行 |
| `GET /api/training-jobs/{id}` | 无 | `200`，配置摘要、状态、阶段、日志、划分计数、错误和 `model_version_id` |
| `GET /api/training-jobs/{id}/logs` | 无 | `200 {job_id,items}` |
| `POST /api/training-jobs/{id}/retry` | 无 | `200`，仅 `FAILED` 可重试；其他状态 `409` |
| `POST /api/training-jobs/{id}/cancel` | 无 | `200`；终态重复取消幂等 |
| `GET /api/training-jobs/{id}/evaluation` | 无 | `200 EvaluationResponse`，成功任务的完整测试集 MAE/RMSE/MAPE/R²、MAPE 计数、误差、候选/基线预测及比较数据 |

训练状态为 `PENDING/RUNNING/PREPROCESSING/SPLITTING/TRAINING/EVALUATING/SUCCEEDED/FAILED/CANCELLED`。实际阶段文本包括 `准备数据`、`数据集划分`、`预处理`、`加载训练脚本`、`执行训练`、`进入评估`；没有伪造百分比进度。指标使用完整测试集，超过 1000 点时仅 `chart_data` 均匀抽样。成功任务的模型版本先短暂为 `DRAFT`，提交完成后为 `READY`，不会自动发布。

训练脚本必须定义 `train(X_train, y_train, X_test, y_test, config)` 并返回带 `predict(X)` 的对象。内部执行器限制导入和危险内建调用，但不是生产沙箱。

### 3.4 模型版本、基线、发布、异常和回滚

模型版本字段由 `ModelVersionResponse` 返回：版本 ID/类型/版本号、模型和预处理制品路径及元数据、训练/脚本快照、输入 schema、划分摘要、metrics、`status`、`health_status`、`is_baseline`、`is_current`、创建/发布时间。生命周期状态为 `DRAFT`、`TRAINING`、`READY`、`PUBLISHED`、`RETIRED`、`ABNORMAL`、`FAILED`；健康状态为 `HEALTHY`、`ABNORMAL`。

每种模型类型初始化一个不可编辑、非当前版本的 `v0-baseline`，状态 `READY/HEALTHY`。训练比较优先使用当前健康生产版本，否则使用该基线；没有独立的基线 HTTP 路由，可通过 `GET /api/models?status=READY` 看到它。

| 方法和路径 | 请求 | 成功响应/状态 |
|---|---|---|
| `POST /api/models` | `ModelSaveRequest`：`model_type`，可选 `version/model_path/model_content_base64`、制品/脚本/输入 schema/摘要/metrics；`status` 只能 `DRAFT/READY` | `201 ModelVersionResponse`；版本号默认递增 |
| `GET /api/models` | 可选 `model_type`、`status`、`health_status` | `200 ModelVersionResponse[]` |
| `GET /api/models/{id}` | 无 | `200`；不存在 `404` |
| `POST /api/models/{id}/save` | `ModelSaveRequest` | `200`，补充草稿；基线不可改 |
| `POST /api/models/{id}/publish` | 可选 `confirm`/`confirmed`/`confirmation`、`idempotency_key`、`message`/`reason` | `200 LifecycleOperationResponse`；必须确认，否则 `400`；幂等键冲突 `409` |
| `POST /api/models/{id}/offline` | 无 | `200`，下线为 `RETIRED`；不删除制品 |
| `POST /api/models/{id}/rollback` | 可选 `target_version_id`/`target_version`（也兼容 `version_id`/`version`），`reason` | `200`，含 rollback 记录和目标模型；无健康备份 `409` |
| `POST /api/models/abnormal` | `model_type`、`model_version`（也兼容 `version`/`model_version_id`）、`abnormal`、`reason` | `200`，标记异常并自动切到健康备份/基线；无备份 `409` |
| `POST /api/models/{id}/abnormal` | 可选 `abnormal`、`reason` | `200`，版本 ID 形式的兼容入口 |
| `GET /api/models/{id}/publish-records` | 无 | `200` 发布审计数组 |
| `GET /api/models/{id}/rollback-records` | 无 | `200 RollbackResponse[]` |
| `GET /api/alerts` | `model_type`、`active_only` | `200 ModelAlertResponse[]` |
| `GET /api/alerts/{id}` | 无 | `200`；不存在 `404` |
| `POST /api/alerts/{id}/acknowledge` | 无 | `200`；确认不解除告警 |

发布响应包含 `operation=publish` 和 `model`；回滚/异常响应还可能包含 `rollback`、`alert`。发布新健康版本会解析该模型类型的活动告警。手动回滚和自动异常回滚都写入 `rollback_records`，状态为 `PENDING/SUCCEEDED/FAILED`；回滚不会删除历史版本。旧客户端可使用未在 OpenAPI 中展示的 `/unpublish`、`/retire`、模型 `DELETE`、告警 `/ack`/`/confirm` 别名。

### 3.5 MCP HTTP transport

后端没有绑定第三方 MCP SDK，而是提供可由 MCP bridge 调用的 HTTP transport：

- `POST /api/mcp/predict`（兼容 `/mcp/predict`）：JSON `{"model_type":"electric_load","model_version":"v1","data":[{"timestamp":"...","outdoor_temp":5,"occupancy":12}]}`。`model_version` 可省略，使用当前健康生产版本；返回 `success`、`model_type`、`model_version`、`preprocess_used`、`predictions` 等服务结果。
- `POST /api/mcp/mark_model_abnormal`（兼容 `/mcp/mark_model_abnormal`）：JSON `model_type`、`model_version`、`abnormal`、`reason`；返回异常标记、告警和回滚结果。

MCP 输入为记录数组；未知类型/版本、输入 schema 不匹配、模型不可用和预测失败分别返回结构化 `error_code`，不返回 traceback。

## 4. 内部演示数据和推荐调用顺序

已提交的最小演示内容：

- `backend/demo/energy_demo.csv`：24 条小时记录，第一列时间、中间两列特征、最后一列 `load` 目标；
- `backend/demo/scripts/demo_preprocessor.py`：实现 `Preprocessor.fit/transform`，填充数值缺失；
- `backend/demo/scripts/demo_trainer.py`：用 `sklearn.linear_model.LinearRegression` 实现规定的 `train`；
- `backend/demo/seed_demo.py`：上传两个启用脚本和 CSV；同名脚本会复用，数据集每次上传都会产生新的不可变记录。

启动服务后，在 `docs/` 执行：

```bash
python backend/demo/seed_demo.py --base-url http://127.0.0.1:8000
```

记下输出的 `dataset_id`、`preprocessor_id`、`trainer_id`，然后按以下顺序调用（jq 只用于阅读响应，可省略）：

```bash
# D 为 seed 输出的 dataset_id，P/T 为两个脚本 ID
curl -X POST http://127.0.0.1:8000/api/preprocessing-tasks \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"electric_load","dataset_id":"D","preprocess_script_id":"P","mode":"use","config":{}}'

curl -X POST http://127.0.0.1:8000/api/datasets/D/split \
  -H 'Content-Type: application/json' -d '{"preprocessing_task_id":"PREPROCESS_TASK_ID"}'

curl -X POST http://127.0.0.1:8000/api/training-jobs \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"electric_load","dataset_id":"D","preprocess_script_id":"P","preprocessing_task_id":"PREPROCESS_TASK_ID","train_script_id":"T","config":{}}'

curl http://127.0.0.1:8000/api/training-jobs/TRAINING_JOB_ID
curl http://127.0.0.1:8000/api/training-jobs/TRAINING_JOB_ID/evaluation
curl -X POST http://127.0.0.1:8000/api/models/MODEL_VERSION_ID/publish \
  -H 'Content-Type: application/json' -d '{"confirm":true,"message":"内部演示发布"}'

curl -X POST http://127.0.0.1:8000/api/mcp/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"electric_load","data":[{"timestamp":"2024-01-02T00:00:00","outdoor_temp":5,"occupancy":12}]}'
```

`PREPROCESS_TASK_ID` 和 `TRAINING_JOB_ID` 需要分别从上一步响应取得；训练任务需要轮询到 `SUCCEEDED`，再用其 `model_version_id` 发布。若想跳过预处理，直接用 `mode=skip` 创建任务，并在 split 时省略请求体；训练创建时也省略两个预处理字段。

## 5. 验证命令和限制

```bash
python -m pytest -q backend/tests
python -m compileall -q backend/app backend/demo
python -m backend.app.cli init-db
python -m backend.app.cli init-db  # 第二次应仍输出 baselines=3
```

演示版已知限制：SQLite 单机存储、进程内后台线程池（最多 2 个训练任务）、脚本执行不是生产级沙箱，不提供高并发隔离、分布式训练、容器/进程级资源限制、生产级权限/审计或对象存储。不要在不受信任的生产环境上传或执行任意脚本；`data/` 下的运行数据库、模型制品、缓存和上传文件不应提交 Git。

## 6. 目录结构

```text
backend/
├── app/
│   ├── main.py                 # FastAPI/lifespan/统一异常边界
│   ├── cli.py                  # init-db
│   ├── datasets/               # CSV 上传和固定划分
│   ├── scripts/                # 脚本库 API
│   ├── preprocessing/          # 预处理 API
│   ├── training_jobs/          # 训练任务 API
│   ├── model_router.py         # 版本、发布、异常、告警 API
│   ├── mcp_router.py           # MCP HTTP transport
│   ├── services/               # 业务执行与生命周期
│   ├── db/                     # SQLite/SQLAlchemy 模型、初始化和兼容升级
│   └── schemas/                # Pydantic 请求/响应合同
├── demo/                       # 仅小型可复现演示输入和脚本
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```
