# 前端 API 接口面（自动生成）

> 生成时间：2026-09-06 09:52:38
>
> 来源：`frontend/src/api/client.ts`（方法/路径）、`frontend/src/types/contracts.ts`（类型字段）、`frontend/src/pages|components`（页面使用统计）
>
> 此文档由脚本覆盖生成，请勿手改。修改源码后重新执行：`npm run api:doc`（在 frontend 目录）

## 总览

| 指标 | 数值 |
|---|---|
| 业务接口方法数 | 27 |
| 被页面调用 | 17 |
| 零调用（仅封装未使用） | 10 |

## 一览表

| # | 方法 | HTTP | 路径 | 入参（类型） | 返回类型 | 页面使用 |
|---|---|---|---|---|---|---|
| 1 | `getHealth` | GET | `/api/health` | `RequestOptions` | `HealthResponse` | 未使用 |
| 2 | `uploadDataset` | POST (multipart) | `/api/datasets/upload` | `file: File, options: DatasetUploadOptions = {}` | `DatasetUploadResult` | 1（pages/WorkflowPage.tsx） |
| 3 | `createPreprocessingTask` | POST | `/api/preprocessing-tasks` | `input: { model_type: ModelTypeCode dataset_id: EntityId preprocess_script_id?: EntityId | …` | `PreprocessTask` | 1（pages/WorkflowPage.tsx） |
| 4 | `getPreprocessingTask` | GET | `/api/preprocessing-tasks/{id}` | `EntityId` | `PreprocessTask` | 未使用 |
| 5 | `splitDataset` | POST | `/api/datasets/{datasetId}/split` | `EntityId` | `DatasetSplitResult` | 1（pages/WorkflowPage.tsx） |
| 6 | `getDatasetSplit` | GET | `/api/datasets/{datasetId}/split` | `EntityId` | `DatasetSplitResult` | 1（pages/WorkflowPage.tsx） |
| 7 | `listScripts` | GET | `/api/scripts` | `ListScriptsParams` | `PaginatedResponse<ScriptContract>` | 1（pages/WorkflowPage.tsx） |
| 8 | `getScript` | GET | `/api/scripts/{id}` | `EntityId` | `ScriptContract` | 未使用 |
| 9 | `enableScript` | POST | `/api/scripts/{id}/enable` | `EntityId` | `ScriptContract` | 未使用 |
| 10 | `disableScript` | POST | `/api/scripts/{id}/disable` | `EntityId` | `ScriptContract` | 未使用 |
| 11 | `uploadScript` | POST (multipart) | `/api/scripts/upload` | `ScriptUploadInput` | `ScriptContract` | 未使用 |
| 12 | `createTrainingJob` | POST | `/api/training-jobs` | `CreateTrainingJobRequest` | `TrainingJob` | 1（pages/WorkflowPage.tsx） |
| 13 | `retryTrainingJob` | POST | `/api/training-jobs/{id}/retry` | `EntityId` | `TrainingJob` | 1（pages/WorkflowPage.tsx） |
| 14 | `getTrainingJob` | GET | `/api/training-jobs/{id}` | `EntityId` | `TrainingJob` | 1（pages/WorkflowPage.tsx） |
| 15 | `getTrainingJobLogs` | GET | `/api/training-jobs/{id}/logs` | `id: EntityId, params?: { since?: string; limit?: number },` | `TrainingLogsResponse` | 未使用 |
| 16 | `getTrainingJobEvaluation` | GET | `/api/training-jobs/{id}/evaluation` | `EntityId` | `ModelEvaluation` | 1（pages/WorkflowPage.tsx） |
| 17 | `saveModel` | POST | `/api/models/{id}/save` | `EntityId, ModelSaveRequest` | `ModelVersionSummary` | 1（pages/WorkflowPage.tsx） |
| 18 | `publishModel` | POST | `/api/models/{id}/publish` | `id: EntityId, input: PublishModelRequest = {},` | `PublishModelResponse` | 2（pages/ModelVersionsPage.tsx、pages/WorkflowPage.tsx） |
| 19 | `listModels` | GET | `/api/models` | `ListModelsParams` | `ModelVersionSummary[]` | 2（pages/HomePage.tsx、pages/ModelVersionsPage.tsx） |
| 20 | `getModel` | GET | `/api/models/{id}` | `EntityId` | `ModelVersionDetail` | 1（pages/ModelVersionsPage.tsx） |
| 21 | `rollbackModel` | POST | `/api/models/{id}/rollback` | `id: EntityId, input: RollbackModelRequest = {},` | `LifecycleOperationResponse` | 1（pages/ModelVersionsPage.tsx） |
| 22 | `offlineModel` | POST | `/api/models/{id}/offline` | `EntityId` | `LifecycleOperationResponse` | 1（pages/ModelVersionsPage.tsx） |
| 23 | `getModelRollbackRecords` | GET | `/api/models/{id}/rollback-records` | `EntityId` | `RollbackRecord[]` | 1（pages/ModelVersionsPage.tsx） |
| 24 | `getModelPublishRecords` | GET | `/api/models/{id}/publish-records` | `EntityId` | `PublishRecord[]` | 未使用 |
| 25 | `markModelAbnormal` | POST | `/api/models/{id}/abnormal` | `EntityId` | `LifecycleOperationResponse` | 未使用 |
| 26 | `listAlerts` | GET | `/api/alerts` | `ListAlertsParams` | `ModelAlert[]` | 2（pages/HomePage.tsx、pages/ModelVersionsPage.tsx） |
| 27 | `predict` | POST | `/api/mcp/predict` | `PredictionRequest` | `PredictionResponse` | 未使用 |

## 明细

### 1. getHealth `GET /api/health`

- 方法签名行号：client.ts L88
- 返回类型：`HealthResponse`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `RequestOptions`

**出参结构**

<details><summary><code>HealthResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `status` | `'ok'` | 必填 |
| `service`? | `string` | 可选 |
| `database`? | `'ok'` | 可选 |
| `environment`? | `string` | 可选 |
| `app_name`? | `string` | 可选 |

</details>

### 2. uploadDataset `POST (multipart) /api/datasets/upload`

- 方法签名行号：client.ts L92
- 返回类型：`DatasetUploadResult`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参结构**

<details><summary><code>DatasetUploadOptions</code>（1 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `model_type`? | `ModelTypeCode` | 可选 |

</details>

**出参结构**

<details><summary><code>DatasetUploadResult</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `file_name` | `string` | 必填 |
| `row_count` | `number` | 必填 |
| `column_count` | `number` | 必填 |
| `columns` | `DatasetColumn[]` | 必填 |
| `time_column` | `string` | 必填 |
| `feature_columns` | `string[]` | 必填 |
| `target_column` | `string` | 必填 |
| `time_parse` | `TimeColumnParseResult` | 必填 |
| `numeric_columns` | `string[]` | 必填 |
| `missing_values` | `Record<string, MissingValueSummary>` | 必填 |
| `preview_rows` | `DatasetPreviewRow[]` | 必填 |
| `status` | `DatasetStatus` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |

</details>

### 3. createPreprocessingTask `POST /api/preprocessing-tasks`

- 方法签名行号：client.ts L99
- 返回类型：`PreprocessTask`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参（内联对象字段）**

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `model_type` | `ModelTypeCode` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocess_script_id`? | `EntityId | null` | 可选 |
| `mode`? | `'use' | 'skip'` | 可选 |
| `skip`? | `boolean` | 可选 |

**出参结构**

<details><summary><code>PreprocessTask</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `model_type`? | `ModelTypeCode` | 可选 |
| `script_id` | `EntityId | null` | 必填 |
| `preprocess_script_id`? | `EntityId | null` | 可选 |
| `preprocess_used`? | `boolean` | 可选 |
| `preprocess_status`? | `'used' | 'unused'` | 可选 |
| `status` | `` | 必填 |
| `current_stage`? | `PreprocessStage` | 可选 |
| `stage`? | `PreprocessStage` | 可选 |
| `progress_stage`? | `PreprocessStage` | 可选 |
| `stages`? | `StageProgress<PreprocessStage>[]` | 可选 |
| `summary`? | `PreprocessResultSummary | null` | 可选 |
| `input_row_count`? | `number | null` | 可选 |
| `output_row_count`? | `number | null` | 可选 |
| `input_columns`? | `string[]` | 可选 |
| `output_columns`? | `string[]` | 可选 |
| `input_summary`? | `Record<string, JsonValue>` | 可选 |
| `output_summary`? | `Record<string, JsonValue>` | 可选 |
| `logs` | `Array<TrainingLogEntry | string>` | 必填 |
| `error_message`? | `string | null` | 可选 |
| `created_at` | `IsoDateTime` | 必填 |
| `started_at`? | `IsoDateTime | null` | 可选 |
| `finished_at`? | `IsoDateTime | null` | 可选 |
| `next_step`? | `'dataset_split' | null` | 可选 |
| `data_source`? | `'raw' | 'preprocessed'` | 可选 |

</details>

### 4. getPreprocessingTask `GET /api/preprocessing-tasks/{id}`

- 方法签名行号：client.ts L110
- 返回类型：`PreprocessTask`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `EntityId`

**出参结构**

<details><summary><code>PreprocessTask</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `model_type`? | `ModelTypeCode` | 可选 |
| `script_id` | `EntityId | null` | 必填 |
| `preprocess_script_id`? | `EntityId | null` | 可选 |
| `preprocess_used`? | `boolean` | 可选 |
| `preprocess_status`? | `'used' | 'unused'` | 可选 |
| `status` | `` | 必填 |
| `current_stage`? | `PreprocessStage` | 可选 |
| `stage`? | `PreprocessStage` | 可选 |
| `progress_stage`? | `PreprocessStage` | 可选 |
| `stages`? | `StageProgress<PreprocessStage>[]` | 可选 |
| `summary`? | `PreprocessResultSummary | null` | 可选 |
| `input_row_count`? | `number | null` | 可选 |
| `output_row_count`? | `number | null` | 可选 |
| `input_columns`? | `string[]` | 可选 |
| `output_columns`? | `string[]` | 可选 |
| `input_summary`? | `Record<string, JsonValue>` | 可选 |
| `output_summary`? | `Record<string, JsonValue>` | 可选 |
| `logs` | `Array<TrainingLogEntry | string>` | 必填 |
| `error_message`? | `string | null` | 可选 |
| `created_at` | `IsoDateTime` | 必填 |
| `started_at`? | `IsoDateTime | null` | 可选 |
| `finished_at`? | `IsoDateTime | null` | 可选 |
| `next_step`? | `'dataset_split' | null` | 可选 |
| `data_source`? | `'raw' | 'preprocessed'` | 可选 |

</details>

### 5. splitDataset `POST /api/datasets/{datasetId}/split`

- 方法签名行号：client.ts L114
- 返回类型：`DatasetSplitResult`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参：** `EntityId`

**出参结构**

<details><summary><code>DatasetSplitResult</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocessing_task_id` | `EntityId | null` | 必填 |
| `data_source` | `'raw' | 'preprocessed'` | 必填 |
| `split_strategy` | `'time_ordered'` | 必填 |
| `split_ratio` | `0.8` | 必填 |
| `test_ratio` | `0.2` | 必填 |
| `total_row_count` | `number` | 必填 |
| `train_row_count` | `number` | 必填 |
| `test_row_count` | `number` | 必填 |
| `train_time_range` | `{ start: IsoDateTime; end: IsoDateTime }` | 必填 |
| `test_time_range` | `{ start: IsoDateTime; end: IsoDateTime }` | 必填 |
| `train_time_start` | `IsoDateTime` | 必填 |
| `train_time_end` | `IsoDateTime` | 必填 |
| `test_time_start` | `IsoDateTime` | 必填 |
| `test_time_end` | `IsoDateTime` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |

</details>

### 6. getDatasetSplit `GET /api/datasets/{datasetId}/split`

- 方法签名行号：client.ts L119
- 返回类型：`DatasetSplitResult`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参：** `EntityId`

**出参结构**

<details><summary><code>DatasetSplitResult</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocessing_task_id` | `EntityId | null` | 必填 |
| `data_source` | `'raw' | 'preprocessed'` | 必填 |
| `split_strategy` | `'time_ordered'` | 必填 |
| `split_ratio` | `0.8` | 必填 |
| `test_ratio` | `0.2` | 必填 |
| `total_row_count` | `number` | 必填 |
| `train_row_count` | `number` | 必填 |
| `test_row_count` | `number` | 必填 |
| `train_time_range` | `{ start: IsoDateTime; end: IsoDateTime }` | 必填 |
| `test_time_range` | `{ start: IsoDateTime; end: IsoDateTime }` | 必填 |
| `train_time_start` | `IsoDateTime` | 必填 |
| `train_time_end` | `IsoDateTime` | 必填 |
| `test_time_start` | `IsoDateTime` | 必填 |
| `test_time_end` | `IsoDateTime` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |

</details>

### 7. listScripts `GET /api/scripts`

- 方法签名行号：client.ts L123
- 返回类型：`PaginatedResponse<ScriptContract>`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参：** `ListScriptsParams`

### 8. getScript `GET /api/scripts/{id}`

- 方法签名行号：client.ts L129
- 返回类型：`ScriptContract`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `EntityId`

**出参结构**

<details><summary><code>ScriptContract</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `name` | `string` | 必填 |
| `script_type` | `ScriptType` | 必填 |
| `version` | `string` | 必填 |
| `source_code`? | `string` | 可选 |
| `supported_model_types` | `ModelTypeCode[]` | 必填 |
| `status` | `ScriptStatus` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |
| `updated_at`? | `IsoDateTime` | 可选 |

</details>

### 9. enableScript `POST /api/scripts/{id}/enable`

- 方法签名行号：client.ts L133
- 返回类型：`ScriptContract`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `EntityId`

**出参结构**

<details><summary><code>ScriptContract</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `name` | `string` | 必填 |
| `script_type` | `ScriptType` | 必填 |
| `version` | `string` | 必填 |
| `source_code`? | `string` | 可选 |
| `supported_model_types` | `ModelTypeCode[]` | 必填 |
| `status` | `ScriptStatus` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |
| `updated_at`? | `IsoDateTime` | 可选 |

</details>

### 10. disableScript `POST /api/scripts/{id}/disable`

- 方法签名行号：client.ts L137
- 返回类型：`ScriptContract`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `EntityId`

**出参结构**

<details><summary><code>ScriptContract</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `name` | `string` | 必填 |
| `script_type` | `ScriptType` | 必填 |
| `version` | `string` | 必填 |
| `source_code`? | `string` | 可选 |
| `supported_model_types` | `ModelTypeCode[]` | 必填 |
| `status` | `ScriptStatus` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |
| `updated_at`? | `IsoDateTime` | 可选 |

</details>

### 11. uploadScript `POST (multipart) /api/scripts/upload`

- 方法签名行号：client.ts L141
- 返回类型：`ScriptContract`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参结构**

<details><summary><code>ScriptUploadInput</code>（5 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `file` | `File` | 必填 |
| `name` | `string` | 必填 |
| `script_type` | `ScriptType` | 必填 |
| `supported_model_types` | `ModelTypeCode[]` | 必填 |
| `version`? | `string` | 可选 |

</details>

**出参结构**

<details><summary><code>ScriptContract</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `name` | `string` | 必填 |
| `script_type` | `ScriptType` | 必填 |
| `version` | `string` | 必填 |
| `source_code`? | `string` | 可选 |
| `supported_model_types` | `ModelTypeCode[]` | 必填 |
| `status` | `ScriptStatus` | 必填 |
| `created_at` | `IsoDateTime` | 必填 |
| `updated_at`? | `IsoDateTime` | 可选 |

</details>

### 12. createTrainingJob `POST /api/training-jobs`

- 方法签名行号：client.ts L151
- 返回类型：`TrainingJob`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参结构**

<details><summary><code>CreateTrainingJobRequest</code>（6 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `model_type` | `ModelTypeCode` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocess_script_id` | `EntityId | null` | 必填 |
| `preprocessing_task_id`? | `EntityId | null` | 可选 |
| `train_script_id` | `EntityId` | 必填 |
| `config`? | `Record<string, JsonValue>` | 可选 |

</details>

**出参结构**

<details><summary><code>TrainingJob</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `model_type` | `ModelTypeCode` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocess_script_id` | `EntityId | null` | 必填 |
| `preprocessing_task_id`? | `EntityId | null` | 可选 |
| `train_script_id` | `EntityId` | 必填 |
| `split_strategy`? | `'time_ordered'` | 可选 |
| `split_ratio` | `0.8` | 必填 |
| `test_ratio`? | `0.2` | 可选 |
| `status` | `TrainingJobStatus` | 必填 |
| `progress_stage`? | `TrainingJobStage | string | null` | 可选 |
| `current_stage`? | `TrainingJobStage | string | null` | 可选 |
| `stage`? | `string | null` | 可选 |
| `progress`? | `StageProgress<TrainingJobStage> | null` | 可选 |
| `stages`? | `StageProgress<TrainingJobStage>[]` | 可选 |
| `logs` | `Array<TrainingLogEntry | string>` | 必填 |
| `error_message` | `string | null` | 必填 |
| `dataset_split`? | `DatasetSplitSummary | null` | 可选 |
| `train_row_count`? | `number | null` | 可选 |
| `test_row_count`? | `number | null` | 可选 |
| `train_time_start`? | `IsoDateTime | null` | 可选 |
| `train_time_end`? | `IsoDateTime | null` | 可选 |
| `test_time_start`? | `IsoDateTime | null` | 可选 |
| `test_time_end`? | `IsoDateTime | null` | 可选 |
| `model_version_id`? | `EntityId | null` | 可选 |
| `created_at` | `IsoDateTime` | 必填 |
| `started_at`? | `IsoDateTime | null` | 可选 |
| `finished_at` | `IsoDateTime | null` | 必填 |

</details>

### 13. retryTrainingJob `POST /api/training-jobs/{id}/retry`

- 方法签名行号：client.ts L155
- 返回类型：`TrainingJob`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参：** `EntityId`

**出参结构**

<details><summary><code>TrainingJob</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `model_type` | `ModelTypeCode` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocess_script_id` | `EntityId | null` | 必填 |
| `preprocessing_task_id`? | `EntityId | null` | 可选 |
| `train_script_id` | `EntityId` | 必填 |
| `split_strategy`? | `'time_ordered'` | 可选 |
| `split_ratio` | `0.8` | 必填 |
| `test_ratio`? | `0.2` | 可选 |
| `status` | `TrainingJobStatus` | 必填 |
| `progress_stage`? | `TrainingJobStage | string | null` | 可选 |
| `current_stage`? | `TrainingJobStage | string | null` | 可选 |
| `stage`? | `string | null` | 可选 |
| `progress`? | `StageProgress<TrainingJobStage> | null` | 可选 |
| `stages`? | `StageProgress<TrainingJobStage>[]` | 可选 |
| `logs` | `Array<TrainingLogEntry | string>` | 必填 |
| `error_message` | `string | null` | 必填 |
| `dataset_split`? | `DatasetSplitSummary | null` | 可选 |
| `train_row_count`? | `number | null` | 可选 |
| `test_row_count`? | `number | null` | 可选 |
| `train_time_start`? | `IsoDateTime | null` | 可选 |
| `train_time_end`? | `IsoDateTime | null` | 可选 |
| `test_time_start`? | `IsoDateTime | null` | 可选 |
| `test_time_end`? | `IsoDateTime | null` | 可选 |
| `model_version_id`? | `EntityId | null` | 可选 |
| `created_at` | `IsoDateTime` | 必填 |
| `started_at`? | `IsoDateTime | null` | 可选 |
| `finished_at` | `IsoDateTime | null` | 必填 |

</details>

### 14. getTrainingJob `GET /api/training-jobs/{id}`

- 方法签名行号：client.ts L159
- 返回类型：`TrainingJob`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参：** `EntityId`

**出参结构**

<details><summary><code>TrainingJob</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `model_type` | `ModelTypeCode` | 必填 |
| `dataset_id` | `EntityId` | 必填 |
| `preprocess_script_id` | `EntityId | null` | 必填 |
| `preprocessing_task_id`? | `EntityId | null` | 可选 |
| `train_script_id` | `EntityId` | 必填 |
| `split_strategy`? | `'time_ordered'` | 可选 |
| `split_ratio` | `0.8` | 必填 |
| `test_ratio`? | `0.2` | 可选 |
| `status` | `TrainingJobStatus` | 必填 |
| `progress_stage`? | `TrainingJobStage | string | null` | 可选 |
| `current_stage`? | `TrainingJobStage | string | null` | 可选 |
| `stage`? | `string | null` | 可选 |
| `progress`? | `StageProgress<TrainingJobStage> | null` | 可选 |
| `stages`? | `StageProgress<TrainingJobStage>[]` | 可选 |
| `logs` | `Array<TrainingLogEntry | string>` | 必填 |
| `error_message` | `string | null` | 必填 |
| `dataset_split`? | `DatasetSplitSummary | null` | 可选 |
| `train_row_count`? | `number | null` | 可选 |
| `test_row_count`? | `number | null` | 可选 |
| `train_time_start`? | `IsoDateTime | null` | 可选 |
| `train_time_end`? | `IsoDateTime | null` | 可选 |
| `test_time_start`? | `IsoDateTime | null` | 可选 |
| `test_time_end`? | `IsoDateTime | null` | 可选 |
| `model_version_id`? | `EntityId | null` | 可选 |
| `created_at` | `IsoDateTime` | 必填 |
| `started_at`? | `IsoDateTime | null` | 可选 |
| `finished_at` | `IsoDateTime | null` | 必填 |

</details>

### 15. getTrainingJobLogs `GET /api/training-jobs/{id}/logs`

- 方法签名行号：client.ts L163
- 返回类型：`TrainingLogsResponse`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `id: EntityId, params?: { since?: string; limit?: number },`

**出参结构**

<details><summary><code>TrainingLogsResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `job_id` | `EntityId` | 必填 |
| `items` | `Array<TrainingLogEntry | string>` | 必填 |
| `next_cursor`? | `string | null` | 可选 |

</details>

### 16. getTrainingJobEvaluation `GET /api/training-jobs/{id}/evaluation`

- 方法签名行号：client.ts L172
- 返回类型：`ModelEvaluation`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参：** `EntityId`

**出参结构**

<details><summary><code>ModelEvaluation</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id`? | `EntityId` | 可选 |
| `job_id`? | `EntityId` | 可选 |
| `model_version_id`? | `EntityId` | 可选 |
| `candidate`? | `EvaluationMetrics` | 可选 |
| `baseline`? | `EvaluationMetrics` | 可选 |
| `metrics`? | `EvaluationMetrics | Record<string, JsonValue>` | 可选 |
| `comparison`? | `MetricComparison[] | Record<string, JsonValue>` | 可选 |
| `chart_data`? | `EvaluationChartData | Array<Record<string, JsonValue>>` | 可选 |
| `error_data`? | `Array<Record<string, JsonValue>>` | 可选 |
| `chart_sampled`? | `boolean` | 可选 |
| `chart_total_count`? | `number` | 可选 |
| `chart_sample_count`? | `number` | 可选 |
| `model_comparison`? | `Record<string, JsonValue>` | 可选 |
| `created_at`? | `IsoDateTime` | 可选 |

</details>

### 17. saveModel `POST /api/models/{id}/save`

- 方法签名行号：client.ts L176
- 返回类型：`ModelVersionSummary`
- 页面使用：1 处（pages/WorkflowPage.tsx）

**入参结构**

<details><summary><code>ModelSaveRequest</code>（11 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `model_type` | `ModelTypeCode` | 必填 |
| `status`? | `'DRAFT' | 'READY'` | 可选 |
| `training_job_id`? | `EntityId` | 可选 |
| `train_script_id`? | `EntityId` | 可选 |
| `preprocess_script_id`? | `EntityId | null` | 可选 |
| `preprocess_used`? | `boolean` | 可选 |
| `input_schema`? | `Record<string, JsonValue>` | 可选 |
| `time_column`? | `string` | 可选 |
| `feature_columns`? | `string[]` | 可选 |
| `target_column`? | `string` | 可选 |
| `metrics`? | `Record<string, JsonValue>` | 可选 |

</details>

**出参结构**

<details><summary><code>ModelVersionSummary</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `id` | `EntityId` | 必填 |
| `model_type` | `ModelTypeCode` | 必填 |
| `version` | `string` | 必填 |
| `status` | `ModelVersionStatus` | 必填 |
| `health_status`? | `HealthStatus | string` | 可选 |
| `is_baseline` | `boolean` | 必填 |
| `is_current` | `boolean` | 必填 |
| `is_abnormal`? | `boolean` | 可选 |
| `is_rollback_available`? | `boolean` | 可选 |
| `metrics` | `EvaluationMetrics | Record<string, JsonValue> | null` | 必填 |
| `preprocess_used`? | `boolean` | 可选 |
| `model_path`? | `string | null` | 可选 |
| `preprocessor_path`? | `string | null` | 可选 |
| `training_job_id`? | `EntityId | null` | 可选 |
| `train_script_id`? | `EntityId | null` | 可选 |
| `train_script_version`? | `string | null` | 可选 |
| `preprocess_script_id`? | `EntityId | null` | 可选 |
| `preprocess_script_version`? | `string | null` | 可选 |
| `previous_healthy_version_id`? | `EntityId | null` | 可选 |
| `train_script`? | `Pick<ScriptContract, 'id' | 'name' | 'version'> | null` | 可选 |
| `preprocess_script`? | `Pick<ScriptContract, 'id' | 'name' | 'version'> | null` | 可选 |
| `input_schema`? | `InputSchema | Record<string, JsonValue>` | 可选 |
| `feature_columns`? | `string[]` | 可选 |
| `time_column`? | `string | null` | 可选 |
| `target_column`? | `string | null` | 可选 |
| `created_at` | `IsoDateTime` | 必填 |
| `published_at` | `IsoDateTime | null` | 必填 |

</details>

### 18. publishModel `POST /api/models/{id}/publish`

- 方法签名行号：client.ts L181
- 返回类型：`PublishModelResponse`
- 页面使用：2 处（pages/ModelVersionsPage.tsx、pages/WorkflowPage.tsx）

**入参结构**

<details><summary><code>PublishModelRequest</code>（3 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `message`? | `string` | 可选 |
| `confirm`? | `boolean` | 可选 |
| `confirmed`? | `boolean` | 可选 |

</details>

**出参结构**

<details><summary><code>PublishModelResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `model` | `ModelVersionDetail | ModelVersionSummary` | 必填 |
| `record`? | `PublishRecord` | 可选 |
| `operation`? | `string` | 可选 |

</details>

### 19. listModels `GET /api/models`

- 方法签名行号：client.ts L196
- 返回类型：`ModelVersionSummary[]`
- 页面使用：2 处（pages/HomePage.tsx、pages/ModelVersionsPage.tsx）

**入参：** `ListModelsParams`

### 20. getModel `GET /api/models/{id}`

- 方法签名行号：client.ts L207
- 返回类型：`ModelVersionDetail`
- 页面使用：1 处（pages/ModelVersionsPage.tsx）

**入参：** `EntityId`

### 21. rollbackModel `POST /api/models/{id}/rollback`

- 方法签名行号：client.ts L216
- 返回类型：`LifecycleOperationResponse`
- 页面使用：1 处（pages/ModelVersionsPage.tsx）

**入参结构**

<details><summary><code>RollbackModelRequest</code>（2 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `target_version_id`? | `EntityId` | 可选 |
| `reason`? | `string` | 可选 |

</details>

**出参结构**

<details><summary><code>LifecycleOperationResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `operation` | `string` | 必填 |
| `model` | `ModelVersionDetail | ModelVersionSummary` | 必填 |
| `rollback`? | `RollbackRecord | null` | 可选 |
| `alert`? | `ModelAlert | null` | 可选 |

</details>

### 22. offlineModel `POST /api/models/{id}/offline`

- 方法签名行号：client.ts L227
- 返回类型：`LifecycleOperationResponse`
- 页面使用：1 处（pages/ModelVersionsPage.tsx）

**入参：** `EntityId`

**出参结构**

<details><summary><code>LifecycleOperationResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `operation` | `string` | 必填 |
| `model` | `ModelVersionDetail | ModelVersionSummary` | 必填 |
| `rollback`? | `RollbackRecord | null` | 可选 |
| `alert`? | `ModelAlert | null` | 可选 |

</details>

### 23. getModelRollbackRecords `GET /api/models/{id}/rollback-records`

- 方法签名行号：client.ts L232
- 返回类型：`RollbackRecord[]`
- 页面使用：1 处（pages/ModelVersionsPage.tsx）

**入参：** `EntityId`

### 24. getModelPublishRecords `GET /api/models/{id}/publish-records`

- 方法签名行号：client.ts L239
- 返回类型：`PublishRecord[]`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `EntityId`

### 25. markModelAbnormal `POST /api/models/{id}/abnormal`

- 方法签名行号：client.ts L247
- 返回类型：`LifecycleOperationResponse`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参：** `EntityId`

**出参结构**

<details><summary><code>LifecycleOperationResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `operation` | `string` | 必填 |
| `model` | `ModelVersionDetail | ModelVersionSummary` | 必填 |
| `rollback`? | `RollbackRecord | null` | 可选 |
| `alert`? | `ModelAlert | null` | 可选 |

</details>

### 26. listAlerts `GET /api/alerts`

- 方法签名行号：client.ts L252
- 返回类型：`ModelAlert[]`
- 页面使用：2 处（pages/HomePage.tsx、pages/ModelVersionsPage.tsx）

**入参：** `ListAlertsParams`

### 27. predict `POST /api/mcp/predict`

- 方法签名行号：client.ts L263
- 返回类型：`PredictionResponse`
- 页面使用：未使用（页面未调用，可能为预留接口）

**入参结构**

<details><summary><code>PredictionRequest</code>（3 行）</summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `model_type` | `ModelTypeCode` | 必填 |
| `model_version`? | `string` | 可选 |
| `data` | `JsonObject[]` | 必填 |

</details>

**出参结构**

<details><summary><code>PredictionResponse</code></summary>

| 字段 | 类型 | 说明/可选 |
|---|---|---|
| `success` | `true` | 必填 |
| `model_type` | `ModelTypeCode` | 必填 |
| `model_version` | `string` | 必填 |
| `preprocess_used` | `boolean` | 必填 |
| `predictions` | `number[]` | 必填 |

</details>

## 零调用接口（页面未使用）

| 方法 | HTTP | 路径 | 返回类型 |
|---|---|---|---|
| `getHealth` | GET | `/api/health` | `HealthResponse` |
| `getPreprocessingTask` | GET | `/api/preprocessing-tasks/{id}` | `PreprocessTask` |
| `getScript` | GET | `/api/scripts/{id}` | `ScriptContract` |
| `enableScript` | POST | `/api/scripts/{id}/enable` | `ScriptContract` |
| `disableScript` | POST | `/api/scripts/{id}/disable` | `ScriptContract` |
| `uploadScript` | POST (multipart) | `/api/scripts/upload` | `ScriptContract` |
| `getTrainingJobLogs` | GET | `/api/training-jobs/{id}/logs` | `TrainingLogsResponse` |
| `getModelPublishRecords` | GET | `/api/models/{id}/publish-records` | `PublishRecord[]` |
| `markModelAbnormal` | POST | `/api/models/{id}/abnormal` | `LifecycleOperationResponse` |
| `predict` | POST | `/api/mcp/predict` | `PredictionResponse` |
