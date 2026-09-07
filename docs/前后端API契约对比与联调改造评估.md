# 前后端 API 契约对比与联调改造评估

> 对比日期：2026-09-05  
> 对比范围：前端 API Client/TypeScript 契约、前端需求文档、后端 README/API 文档、FastAPI 路由与 Pydantic Schema。  
> 本报告仅记录分析结果和改造评估，不包含代码修改。

## 1. 总体结论

当前前后端并非整体不可联调，主训练流程已经基本贯通；但存在两个明确的路径级错配，其中预测接口是高优先级运行时故障：

1. 前端声明 `GET /api/model-types`，后端明确没有该路由。
2. 前端预测调用 `/api/predict`，后端实际提供 `/api/mcp/predict`。
3. MCP 页面仍显示“待接入”，但后端 MCP 路由已经实现，前端文案和 Client 均落后于后端。
4. 分页、日志增量查询、预处理状态枚举等存在“前端声明支持、后端未实现或大小写不一致”的问题。

**结论：最小联调修复难度低，完整契约收敛难度中等。**

---

## 2. 对比统计

统计口径：以 `frontend/src/api/client.ts` 中的业务 API 方法为前端接口面，以 `backend/README.md`、后端路由实现和公开后端需求文档为后端接口面；健康检查的 `/api/health` 兼容路径计入匹配。

| 项目 | 数量 | 说明 |
|---|---:|---|
| 前端 API Client 业务接口 | 28 | 不含通用 `get/post` 封装方法 |
| 路径和 HTTP 方法直接匹配 | 26 | 约 92.9% 的前端接口面 |
| 明确路径错配 | 2 | `model-types`、`predict` |
| 后端已有但前端未封装 | 9 | 取消任务、MCP、告警详情等 |
| 参数语义差异 | 4 组 | 参数会被后端忽略或没有对应实现 |
| 前端类型与后端响应差异 | 3 处 | 预处理状态、失败阶段、异常响应结构 |
| 文档相互矛盾 | 2 类 | 模型类型接口、MCP 接入状态 |

---

## 3. 详细错配清单

### 3.1 P0：预测路径错配

#### 前端

文件：`frontend/src/api/client.ts`

```ts
predict(input: PredictionRequest): Promise<PredictionResponse> {
  return this.postJson<PredictionResponse>('predict', input)
}
```

API 基址默认为 `/api`，因此实际请求：

```http
POST /api/predict
```

#### 后端

后端实际接口：

```http
POST /api/mcp/predict
```

实现：`backend/app/mcp_router.py`

后端还提供兼容路径：

```http
POST /mcp/predict
```

但没有 `/api/predict`。

#### 影响

调用 `apiClient.predict()` 会直接得到 404。该问题属于真实运行时阻断。

#### 推荐改法

优先修改前端：

```ts
return this.postJson<PredictionResponse>('mcp/predict', input)
```

改动范围小，不需要改变后端。

---

### 3.2 P1：模型类型接口不存在

#### 前端

```ts
listModelTypes(): Promise<ModelTypeContract[]> {
  return this.get<ModelTypeContract[]>('model-types')
}
```

前端需求文档也声明了：

```http
GET /api/model-types
```

#### 后端

`backend/README.md` 明确说明当前版本没有：

```http
GET /api/model-types
```

模型类型由初始化逻辑固定提供：

```text
electric_load
heating_cooling_load
integrated_energy
```

#### 当前影响

当前 UI 使用 `MODEL_TYPE_CODES` 本地常量，没有调用该接口，因此暂未阻断页面。

#### 推荐改法

二选一：

- 保持三种模型固定：删除或废弃前端 `listModelTypes()`，同步删除需求文档中的该依赖。
- 需要动态模型类型：后端新增 `/api/model-types`，同时补充响应 Schema 和测试。

当前项目更适合第一种，改造成本更低。

---

### 3.3 P1：MCP 页面和后端实现状态相反

文件：`frontend/src/pages/McpPage.tsx`

当前文案包含：

- `MCP 路由待接入`
- `当前代码未提供独立的 MCP predict`
- `独立 MCP 适配层尚未实现`

但后端已经实现：

```http
POST /api/mcp/predict
POST /api/mcp/mark_model_abnormal
```

后端文档也已经将其作为正式 HTTP transport 描述。

#### 影响

- 页面向联调人员传达错误能力边界。
- MCP 地址显示为“未提供”。
- 前端没有封装 `/api/mcp/mark_model_abnormal`。
- `predict()` 还调用了错误路径。

#### 推荐改法

- 将页面状态改为“已提供 HTTP MCP transport”。
- 展示 `/api/mcp/predict` 和 `/api/mcp/mark_model_abnormal`。
- 前端增加对应 Client 方法。
- 保留当前请求示例和错误码说明，但删除“后端未实现”的旧说明。

---

### 3.4 P1：预处理状态大小写不一致

后端返回状态：

```text
WAITING / RUNNING / SUCCEEDED / SKIPPED / FAILED
```

前端 `PreprocessTask.status` 主要声明小写状态：

```text
waiting / running / succeeded / skipped / failed
```

前端页面目前通过 `.toLowerCase()` 规避了大部分运行时问题，但 TypeScript 契约不准确。

此外，后端阶段枚举包含：

```text
failed
```

前端 `PreprocessStage` 未声明 `failed`。

#### 推荐改法

前端类型改为明确包含后端枚举值；或者在 API Client 归一化为统一小写值。建议统一在 Client 层归一化，页面只消费一种格式。

---

### 3.5 P1：异常标记响应类型不一致

前端定义的旧类型：

```ts
interface MarkModelAbnormalResponse {
  alert: ModelAlert | null
  current_model: ModelVersionSummary
  rollback: RollbackRecord | null
}
```

后端 `/api/models/abnormal` 实际使用 `LifecycleOperationResponse`，核心返回：

```json
{
  "operation": "abnormal",
  "model": {},
  "rollback": {},
  "alert": {}
}
```

没有前端类型声明的 `current_model` 字段。

当前没有 Client 方法使用这个旧类型，因此属于潜在错配；但如果后续接入按模型类型的异常标记，会出现读取字段错误。

---

### 3.6 P2：查询参数被后端静默忽略

#### 数据集上传

前端在 multipart 中发送 `model_type`，后端上传接口只声明 `file`，该字段不会被保存或参与校验。

#### 训练日志

前端支持：

```text
since
limit
```

后端日志接口只接收 `job_id`，因此增量日志和限制条数不会生效。

#### 模型列表

前端声明：

```text
page
page_size
model_type
status
```

后端支持：

```text
model_type
status
health_status
```

`page`、`page_size` 会被忽略；后端返回数组，不返回分页对象。

#### 告警列表

前端声明：

```text
page
page_size
model_type
status
active_only
```

后端支持：

```text
model_type
active_only
```

`page`、`page_size`、`status` 会被忽略。

---

## 4. 已匹配良好的部分

以下部分目前基本可以直接联调：

- CSV 上传成功响应的核心字段。
- 数据集固定时间升序 80/20 划分。
- 预处理任务创建和查询。
- 训练任务创建、轮询、重试、评估。
- 模型保存草稿、发布、下线、回滚。
- 模型列表和详情。
- 告警列表。
- 发布确认字段：`confirm`、`confirmed` 均被后端兼容。
- 评估中的 `time/timestamp`、`predicted/candidate_prediction` 兼容处理。
- 统一错误响应 `success/error_code/message/details`。

主训练流程的请求顺序目前是可用的：

```text
上传数据
→ 创建预处理任务
→ 创建数据集划分
→ 创建训练任务
→ 轮询训练状态
→ 获取评估
→ 保存模型
→ 发布模型
```

---

## 5. 后端已有但前端未封装的接口

这些不一定都是缺陷，但属于前端 Client 覆盖不完整：

1. `POST /api/preprocessing-tasks/{id}/execute`
2. `POST /api/preprocessing-tasks/{id}/transform`
3. `POST /api/training-jobs/{id}/cancel`
4. `POST /api/models`
5. `POST /api/models/abnormal`
6. `GET /api/alerts/{id}`
7. `POST /api/alerts/{id}/acknowledge`
8. `POST /api/mcp/predict`
9. `POST /api/mcp/mark_model_abnormal`

---

## 6. 联调改造难度评估

### 6.1 最小可联调修复

目标：让当前前端主流程和预测调用可正确联调，不扩展新能力。

| 工作项 | 预计难度 | 预计工作量 | 说明 |
|---|---|---:|---|
| 修正 `predict` 路径 | 低 | 0.1～0.25 人日 | 前端单处路径修改，加请求测试 |
| 清理模型类型接口依赖 | 低 | 0.25～0.5 人日 | 保留本地常量，废弃未实现接口 |
| 修正 MCP 页面过期文案 | 低 | 0.25～0.5 人日 | 更新地址、状态和调用说明 |
| 修正预处理状态类型 | 低 | 0.25～0.5 人日 | 类型和归一化逻辑调整 |
| 增加基础联调回归测试 | 中 | 0.5～1 人日 | 覆盖预测、发布、训练主流程 |
| **合计** | **低** | **约 1.5～2.75 人日** | 不新增后端能力 |

### 6.2 完整契约收敛

目标：前端契约、后端实现、需求文档和接口行为全部统一。

| 工作项 | 预计难度 | 预计工作量 | 说明 |
|---|---|---:|---|
| 完成 MCP Client 封装 | 中 | 0.5～1 人日 | predict、mark abnormal、错误码处理 |
| 统一预处理状态模型 | 低 | 0.25～0.5 人日 | 后端大写和页面小写统一 |
| 统一异常响应结构 | 低 | 0.25～0.5 人日 | 删除旧 `current_model` 兼容类型或增加适配器 |
| 明确列表分页方案 | 中 | 1～2 人日 | 需要决定前后端是否真正实现分页 |
| 日志增量查询 | 中 | 0.5～1.5 人日 | 后端增加 since/limit 逻辑和测试 |
| 数据集与模型类型关联 | 中 | 0.5～1 人日 | 后端保存字段、Schema、迁移和测试 |
| 更新前后端需求文档 | 低 | 0.5～1 人日 | 同步实际路径和能力边界 |
| 完整 API 回归测试 | 中 | 1～2 人日 | 正常、异常、状态冲突、响应 Schema |
| **合计** | **中等** | **约 4.5～9.5 人日** | 取决于是否实现分页和增量日志 |

### 6.3 如果选择后端兼容别名方案

也可以只在后端增加：

```http
POST /api/predict
```

转发到现有 MCP predict 服务。

这种方式能快速兼容现有前端，但不建议作为唯一方案，因为：

- 前端 MCP 页面仍然会继续显示错误状态。
- `/api/predict` 和 `/api/mcp/predict` 会形成两个概念不清的入口。
- 后续其他 MCP 工具仍然需要前端单独接入。

后端增加别名的工作量约为 **0.25～0.75 人日**，但完整修复仍建议同时调整前端路径和文案。

---

## 7. 推荐实施顺序

### 第一阶段：先恢复联调

1. 修正 `ApiClient.predict()` 为 `/mcp/predict`。
2. 用后端现有 MCP 示例验证成功响应和错误响应。
3. 修正 MCP 页面“待接入”文案。
4. 确认当前页面没有调用 `/api/model-types`。

### 第二阶段：统一类型和查询契约

5. 统一预处理状态和阶段枚举。
6. 处理 `MarkModelAbnormalResponse` 旧类型。
7. 明确模型列表、告警列表是否需要分页。
8. 删除未实现的日志 `since/limit`，或者补充后端实现。

### 第三阶段：补齐能力和文档

9. 增加 MCP 异常标记 Client 方法。
10. 增加取消训练、告警详情和告警确认方法（如果页面需要）。
11. 更新前端接口依赖文档。
12. 增加前后端 API 回归测试或从 OpenAPI 生成契约测试。

---

## 8. 最终判断

当前项目属于：

> **主业务流程基本可联调，边缘能力和 MCP 入口存在契约漂移。**

不是大规模重构问题。若只解决当前实际阻断，预计 **1.5～2.75 人日**；若要把接口、类型、分页、日志、MCP 和文档全部收敛，预计 **4.5～9.5 人日**。

最优先的单点修复是：

```text
frontend/src/api/client.ts
predict(): 'predict' → 'mcp/predict'
```

