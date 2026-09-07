# Implement 运行索引

本文件由 implement 扩展自动维护：每个步骤验证通过并推送后追加；subagent 应先读它定位改动范围，避免重读整个仓库。
- 运行：2026-09-07T14:19:10.219Z
- 工作目录：C:\Users\62001\Documents\Trae\模型训练
- session：01a07c0c-a0b5-7282-8d5f-3bea9881b36e
<!-- implement-run:2026-09-07T14-19-10-219Z -->

## 计划步骤
1. 以 产品原型.html、统一产品设计与最小改造方案.md、backend/app 和 frontend/src 为基线，建立页面字段、接口、状态枚举和资源 ID 链路差异清单。
2. 冻结统一合同模型：规范 model_type、资源 ID、READY/PUBLISHED/RETIRED/FAILED 生命周期、HEALTHY/ABNORMAL/UNKNOWN 健康状态及 ACTIVE/ACKNOWLEDGED/RESOLVED 告警状态。
3. 在 backend/app/db/models.py 增加 append-only 审计事件表、索引和关联字段，并保证 SQLite 初始化可兼容已有数据库。
4. 新增审计 schema、repository/service 和 GET /api/audit-events，支持分页、时间、模型类型、事件类型、对象类型、结果及关键词筛选。
5. 在数据上传、校验、预处理、跳过、划分、训练、评估、保存、发布、下线、异常、回滚和告警确认/解决的服务端成功与失败出口写入审计事件。
6. 补齐后端合同：训练日志支持 since/limit/next_cursor，发布支持 confirmed/reason/idempotency_key，回滚强制 target_version_id/reason/idempotency_key，告警查询和确认返回规范状态与统计信息。 [verify: python:pytest]
7. 修改 frontend/src/types/contracts.ts，统一空值、可选字段、状态枚举、分页响应、评估结果、告警和审计类型。
8. 新增 frontend/src/api/normalizers.ts，映射上传、预处理、训练、评估、模型和兼容字段；缺失值统一显示“未知”或“未提供”，不默认健康。
9. 修改 frontend/src/api/client.ts，移除上传请求中的无效字段，接入预处理查询、训练日志增量、取消训练、异常标记、告警确认、审计查询，并规范发布、回滚请求体和幂等键。
10. 修改 frontend/src/store/useAppStore.ts 与 frontend/src/pages/WorkflowPage.tsx，按资源 ID 持久化和恢复训练上下文；切换模型类型时清理下游资源，并落实七步流程的前置门禁。
11. 按原型重构训练页面字段和交互：接入脚本选择/上传、预处理跳过与结果、固定 80/20 划分、训练轮询与日志、取消/重试、SUCCEEDED + model_version_id 门禁、评估空态及 READY 保存。
12. 修改 frontend/src/pages/HomePage.tsx，按 model_type 去重统计生产版本，区分健康/未知，统计 READY 候选和 ACTIVE 告警，补充训练完成时间二次查询及告警确认，删除静态描述和固定结论。
13. 修改 frontend/src/pages/ModelVersionsPage.tsx，按原型收敛版本字段，分离生命周期与健康状态，实现发布、下线、异常和显式目标回滚门禁、原因输入及操作后的模型/告警/审计刷新。
14. 新增 frontend/src/pages/AuditPage.tsx，在 frontend/src/App.tsx 和 frontend/src/components/layout/AppLayout.tsx 注册 /audit 路由与导航，实现原型五列表格、事件筛选、后端 total 分页、未知操作人和对象链接；不提供编辑删除。
15. 修改 frontend/src/pages/McpPage.tsx，根据正式 OpenAPI 和真实接入状态，仅展示已声明能力；未正式接入时只显示“接入状态 / NOT AVAILABLE / 暂未接入”。
16. 补充前端 normalizer、API 请求、页面门禁、刷新恢复、生命周期操作、审计筛选分页和 MCP 边界测试。 [verify: npm:test]
17. 完成前端类型检查并修复合同、组件 props、路由和状态适配问题。 [verify: npm:typecheck]
18. 使用 fixture 和原型截图进行视觉回归，仅调整 frontend/src/styles/global.css、workflow.css、registry.css 及组件布局，使导航、步骤条、表格、按钮位置和审计区域符合原型，不改变字段语义和流程顺序。
19. 执行真实纵向链路验收：选择模型、上传数据、预处理/跳过、划分、训练轮询、评估、保存 READY、发布、下线、异常、回滚、告警确认，并确认所有成功/失败事件可在 /audit 分页查询。 [verify: python:pytest]
20. 交付改造后的完整文件路径、接口合同变更、数据库兼容说明、测试结果及仍需后端补合同的能力边界。
<!-- step:2026-09-07T14-19-10-219Z:1 -->
## 步骤 1：以 产品原型.html、统一产品设计与最小改造方案.md、backend/app 和 frontend/src 为基线，建立页面字段、接口、状态枚举和资源 ID 链路差异清单。
- 完成时间：2026-09-07T14:19:11.089Z
- verify：无
- commit：未检测到新提交（worker 可能未按约定提交）
<!-- step:2026-09-07T14-19-10-219Z:2 -->
## 步骤 2：冻结统一合同模型：规范 model_type、资源 ID、READY/PUBLISHED/RETIRED/FAILED 生命周期、HEALTHY/ABNORMAL/UNKNOWN 健康状态及 ACTIVE/ACKNOWLEDGED/RESOLVED 告警状态。
- 完成时间：2026-09-07T14:48:07.844Z
- verify：无
- commit：未检测到新提交（worker 可能未按约定提交）