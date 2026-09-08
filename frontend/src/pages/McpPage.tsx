import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import { MODEL_TYPE_CODES } from '../types/contracts'

type SchemaRow = readonly [string, string, string, string]

type McpPageState = 'loading' | 'unavailable' | 'available'

const modelTypes = MODEL_TYPE_CODES.join(' / ')

// These rows mirror backend/app/schemas/mcp.py. They describe the validated
// request bodies, not an executable example or a client-side prediction form.
const predictRequestSchema: SchemaRow[] = [
  ['model_type', '必填', 'ModelType', `取值：${modelTypes}`],
  ['model_version', '可选', 'string | null', '长度 1–100；未提供时由后端版本解析合同处理'],
  ['data', '必填', 'list[object]', 'JSON 记录数组；输入字段按已保存的模型版本 schema 校验'],
]

const predictResponseSchema: SchemaRow[] = [
  ['success', '返回', 'true', '预测成功标志'],
  ['model_type', '返回', 'ModelType', '实际使用的模型类型'],
  ['model_version', '返回', 'string', '实际使用的模型版本'],
  ['preprocess_used', '返回', 'boolean', '是否使用已保存的预处理状态'],
  ['predictions', '返回', 'number[]', '按输入记录顺序返回的预测值'],
]

const abnormalRequestSchema: SchemaRow[] = [
  ['model_type', '必填', 'ModelType', `取值：${modelTypes}`],
  ['model_version', '必填', 'string', '长度 1–100；版本号或版本资源 ID'],
  ['abnormal', '可选', 'boolean', '默认值：true'],
  ['reason', '可选', 'string', '默认值：健康检查异常；长度 1–2000'],
]

const abnormalResponseSchema: SchemaRow[] = [
  ['success', '返回', 'true', '异常标记处理成功标志'],
  ['model_type', '返回', 'ModelType', '处理的模型类型'],
  ['model_version', '返回', 'string', '处理的模型版本'],
  ['abnormal', '返回', 'boolean', '本次请求的异常状态'],
  ['current_model_version', '返回', 'string', '处理后当前模型版本'],
  ['alert', '返回', 'object | null', '告警响应或 null'],
  ['rollback', '返回', 'object | null', '回滚响应或 null'],
  ['rollback_triggered', '返回', 'boolean', '是否触发回滚'],
  ['alert_cleared', '返回', 'boolean', '是否清除告警'],
]

const errorEnvelopeSchema: SchemaRow[] = [
  ['success', '返回', 'false', '失败标志'],
  ['error_code', '返回', 'string', '稳定业务错误码'],
  ['message', '返回', 'string', '面向调用方的错误消息'],
  ['details', '返回', 'object', '结构化错误详情；无详情时为空对象'],
]

const errorStatusSchema: SchemaRow[] = [
  ['400', '错误', 'INVALID_REQUEST / INVALID_INPUT_DATA / MISSING_TIME_FIELD / MISSING_FEATURE / INVALID_FIELD_TYPE / INVALID_TIME_FORMAT', '请求 JSON、字段或输入记录校验失败'],
  ['404', '错误', 'MODEL_TYPE_NOT_FOUND / MODEL_VERSION_NOT_FOUND', '模型类型或版本解析失败'],
  ['409', '错误', 'MODEL_VERSION_UNAVAILABLE / NO_HEALTHY_BACKUP / NO_HEALTHY_ROLLBACK_VERSION', '模型不可用或回滚条件冲突'],
  ['500', '错误', 'MODEL_LOAD_FAILED / PREPROCESS_FAILED / PREDICTION_FAILED / INTERNAL_ERROR', '模型加载、预处理或预测失败'],
]

function SchemaTable({ rows }: { rows: ReadonlyArray<SchemaRow> }) {
  return <div className="table-wrap parameter-table-wrap"><table><thead><tr><th>字段</th><th>约束</th><th>类型</th><th>说明</th></tr></thead><tbody>{rows.map(([name, constraint, type, description]) => <tr key={name}><td><code>{name}</code></td><td>{constraint}</td><td>{type}</td><td>{description}</td></tr>)}</tbody></table></div>
}

function McpFrame({ children }: { children: React.ReactNode }) {
  return <div className="page-content mcp-page">
    <div className="mcp-header"><div><p className="eyebrow">MCP</p><h1 className="page-title">MCP 服务</h1></div></div>
    {children}
  </div>
}

function UnavailableMcp() {
  return <McpFrame>
    <section className="mcp-panel" aria-label="MCP 接入状态">
      <div className="mcp-panel-heading"><div><h2>接入状态</h2></div><span className="status-badge neutral">NOT AVAILABLE</span></div>
      <strong>暂未接入</strong>
    </section>
  </McpFrame>
}

function LoadingMcp() {
  return <McpFrame>
    <section className="mcp-panel" aria-label="MCP 接入状态" aria-live="polite">
      <div className="mcp-panel-heading"><div><h2>接入状态</h2></div><span className="status-badge neutral">CHECKING</span></div>
      <strong>正在确认正式 OpenAPI 声明</strong>
    </section>
  </McpFrame>
}

function AvailableMcp() {
  return <McpFrame>
    <section className="mcp-panel">
      <div className="mcp-panel-heading"><div><p className="eyebrow">POST /api/mcp/predict</p><h2><code>predict</code></h2><p>正式 MCP 预测路由的请求与响应 schema。</p></div><span className="status-badge success">已接入</span></div>
      <div className="mcp-two-column">
        <div><h3>请求 schema · PredictRequest</h3><SchemaTable rows={predictRequestSchema} /></div>
        <div><h3>响应 schema · PredictionResponse</h3><SchemaTable rows={predictResponseSchema} /></div>
      </div>
    </section>

    <section className="mcp-panel">
      <div className="mcp-panel-heading"><div><p className="eyebrow">POST /api/mcp/mark_model_abnormal</p><h2><code>mark_model_abnormal</code></h2><p>正式 MCP 异常标记路由的请求与响应 schema。</p></div><span className="status-badge success">已接入</span></div>
      <div className="mcp-two-column">
        <div><h3>请求 schema · MCPModelAbnormalRequest</h3><SchemaTable rows={abnormalRequestSchema} /></div>
        <div><h3>响应 schema · McpMarkModelAbnormalResponse</h3><SchemaTable rows={abnormalResponseSchema} /></div>
      </div>
    </section>

    <section className="mcp-panel">
      <div className="mcp-panel-heading"><div><p className="eyebrow">ERROR CONTRACT</p><h2>统一错误合同</h2><p>两个正式 MCP 路由均返回统一失败 envelope。</p></div></div>
      <div className="mcp-two-column error-contract">
        <div><h3>失败响应 schema · McpErrorResponse</h3><SchemaTable rows={errorEnvelopeSchema} /></div>
        <div><h3>HTTP 状态与错误码</h3><SchemaTable rows={errorStatusSchema} /></div>
      </div>
    </section>
  </McpFrame>
}

export function McpPage() {
  const [state, setState] = useState<McpPageState>('loading')

  useEffect(() => {
    let active = true
    apiClient.getMcpCapabilities()
      .then((capabilities) => {
        if (active) setState(capabilities.available ? 'available' : 'unavailable')
      })
      .catch(() => {
        // An unreadable OpenAPI document cannot confirm either route, so keep
        // the prototype's strict empty boundary rather than guessing.
        if (active) setState('unavailable')
      })
    return () => {
      active = false
    }
  }, [])

  if (state === 'loading') return <LoadingMcp />
  if (state === 'unavailable') return <UnavailableMcp />
  return <AvailableMcp />
}
