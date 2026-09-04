const modelDefaults = [
  ['electric_load', '电力负荷预测'],
  ['heating_cooling_load', '冷热负荷预测'],
  ['integrated_energy', '综合能耗预测'],
] as const

const predictRequest = `{
  "model_type": "electric_load",
  "model_version": "v2",
  "data": [{
    "timestamp": "2026-01-01 10:00:00",
    "temperature": 23.5,
    "humidity": 60
  }]
}`

const predictResponse = `{
  "success": true,
  "model_type": "electric_load",
  "model_version": "v2",
  "preprocess_used": true,
  "predictions": [1250.36]
}`

const abnormalRequest = `{
  "model_type": "electric_load",
  "model_version": "v2",
  "abnormal": true,
  "reason": "连续预测偏差过大"
}`

const errorResponse = `{
  "success": false,
  "error_code": "MISSING_FEATURE",
  "message": "缺少特征字段 humidity",
  "details": {
    "missing_fields": ["humidity"],
    "required_fields": ["timestamp", "temperature", "humidity"]
  }
}`

const errorCodes = [
  ['MODEL_TYPE_NOT_FOUND', '模型类型不存在'],
  ['MODEL_VERSION_NOT_FOUND', '指定版本不存在'],
  ['MODEL_VERSION_UNAVAILABLE', '版本异常、训练中、待发布或已下线，不能用于预测'],
  ['MISSING_TIME_FIELD', '缺少时间字段'],
  ['MISSING_FEATURE', '缺少必填特征字段'],
  ['INVALID_FIELD_TYPE', '字段类型不正确'],
  ['INVALID_TIME_FORMAT', '时间格式不正确'],
  ['PREPROCESS_FAILED', '预处理执行失败'],
  ['PREDICTION_FAILED', '模型预测失败'],
] as const

function CodeBlock({ children }: { children: string }) {
  return <pre className="code-block"><code>{children}</code></pre>
}

function ParameterTable({ rows }: { rows: Array<[string, string, string, string]> }) {
  return <div className="table-wrap parameter-table-wrap"><table><thead><tr><th>参数</th><th>必填</th><th>类型</th><th>说明</th></tr></thead><tbody>{rows.map(([name, required, type, description]) => <tr key={name}><td><code>{name}</code></td><td>{required}</td><td>{type}</td><td>{description}</td></tr>)}</tbody></table></div>
}

export function McpPage() {
  return <div className="page-content mcp-page">
    <div className="mcp-header"><div><p className="eyebrow">MCP SERVICE / API GUIDE</p><h1 className="page-title">MCP 服务说明</h1><p className="page-description">面向预测调用方的工具契约、默认版本规则和错误处理说明。以下示例以当前后端数据契约为准。</p></div><div className="service-address-card"><small>MCP 服务地址</small><code>未提供</code><span className="api-base-note">当前 API 基址：/api</span><span className="status-badge warning-badge">MCP 路由待接入</span></div></div>
    <div className="alert-box warning mcp-notice" role="status"><b>当前后端能力边界</b><span>已实现模型生命周期接口（包括 <code>/api/models/{'{id}'}/abnormal</code>），但当前代码未提供独立的 MCP <code>predict</code> 或工具发现路由。页面中的 MCP 契约用于联调说明，不会伪造可用接口。</span></div>
    <section className="mcp-panel"><div className="mcp-panel-heading"><div><p className="eyebrow">TOOL 01</p><h2><code>predict</code> · 模型预测</h2><p>提交 JSON 行记录，按指定版本或默认规则执行预测。</p></div><span className="status-badge neutral">待后端提供 MCP endpoint</span></div><ParameterTable rows={[["model_type", "是", "string", "模型类型：electric_load / heating_cooling_load / integrated_energy"], ["model_version", "否", "string", "模型版本；省略时自动选择当前有效、最新已发布且健康的版本"], ["data", "是", "object[]", "待预测的 JSON 行记录数组，字段必须符合模型输入规范"]]} /><div className="mcp-two-column"><div><h3>请求示例</h3><CodeBlock>{predictRequest}</CodeBlock></div><div><h3>成功返回示例</h3><CodeBlock>{predictResponse}</CodeBlock></div></div><div className="rule-callout"><strong>默认版本规则</strong><span>省略 <code>model_version</code> 时，系统自动选择该模型类型的<strong>当前有效 + 最新已发布 + 健康</strong>版本；指定了版本但不可用时不会静默切换。</span></div></section>
    <section className="mcp-panel"><div className="mcp-panel-heading"><div><p className="eyebrow">TOOL 02</p><h2><code>mark_model_abnormal</code> · 标记模型异常</h2><p>当监控发现预测偏差或制品异常时，触发异常记录与可用的自动回滚。</p></div><span className="status-badge neutral">生命周期 API 已提供</span></div><ParameterTable rows={[["model_type", "是", "string", "MCP 工具契约中的模型类型"], ["model_version", "是", "string", "需要标记的版本号"], ["abnormal", "是", "boolean", "必须为 true；false 不能自动解除告警"], ["reason", "是", "string", "异常原因，不能为空"]]} /><div className="mcp-two-column"><div><h3>工具请求示例</h3><CodeBlock>{abnormalRequest}</CodeBlock></div><div className="implementation-note"><h3>当前后端调用方式</h3><p>后端实际提供的是按版本 ID 调用的生命周期接口：</p><CodeBlock>{`POST /api/models/{id}/abnormal

{ "reason": "连续预测偏差过大" }`}</CodeBlock><p>需要调用方先通过 <code>GET /api/models</code> 将模型类型与版本号解析为版本 ID。独立 MCP 适配层尚未实现。</p></div></div></section>
    <section className="mcp-panel"><div className="mcp-panel-heading"><div><p className="eyebrow">DEFAULT RESOLUTION</p><h2>三类模型的默认版本</h2><p>三类模型统一采用同一套安全解析规则，不因模型类型不同而降级。</p></div></div><div className="default-model-grid">{modelDefaults.map(([code, name]) => <article key={code}><span className="model-symbol">{code === 'electric_load' ? '⚡' : code === 'heating_cooling_load' ? '◒' : '⌁'}</span><h3>{name}</h3><code>{code}</code><p>省略 <code>model_version</code> 时 → 当前有效、最新已发布、健康版本。</p></article>)}</div></section>
    <section className="mcp-panel"><div className="mcp-panel-heading"><div><p className="eyebrow">ERROR CONTRACT</p><h2>错误码与处理</h2><p>目标 MCP 错误返回统一使用 <code>success: false</code>；指定版本不可用时必须直接报错。</p></div></div><div className="mcp-two-column error-contract"><div><h3>错误返回示例</h3><CodeBlock>{errorResponse}</CodeBlock></div><div className="error-code-list">{errorCodes.map(([code, description]) => <div key={code}><code>{code}</code><span>{description}</span></div>)}</div></div><div className="implementation-note compact-note"><b>当前实现提示：</b>生命周期接口的错误由 FastAPI 以 HTTP 400/409 和 <code>detail.code</code> 返回；预测及 MCP 统一错误 envelope 需后端补充后才能正式调用。</div></section>
  </div>
}
