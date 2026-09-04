import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, apiClient } from '../api'
import { MODEL_TYPE_CODES, MODEL_TYPE_NAMES, type LifecycleOperationResponse, type ModelAlert, type ModelTypeCode, type ModelVersionDetail, type ModelVersionSummary, type RollbackRecord } from '../types/contracts'

type ActionType = 'publish' | 'offline' | 'rollback'

const lifecycleLabels: Record<string, string> = {
  DRAFT: '草稿', TRAINING: '训练中', READY: '待发布', PUBLISHED: '已发布', RETIRED: '已下线', ABNORMAL: '异常', FAILED: '失败',
}
const healthLabels: Record<string, string> = { HEALTHY: '健康', ABNORMAL: '异常' }

function errorMessage(reason: unknown) {
  return reason instanceof ApiError ? reason.message : reason instanceof Error ? reason.message : '请求失败，请稍后重试'
}

function formatDate(value?: string | null) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function lifecycle(model: ModelVersionSummary) {
  return lifecycleLabels[model.status] ?? model.status
}

function health(model: ModelVersionSummary) {
  const value = String(model.health_status ?? (model.is_abnormal ? 'ABNORMAL' : 'HEALTHY')).toUpperCase()
  return healthLabels[value] ?? value
}

function isHealthy(model: ModelVersionSummary) {
  return String(model.health_status ?? 'HEALTHY').toUpperCase() === 'HEALTHY' && !model.is_abnormal
}

function isRollbackTarget(model: ModelVersionSummary, current?: ModelVersionSummary) {
  return Boolean(current && current.id !== model.id && isHealthy(model) && model.published_at && ['PUBLISHED', 'RETIRED'].includes(model.status))
}

function metricEntries(metrics: ModelVersionSummary['metrics']) {
  if (!metrics || Array.isArray(metrics)) return []
  return Object.entries(metrics).filter(([, value]) => typeof value === 'number')
}

function versionScript(model: ModelVersionSummary, kind: 'train' | 'preprocess') {
  const script = kind === 'train' ? model.train_script : model.preprocess_script
  const id = kind === 'train' ? model.train_script_id : model.preprocess_script_id
  const version = kind === 'train' ? model.train_script_version : model.preprocess_script_version
  if (!id && !script) return '未使用'
  return `${script?.name ?? id ?? '脚本'}${script?.version ?? version ? ` · ${script?.version ?? version}` : ''}`
}

function ActionButton({ action, model, disabled, onClick }: { action: ActionType; model: ModelVersionSummary; disabled?: boolean; onClick: () => void }) {
  const labels: Record<ActionType, string> = { publish: '发布', offline: '下线', rollback: '回滚到此版本' }
  return <button className={action === 'rollback' ? 'text-button action-link-button' : action === 'offline' ? 'secondary-button action-button' : 'primary-button action-button'} disabled={disabled} onClick={onClick} type="button">{labels[action]}{action === 'rollback' && ` · ${model.version}`}</button>
}

function ConfirmDialog({ action, model, busy, onCancel, onConfirm }: { action: ActionType; model: ModelVersionSummary; busy: boolean; onCancel: () => void; onConfirm: () => void }) {
  const copy: Record<ActionType, { title: string; description: string }> = {
    publish: { title: `确认发布 ${model.version}？`, description: '发布会替换该模型类型的当前有效版本，并解除该类型现有告警。' },
    offline: { title: `确认下线 ${model.version}？`, description: '下线只改变生命周期状态，不会删除模型制品；如果它是当前版本，当前指针也会被清空。' },
    rollback: { title: `确认回滚到 ${model.version}？`, description: '系统会将该健康的历史已发布版本切换为当前有效版本，并保留回滚记录。' },
  }
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel() }}>
    <section aria-labelledby="confirm-title" aria-modal="true" className="confirm-dialog" role="dialog">
      <p className="eyebrow">危险操作确认</p>
      <h2 id="confirm-title">{copy[action].title}</h2>
      <p>{copy[action].description}</p>
      <div className="confirm-model"><b>{MODEL_TYPE_NAMES[model.model_type]}</b><span>{model.version} · 生命周期：{lifecycle(model)} · 健康：{health(model)}</span></div>
      <div className="dialog-actions"><button className="secondary-button action-button" disabled={busy} onClick={onCancel} type="button">取消</button><button className={action === 'publish' ? 'primary-button action-button' : 'danger-button action-button'} disabled={busy} onClick={onConfirm} type="button">{busy ? '处理中…' : `确认${action === 'publish' ? '发布' : action === 'offline' ? '下线' : '回滚'}`}</button></div>
    </section>
  </div>
}

function VersionDetails({ model, records, alerts, loading, error, onClose }: { model: ModelVersionDetail | ModelVersionSummary; records: RollbackRecord[]; alerts: ModelAlert[]; loading: boolean; error: string | null; onClose: () => void }) {
  const metrics = metricEntries(model.metrics)
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <aside aria-labelledby="version-detail-title" aria-modal="true" className="details-drawer" role="dialog">
      <div className="drawer-header"><div><p className="eyebrow">VERSION DETAIL</p><h2 id="version-detail-title">{model.version}</h2><span className="muted-caption">{MODEL_TYPE_NAMES[model.model_type]}</span></div><button aria-label="关闭详情" className="icon-button" onClick={onClose} type="button">×</button></div>
      {loading && <div className="loading-line"><span className="spinner" />正在加载版本详情…</div>}
      {error && <div className="alert-box error" role="alert">{error}</div>}
      <div className="detail-statuses"><span className="status-badge info">生命周期：{lifecycle(model)}</span><span className={`status-badge ${isHealthy(model) ? 'success' : 'danger'}`}>健康：{health(model)}</span>{model.is_current && <span className="status-badge success">当前有效</span>}</div>
      <div className="detail-grid"><div><small>创建时间</small><b>{formatDate(model.created_at)}</b></div><div><small>发布时间</small><b>{formatDate(model.published_at)}</b></div><div><small>训练脚本</small><b>{versionScript(model, 'train')}</b></div><div><small>预处理脚本</small><b>{versionScript(model, 'preprocess')}</b></div><div><small>模型制品</small><b>{model.model_path ?? '—'}</b></div><div><small>回滚基线</small><b>{model.previous_healthy_version_id ?? '—'}</b></div></div>
      <section className="detail-section"><div className="detail-section-title"><h3>评估指标</h3><span>{metrics.length ? `${metrics.length} 项` : '暂无指标'}</span></div>{metrics.length ? <div className="detail-metrics">{metrics.map(([name, value]) => <div key={name}><small>{name.toUpperCase()}</small><strong>{typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 6 }) : '—'}</strong></div>)}</div> : <div className="empty-state compact">该版本没有返回评估指标。</div>}</section>
      <section className="detail-section"><div className="detail-section-title"><h3>输入与预处理</h3></div><dl className="detail-list"><div><dt>时间列</dt><dd>{model.time_column ?? '—'}</dd></div><div><dt>特征列</dt><dd>{model.feature_columns?.join('、') || '—'}</dd></div><div><dt>目标列</dt><dd>{model.target_column ?? '—'}</dd></div><div><dt>预处理状态</dt><dd>{model.preprocess_used ? '已使用预处理' : '未使用预处理'}</dd></div></dl></section>
      <section className="detail-section"><div className="detail-section-title"><h3>关联告警</h3><span>{alerts.length} 条</span></div>{alerts.length ? <div className="record-list">{alerts.map((alert) => <div className="record-item" key={alert.id}><div><b>{alert.status === 'ACTIVE' ? '进行中' : '已解决'}</b><span>{alert.reason}</span></div><small>{formatDate(alert.created_at)}</small></div>)}</div> : <div className="empty-state compact">暂无该模型类型的告警记录。</div>}</section>
      <section className="detail-section"><div className="detail-section-title"><h3>回滚记录</h3><span>{records.length} 条</span></div>{records.length ? <div className="record-list">{records.map((record) => <div className="record-item" key={record.id}><div><b>{record.status === 'SUCCEEDED' ? '成功' : record.status === 'FAILED' ? '失败' : '处理中'}</b><span>{record.rollback_from ?? '—'} → {record.rollback_to ?? '无目标'}</span><small>{record.reason ?? '未提供原因'}</small></div><small>{formatDate(record.created_at)}</small></div>)}</div> : <div className="empty-state compact">暂无与此版本关联的回滚记录。</div>}</section>
    </aside>
  </div>
}

export function ModelVersionsPage() {
  const [models, setModels] = useState<ModelVersionSummary[]>([])
  const [alerts, setAlerts] = useState<ModelAlert[]>([])
  const [selectedType, setSelectedType] = useState<ModelTypeCode>('electric_load')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [selected, setSelected] = useState<ModelVersionSummary | null>(null)
  const [detail, setDetail] = useState<ModelVersionDetail | null>(null)
  const [detailRecords, setDetailRecords] = useState<RollbackRecord[]>([])
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<{ action: ActionType; model: ModelVersionSummary } | null>(null)
  const [actionLoading, setActionLoading] = useState(false)

  const load = useCallback(async (initial = false) => {
    if (initial) setLoading(true); else setRefreshing(true)
    setError(null)
    try {
      const [versions, alertRows] = await Promise.all([apiClient.listModels(), apiClient.listAlerts()])
      setModels(versions); setAlerts(alertRows)
    } catch (reason) { setError(errorMessage(reason)) } finally { setLoading(false); setRefreshing(false) }
  }, [])

  useEffect(() => { void load(true) }, [load])

  const versions = useMemo(() => models.filter((model) => model.model_type === selectedType), [models, selectedType])
  const current = versions.find((model) => model.is_current)
  const typeAlerts = alerts.filter((alert) => alert.model_type === selectedType)

  const openDetails = async (model: ModelVersionSummary) => {
    setSelected(model); setDetail(null); setDetailRecords([]); setDetailError(null); setDetailLoading(true)
    try {
      const [modelDetail, records] = await Promise.all([apiClient.getModel(model.id), apiClient.getModelRollbackRecords(model.id)])
      setDetail(modelDetail); setDetailRecords(records)
    } catch (reason) { setDetailError(errorMessage(reason)) } finally { setDetailLoading(false) }
  }

  const execute = async () => {
    if (!confirm) return
    setActionLoading(true); setError(null); setSuccess(null)
    try {
      let response: LifecycleOperationResponse
      if (confirm.action === 'publish') response = await apiClient.publishModel(confirm.model.id, { confirm: true }) as LifecycleOperationResponse
      else if (confirm.action === 'offline') response = await apiClient.offlineModel(confirm.model.id)
      else response = await apiClient.rollbackModel(confirm.model.id, { reason: '手动回滚' })
      const actionLabel = confirm.action === 'publish' ? '发布' : confirm.action === 'offline' ? '下线' : '回滚'
      setSuccess(`${confirm.model.version} ${actionLabel}成功，列表已刷新。`)
      setConfirm(null)
      await load()
      if (selected?.id === confirm.model.id) await openDetails({ ...confirm.model, ...(response.model as ModelVersionSummary) })
    } catch (reason) { setError(errorMessage(reason)) } finally { setActionLoading(false) }
  }

  return <div className="page-content model-registry-page">
    <div className="registry-header"><div><p className="eyebrow">MODEL REGISTRY / LIFECYCLE</p><h1 className="page-title">模型版本管理</h1><p className="page-description">按模型类型查看版本、评估指标和生产状态。生命周期状态与运行健康状态分别展示，危险操作均需二次确认。</p></div><button className="secondary-button action-button" disabled={loading || refreshing} onClick={() => void load()} type="button">{refreshing ? '同步中…' : '刷新列表 ↻'}</button></div>
    {error && <div className="alert-box error" role="alert"><span>{error}</span><button type="button" onClick={() => void load()}>重试</button></div>}
    {success && <div className="alert-box success" role="status"><span>{success}</span><button type="button" onClick={() => setSuccess(null)}>关闭</button></div>}
    <div className="model-type-tabs" role="tablist" aria-label="模型类型"><span className="tab-caption">模型类型</span>{MODEL_TYPE_CODES.map((code) => <button aria-selected={selectedType === code} className={`model-type-tab${selectedType === code ? ' active' : ''}`} key={code} onClick={() => setSelectedType(code)} role="tab" type="button">{MODEL_TYPE_NAMES[code]}<small>{models.filter((item) => item.model_type === code).length} 个版本</small></button>)}</div>
    <section className="registry-panel"><div className="panel-heading"><div><p className="eyebrow">{selectedType.toUpperCase()}</p><h2>{MODEL_TYPE_NAMES[selectedType]} · 版本列表</h2><p>当前有效版本：<b>{current?.version ?? '暂无'}</b> · 当前健康状态由后端模型状态直接返回。</p></div><span className="panel-count">{loading ? '加载中…' : `${versions.length} 个版本`}</span></div>
      {loading && <div className="loading-state"><span className="spinner" />正在加载模型版本与告警…</div>}
      {!loading && !versions.length && <div className="empty-state"><strong>暂无{MODEL_TYPE_NAMES[selectedType]}版本</strong><span>完成训练并保存模型后，版本会出现在这里。</span></div>}
      {!loading && versions.length > 0 && <div className="table-wrap registry-table-wrap"><table className="registry-table"><thead><tr><th>版本</th><th>创建时间</th><th>训练 / 预处理脚本</th><th>评估指标</th><th>生命周期</th><th>健康状态</th><th>当前有效</th><th>可回滚</th><th>操作</th></tr></thead><tbody>{versions.map((model) => { const rollback = isRollbackTarget(model, current); return <tr key={model.id} className={model.is_current ? 'current-row' : ''}><td><button className="version-link" onClick={() => void openDetails(model)} type="button"><b>{model.version}</b><span>{model.is_baseline ? '基线版本' : model.id}</span></button></td><td>{formatDate(model.created_at)}</td><td><span className="script-cell">训练：{versionScript(model, 'train')}<br />预处理：{versionScript(model, 'preprocess')}</span></td><td>{metricEntries(model.metrics).slice(0, 3).map(([name, value]) => <span className="metric-inline" key={name}>{name.toUpperCase()} {typeof value === 'number' ? value.toFixed(4) : '—'}</span>)}{!metricEntries(model.metrics).length && <span className="muted-caption">暂无</span>}</td><td><span className={`status-badge ${model.status === 'ABNORMAL' ? 'danger' : model.status === 'PUBLISHED' ? 'success' : 'info'}`}>{lifecycle(model)}</span></td><td><span className={`health-indicator ${isHealthy(model) ? 'healthy' : 'abnormal'}`}><i />{health(model)}</span></td><td>{model.is_current ? <span className="current-mark">✓ 当前</span> : '—'}</td><td>{rollback ? <span className="can-rollback">可回滚</span> : <span className="muted-caption">不可用</span>}</td><td><div className="row-actions"><button className="text-button" onClick={() => void openDetails(model)} type="button">详情</button>{model.status === 'READY' && isHealthy(model) && <ActionButton action="publish" disabled={actionLoading} model={model} onClick={() => setConfirm({ action: 'publish', model })} />}{model.status === 'PUBLISHED' && <ActionButton action="offline" disabled={actionLoading} model={model} onClick={() => setConfirm({ action: 'offline', model })} />}{rollback && <ActionButton action="rollback" disabled={actionLoading} model={model} onClick={() => setConfirm({ action: 'rollback', model })} />}</div></td></tr> })}</tbody></table></div>}
    </section>
    <section className="registry-lower-grid"><div className="registry-panel compact-panel"><div className="panel-heading"><div><h2>告警记录</h2><p>按当前模型类型展示 ACTIVE / RESOLVED 历史。</p></div><span className="panel-count">{typeAlerts.length} 条</span></div>{typeAlerts.length ? <div className="record-list">{typeAlerts.map((alert) => <div className="record-item" key={alert.id}><div><b className={alert.status === 'ACTIVE' ? 'danger-text' : 'success-text'}>{alert.status === 'ACTIVE' ? '进行中' : '已解决'}</b><span>{alert.reason}</span><small>版本：{alert.model_version_id ?? '—'} · 回滚目标：{alert.rollback_to ?? '无'}</small></div><small>{formatDate(alert.created_at)}</small></div>)}</div> : <div className="empty-state compact">暂无告警记录。</div>}</div><div className="registry-panel compact-panel"><div className="panel-heading"><div><h2>状态规则</h2><p>后端当前生产指针与可回滚范围。</p></div></div><ul className="rule-list"><li><b>生命周期</b><span>草稿 / 待发布 / 已发布 / 已下线 / 异常 / 失败</span></li><li><b>健康状态</b><span>健康与异常独立于生命周期；异常版本不能发布或成为当前版本。</span></li><li><b>回滚</b><span>仅健康且已有发布时间的发布版本可作为回滚目标，模型制品不会被删除。</span></li></ul></div></section>
    {selected && <VersionDetails alerts={typeAlerts} error={detailError} loading={detailLoading} model={detail ?? selected} onClose={() => setSelected(null)} records={detailRecords} />}
    {confirm && <ConfirmDialog action={confirm.action} busy={actionLoading} model={confirm.model} onCancel={() => setConfirm(null)} onConfirm={() => void execute()} />}
  </div>
}
