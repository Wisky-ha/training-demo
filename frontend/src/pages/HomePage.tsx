import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiClient, ApiError } from '../api'
import { MODEL_TYPE_CODES, MODEL_TYPE_NAMES, type AuditEvent, type ModelAlert, type ModelTypeCode, type ModelVersionSummary } from '../types/contracts'

const UNKNOWN_TEXT = '未知'
const NOT_PROVIDED_TEXT = '未提供'

const lifecycleLabels: Record<string, string> = {
  DRAFT: '草稿',
  TRAINING: '训练中',
  READY: '待发布',
  PUBLISHED: '已发布',
  RETIRED: '已下线',
  FAILED: '失败',
}
const healthLabels: Record<string, string> = { HEALTHY: '健康', ABNORMAL: '异常', UNKNOWN: UNKNOWN_TEXT }
const alertStatusLabels: Record<string, string> = {
  ACTIVE: '活动',
  ACKNOWLEDGED: '已确认',
  RESOLVED: '已解决',
  UNKNOWN: UNKNOWN_TEXT,
}

function errorMessage(reason: unknown) {
  return reason instanceof ApiError ? reason.message : reason instanceof Error ? reason.message : '请求失败，请稍后重试'
}

function formatDate(value?: string | null) {
  if (!value) return UNKNOWN_TEXT
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return UNKNOWN_TEXT
  try {
    return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
  } catch {
    return UNKNOWN_TEXT
  }
}

function isModelType(value: string): value is ModelTypeCode {
  return (MODEL_TYPE_CODES as readonly string[]).includes(value)
}

function productionModelsByType(models: ModelVersionSummary[]) {
  const result = new Map<ModelTypeCode, ModelVersionSummary>()
  models.forEach((model) => {
    if (model.is_baseline || !model.is_current || !isModelType(model.model_type)) return
    // A production pointer is unique by model_type even if a legacy response
    // accidentally contains duplicate current rows.
    if (!result.has(model.model_type)) result.set(model.model_type, model)
  })
  return result
}

function lifecycleLabel(model?: ModelVersionSummary) {
  return model?.status ? lifecycleLabels[model.status] ?? UNKNOWN_TEXT : UNKNOWN_TEXT
}

function healthLabel(model?: ModelVersionSummary) {
  const value = model?.health_status ?? 'UNKNOWN'
  return healthLabels[value] ?? UNKNOWN_TEXT
}

function lifecycleClass(model?: ModelVersionSummary) {
  if (model?.status === 'PUBLISHED') return 'success'
  if (model?.status === 'FAILED') return 'danger'
  if (model?.status === 'READY') return 'info'
  return 'neutral'
}

function healthClass(model?: ModelVersionSummary) {
  if (model?.health_status === 'HEALTHY') return 'success'
  if (model?.health_status === 'ABNORMAL') return 'danger'
  return 'neutral'
}

function modelTypeName(value?: string | null) {
  return value && isModelType(value) ? MODEL_TYPE_NAMES[value] : UNKNOWN_TEXT
}

function alertStatusLabel(status?: ModelAlert['status'] | null) {
  if (status !== 'ACTIVE' && status !== 'ACKNOWLEDGED' && status !== 'RESOLVED') return UNKNOWN_TEXT
  return `${status}（${alertStatusLabels[status] ?? UNKNOWN_TEXT}）`
}

// Closing an alert only hides it in this browser: the server-side alert stays
// ACTIVE. The ids are persisted so a refresh does not resurrect a row the user
// already closed, and the panel always offers a restore control so the hidden
// state is visible and reversible rather than silent.
const DISMISSED_ALERTS_STORAGE_KEY = 'model-training-platform.dismissed-alert-ids'

function loadDismissedAlertIds(): string[] {
  try {
    const raw = window.localStorage.getItem(DISMISSED_ALERTS_STORAGE_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === 'string')
      : []
  } catch {
    // Unavailable or corrupt storage must never break the dashboard.
    return []
  }
}

function persistDismissedAlertIds(ids: string[]) {
  try {
    window.localStorage.setItem(DISMISSED_ALERTS_STORAGE_KEY, JSON.stringify(ids))
  } catch {
    // Without storage the dismissal simply lasts for this view.
  }
}

function auditObject(event: AuditEvent) {
  const objectId = event.object_id ?? event.model_version_id ?? event.training_job_id ?? NOT_PROVIDED_TEXT
  return {
    type: event.object_type || UNKNOWN_TEXT,
    name: event.model_type ? modelTypeName(event.model_type) : null,
    id: objectId,
  }
}

export function HomePage() {
  const [models, setModels] = useState<ModelVersionSummary[]>([])
  const [alerts, setAlerts] = useState<ModelAlert[]>([])
  const [activeAlertTotal, setActiveAlertTotal] = useState(0)
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([])
  const [trainingFinishedAt, setTrainingFinishedAt] = useState<Record<string, string | null>>({})
  const [modelsLoading, setModelsLoading] = useState(true)
  const [alertsLoading, setAlertsLoading] = useState(true)
  const [auditLoading, setAuditLoading] = useState(true)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [alertsError, setAlertsError] = useState<string | null>(null)
  const [auditError, setAuditError] = useState<string | null>(null)
  const [dismissedAlertIds, setDismissedAlertIds] = useState<string[]>(() => loadDismissedAlertIds())
  const mounted = useRef(true)

  const loadDashboard = useCallback(async () => {
    setModelsLoading(true)
    setAlertsLoading(true)
    setAuditLoading(true)
    setModelsError(null)
    setAlertsError(null)
    setAuditError(null)

    const [modelsResult, alertsResult, auditResult] = await Promise.allSettled([
      apiClient.listModels(),
      apiClient.listAlerts({ status: 'ACTIVE', page: 1, page_size: 100 }),
      apiClient.listAuditEvents({ page: 1, page_size: 10 }),
    ])
    if (!mounted.current) return

    if (modelsResult.status === 'fulfilled') {
      const nextModels = [...modelsResult.value]
      setModels(nextModels)
      const currentModels = [...productionModelsByType(nextModels).values()]
      const finishedEntries = await Promise.all(currentModels.map(async (model) => {
        if (!model.training_job_id) return [model.id, null] as const
        try {
          const job = await apiClient.getTrainingJob(model.training_job_id)
          return [model.id, job.finished_at ?? null] as const
        } catch {
          // A missing or inaccessible training job is a real unknown value,
          // not a reason to fall back to created_at or published_at.
          return [model.id, null] as const
        }
      }))
      if (!mounted.current) return
      setTrainingFinishedAt(Object.fromEntries(finishedEntries))
      setModelsLoading(false)
    } else {
      setModelsError(errorMessage(modelsResult.reason))
      setModelsLoading(false)
    }

    if (alertsResult.status === 'fulfilled') {
      const nextAlerts = [...alertsResult.value]
      setAlerts(nextAlerts)
      const total = alertsResult.value.total
      setActiveAlertTotal(typeof total === 'number' && Number.isFinite(total)
        ? total
        : nextAlerts.filter((alert) => alert.status === 'ACTIVE').length)
      setAlertsLoading(false)
    } else {
      setAlertsError(errorMessage(alertsResult.reason))
      setAlertsLoading(false)
    }

    if (auditResult.status === 'fulfilled') {
      setAuditEvents(auditResult.value.items)
      setAuditLoading(false)
    } else {
      setAuditError(errorMessage(auditResult.reason))
      setAuditLoading(false)
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    void loadDashboard()
    return () => { mounted.current = false }
  }, [loadDashboard])

  const productionByType = useMemo(() => productionModelsByType(models), [models])
  const productionRows = useMemo(
    () => MODEL_TYPE_CODES.map((code) => ({ code, model: productionByType.get(code) })),
    [productionByType],
  )
  const modelById = useMemo(() => new Map(models.map((model) => [model.id, model])), [models])
  const visibleAlerts = useMemo(
    () => alerts.filter((alert) => !dismissedAlertIds.includes(alert.id)),
    [alerts, dismissedAlertIds],
  )
  const dismissedAlertCount = alerts.length - visibleAlerts.length
  const statistics = useMemo(() => {
    const current = [...productionByType.values()]
    return {
      current: current.length,
      healthy: current.filter((model) => model.health_status === 'HEALTHY').length,
      unknown: current.filter((model) => (model.health_status ?? 'UNKNOWN') === 'UNKNOWN').length,
      ready: models.filter((model) => !model.is_baseline && model.status === 'READY').length,
    }
  }, [models, productionByType])

  const dismissAlert = useCallback((alertId: string) => {
    setDismissedAlertIds((previous) => {
      if (previous.includes(alertId)) return previous
      const next = [...previous, alertId]
      persistDismissedAlertIds(next)
      return next
    })
  }, [])

  const restoreDismissedAlerts = useCallback(() => {
    persistDismissedAlertIds([])
    setDismissedAlertIds([])
  }, [])

  return (
    <div className="page-content home-page">
      <div className="home-dashboard-grid">
        <div className="home-dashboard-main">
          <div className="registry-header home-header">
            <div>
              <p className="eyebrow">MODEL REGISTRY / 03 MODELS</p>
              <h1 className="page-title">模型总览</h1>
            </div>
            <Link className="primary-button action-button" to="/workflow/model-type?new=1">开始训练　→</Link>
          </div>

          <section aria-label="模型统计" className="summary-stat-grid home-status-strip">
            <div>
              <small>生产模型</small>
              <strong>{modelsLoading ? '加载中…' : modelsError ? '加载失败' : `${statistics.current} 个当前版本 / ${statistics.healthy} 个健康 / ${statistics.unknown} 个未知`}</strong>
            </div>
            <div>
              <small>待发布候选</small>
              <strong className={statistics.ready > 0 ? 'warning-text' : ''}>{modelsLoading ? '加载中…' : modelsError ? '加载失败' : `${statistics.ready} 个 READY 版本`}</strong>
            </div>
            <div>
              <small>活动告警</small>
              <strong className={activeAlertTotal > 0 ? 'danger-text' : ''}>{alertsLoading ? '加载中…' : alertsError ? '加载失败' : `${activeAlertTotal} 条 ACTIVE 告警`}</strong>
            </div>
          </section>

          <section className="registry-panel home-model-panel">
            <div className="panel-heading">
              <div><h2>当前生产模型</h2></div>
            </div>
            {modelsLoading && <div className="loading-state"><span className="spinner" />正在加载生产模型…</div>}
            {!modelsLoading && modelsError && <div className="alert-box error" role="alert"><span>{modelsError}</span><button type="button" onClick={() => void loadDashboard()}>重试</button></div>}
            {!modelsLoading && !modelsError && models.length === 0 && <div className="empty-state"><strong>暂无模型版本</strong><span>当前生产版本未指定。</span></div>}
            {!modelsLoading && !modelsError && models.length > 0 && <div className="table-wrap registry-table-wrap"><table className="registry-table home-model-table"><thead><tr><th>模型类型</th><th>当前有效</th><th>训练完成</th><th>生命周期 / 健康</th><th>入口</th></tr></thead><tbody>{productionRows.map(({ code, model }) => <tr key={code}>
              <td><b>{MODEL_TYPE_NAMES[code]}</b></td>
              <td><span className="version-value">{model?.version ?? '未指定'}</span></td>
              <td>{formatDate(model ? trainingFinishedAt[model.id] : null)}</td>
              <td><div className="home-state"><span className={`status-badge ${lifecycleClass(model)}`}>生命周期：{lifecycleLabel(model)}</span><span className={`status-badge ${healthClass(model)}`}>健康：{healthLabel(model)}</span></div></td>
              <td><div className="row-actions"><Link className="text-button" to={`/models?model_type=${code}`}>版本</Link><Link className="text-button" to={`/workflow/model-type?model=${code}&new=1`}>训练</Link></div></td>
            </tr>)}</tbody></table></div>}
          </section>

          <section className="registry-panel home-alert-panel">
            <div className="panel-heading">
              <div><h2>活动告警</h2></div>
              {!alertsLoading && !alertsError && <span className="panel-count">{alerts.length > 0 ? `${visibleAlerts.length} 条` : `${activeAlertTotal} 条`}</span>}
            </div>
            {alertsLoading && <div className="loading-state"><span className="spinner" />正在加载活动告警…</div>}
            {!alertsLoading && alertsError && <div className="alert-box error" role="alert"><span>{alertsError}</span><button type="button" onClick={() => void loadDashboard()}>重试</button></div>}
            {!alertsLoading && !alertsError && activeAlertTotal === 0 && <div className="empty-state compact">暂无活动告警</div>}
            {!alertsLoading && !alertsError && activeAlertTotal > 0 && alerts.length === 0 && <div className="empty-state compact">活动告警明细未提供</div>}
            {!alertsLoading && !alertsError && alerts.length > 0 && visibleAlerts.length === 0 && <div className="empty-state compact home-alert-dismissed-note">
              <span>已在本机关闭全部 {dismissedAlertCount} 条告警显示，服务端告警状态未改变。</span>
              <button className="text-button" onClick={restoreDismissedAlerts} type="button">恢复显示</button>
            </div>}
            {!alertsLoading && !alertsError && visibleAlerts.length > 0 && <div className="record-list">{visibleAlerts.map((alert) => {
              const version = alert.model_version_id ? modelById.get(alert.model_version_id) : undefined
              const rollbackFrom = alert.rollback_from ? modelById.get(alert.rollback_from) : undefined
              const rollbackTo = alert.rollback_to ? modelById.get(alert.rollback_to) : undefined
              const rollback = alert.rollback_from || alert.rollback_to
              const rollbackLabel = (resolved?: ModelVersionSummary, fallbackId?: string | null) =>
                resolved?.version ?? (fallbackId ? UNKNOWN_TEXT : NOT_PROVIDED_TEXT)
              return <div className="record-item home-alert-row" key={alert.id}>
                <div>
                  <b>{version?.version ?? UNKNOWN_TEXT}</b>
                  <div className="home-alert-copy">
                    <span>{modelTypeName(version?.model_type ?? alert.model_type)} · 原因：{alert.reason || NOT_PROVIDED_TEXT}</span>
                    <small>状态：{alertStatusLabel(alert.status)}</small>
                    {rollback && <small>回滚：{rollbackLabel(rollbackFrom, alert.rollback_from)} → {rollbackLabel(rollbackTo, alert.rollback_to)}</small>}
                  </div>
                </div>
                <div className="row-actions">
                  <Link className="text-button" to="/audit">查看审计事件</Link>
                  <button
                    aria-label={`关闭告警显示：${version?.version ?? alert.reason ?? NOT_PROVIDED_TEXT}`}
                    className="icon-button home-alert-dismiss"
                    onClick={() => dismissAlert(alert.id)}
                    title="关闭告警显示（仅在本机隐藏，不修改服务端告警状态）"
                    type="button"
                  >×</button>
                </div>
              </div>
            })}</div>}
            {!alertsLoading && !alertsError && visibleAlerts.length > 0 && dismissedAlertCount > 0 && <div className="home-alert-dismissed-note">
              <span>已在本机关闭 {dismissedAlertCount} 条告警显示，服务端告警状态未改变。</span>
              <button className="text-button" onClick={restoreDismissedAlerts} type="button">恢复显示</button>
            </div>}
          </section>
        </div>

        <aside className="registry-panel compact-panel home-audit-panel">
          <div className="panel-heading">
            <div><h2>最近审计事件</h2></div>
          </div>
          {auditLoading && <div className="loading-state"><span className="spinner" />正在加载审计事件…</div>}
          {!auditLoading && auditError && <div className="alert-box error" role="alert"><span>{auditError}</span><button type="button" onClick={() => void loadDashboard()}>重试</button></div>}
          {!auditLoading && !auditError && auditEvents.length === 0 && <div className="empty-state compact">暂无审计事件</div>}
          {!auditLoading && !auditError && auditEvents.length > 0 && <div className="record-list home-audit-list">{auditEvents.map((event) => {
            const object = auditObject(event)
            return <div className="record-item home-audit-item" key={event.id}>
              <div><b>{event.event_type}</b><span>{object.type}{object.name ? ` · ${object.name}` : ''} · {object.id}</span></div>
              <small>{formatDate(event.occurred_at)}</small>
            </div>
          })}</div>}
          <Link className="text-button" to="/audit">查看全部审计事件</Link>
        </aside>
      </div>
    </div>
  )
}
