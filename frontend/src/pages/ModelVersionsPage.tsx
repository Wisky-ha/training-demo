import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'
import { ApiError, apiClient, createIdempotencyKey } from '../api'
import { MODEL_TYPE_CODES, MODEL_TYPE_NAMES, type ModelAlert, type ModelTypeCode, type ModelVersionDetail, type ModelVersionSummary } from '../types/contracts'

export const MODEL_REGISTRY_UPDATED_EVENT = 'model-registry-updated'

type ActionType = 'publish' | 'offline' | 'abnormal' | 'rollback'
type LifecycleCode = 'READY' | 'PUBLISHED' | 'RETIRED' | 'FAILED'
type HealthCode = 'HEALTHY' | 'ABNORMAL' | 'UNKNOWN'

type ConfirmState = {
  action: ActionType
  model: ModelVersionSummary
  current: ModelVersionSummary | null
}

type AuxiliaryData = {
  trainingFinishedAt: Record<string, string | null>
  scriptNames: Record<string, string | null>
}

type RefreshResult = {
  ok: boolean
  stale: boolean
}

const lifecycleLabels: Record<LifecycleCode, string> = {
  READY: '待发布',
  PUBLISHED: '已发布',
  RETIRED: '已下线',
  FAILED: '失败',
}
const healthLabels: Record<HealthCode, string> = {
  HEALTHY: '健康',
  ABNORMAL: '异常',
  UNKNOWN: '未知',
}
const lifecycleCodes = new Set<LifecycleCode>(['READY', 'PUBLISHED', 'RETIRED', 'FAILED'])

const UNKNOWN_TEXT = '未知'
const NOT_PROVIDED_TEXT = '未提供'

function errorMessage(reason: unknown) {
  return reason instanceof ApiError ? reason.message : reason instanceof Error ? reason.message : '请求失败，请稍后重试'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isModelType(value: string | null): value is ModelTypeCode {
  return value !== null && (MODEL_TYPE_CODES as readonly string[]).includes(value)
}

function upper(value: unknown) {
  return typeof value === 'string' ? value.toUpperCase() : null
}

function lifecycleCode(model: ModelVersionSummary): LifecycleCode | null {
  const value = upper(model.status)
  return value && lifecycleCodes.has(value as LifecycleCode) ? value as LifecycleCode : null
}

function healthCode(model: ModelVersionSummary): HealthCode {
  // ABNORMAL in status is a legacy mixed value. It remains health evidence,
  // while the lifecycle display intentionally stays within the four public
  // lifecycle values above.
  if (upper(model.status) === 'ABNORMAL') return 'ABNORMAL'
  const value = upper(model.health_status)
  if (value === 'ABNORMAL') return 'ABNORMAL'
  if (value === 'HEALTHY' && model.is_abnormal !== true) return 'HEALTHY'
  // An explicit legacy abnormal flag is useful evidence, but an absent flag
  // must never turn an absent health check into HEALTHY.
  if (model.is_abnormal === true) return 'ABNORMAL'
  return 'UNKNOWN'
}

function lifecycleValue(model: ModelVersionSummary) {
  const code = lifecycleCode(model)
  return code ? `${code}（${lifecycleLabels[code]}）` : UNKNOWN_TEXT
}

function healthValue(model: ModelVersionSummary) {
  const code = healthCode(model)
  return `${code}（${healthLabels[code]}）`
}

function versionTitle(model: ModelVersionSummary) {
  const health = healthCode(model)
  if (health === 'ABNORMAL') return model.is_current ? '异常当前版本' : '异常历史版本'
  if (model.is_current) return '当前有效版本'
  switch (lifecycleCode(model)) {
    case 'READY': return '候选版本'
    case 'PUBLISHED': return '已发布版本'
    case 'RETIRED': return '历史版本'
    case 'FAILED': return '失败版本'
    default: return '版本状态未知'
  }
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

function metricValue(metrics: ModelVersionSummary['metrics'], name: 'mae' | 'rmse' | 'mape' | 'r2') {
  if (!isRecord(metrics)) return UNKNOWN_TEXT
  const value = metrics[name]
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: 6 })
    : UNKNOWN_TEXT
}

function embeddedReason(model: ModelVersionSummary) {
  const value = (model as ModelVersionSummary & { reason?: unknown }).reason
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function modelReason(model: ModelVersionSummary, alerts: ModelAlert[]) {
  const fromModel = embeddedReason(model)
  if (fromModel) return fromModel
  // An alert belongs to a version only when its immutable version ID matches.
  // Never borrow a reason from another version of the same model type.
  const alert = alerts.find((item) => item.model_version_id === model.id && typeof item.reason === 'string' && item.reason.trim())
  return alert?.reason.trim() || null
}

function nestedScriptName(model: ModelVersionSummary, kind: 'train' | 'preprocess') {
  return kind === 'train' ? model.train_script?.name : model.preprocess_script?.name
}

function scriptName(model: ModelVersionSummary, kind: 'train' | 'preprocess', names: Record<string, string | null>) {
  const nested = nestedScriptName(model, kind)
  if (nested) return nested
  const id = kind === 'train' ? model.train_script_id : model.preprocess_script_id
  return id && names[id] ? names[id] : NOT_PROVIDED_TEXT
}

function scriptDetail(model: ModelVersionSummary, kind: 'train' | 'preprocess', names: Record<string, string | null>) {
  const id = kind === 'train' ? model.train_script_id : model.preprocess_script_id
  const nested = kind === 'train' ? model.train_script : model.preprocess_script
  if (kind === 'preprocess' && model.preprocess_used === false && !id && !nested) return '未使用'
  const name = scriptName(model, kind, names)
  if (name === NOT_PROVIDED_TEXT) return name
  const version = nested?.version ?? (kind === 'train' ? model.train_script_version : model.preprocess_script_version)
  return version ? `${name} · ${version}` : name
}

function trainingFinished(model: ModelVersionSummary, finishedAt: Record<string, string | null>) {
  return formatDate(Object.prototype.hasOwnProperty.call(finishedAt, model.id) ? finishedAt[model.id] : null)
}

function isHealthy(model: ModelVersionSummary) {
  return healthCode(model) === 'HEALTHY'
}

function canMarkAbnormal(model: ModelVersionSummary) {
  const lifecycle = lifecycleCode(model)
  return !model.is_baseline
    && lifecycle !== null
    && ['READY', 'PUBLISHED', 'RETIRED'].includes(lifecycle)
    && healthCode(model) !== 'ABNORMAL'
}

function canRollbackTarget(model: ModelVersionSummary, current: ModelVersionSummary | null) {
  const lifecycle = lifecycleCode(model)
  return Boolean(current)
    && current!.id !== model.id
    && ['PUBLISHED', 'RETIRED'].includes(lifecycle ?? '')
    && Boolean(model.published_at)
    && isHealthy(model)
}

function lifecycleClass(model: ModelVersionSummary) {
  const code = lifecycleCode(model)
  return code === 'PUBLISHED' ? 'success' : code === 'FAILED' ? 'danger' : code === 'READY' ? 'info' : 'neutral'
}

function healthClass(model: ModelVersionSummary) {
  const code = healthCode(model)
  return code === 'HEALTHY' ? 'healthy' : code === 'ABNORMAL' ? 'abnormal' : 'unknown'
}

function dispatchRegistryUpdated(action: ActionType, modelVersionId: string) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(MODEL_REGISTRY_UPDATED_EVENT, {
    detail: { action, model_version_id: modelVersionId },
  }))
}

function ConfirmDialog({
  state,
  reason,
  reasonError,
  operationError,
  busy,
  onReasonChange,
  onCancel,
  onConfirm,
}: {
  state: ConfirmState
  reason: string
  reasonError: string | null
  operationError: string | null
  busy: boolean
  onReasonChange: (value: string) => void
  onCancel: () => void
  onConfirm: () => void
}) {
  const { action, model, current } = state
  const needsReason = action === 'publish' || action === 'rollback' || action === 'abnormal'
  const copy: Record<ActionType, { title: string; description: string; confirm: string }> = {
    publish: {
      title: '确认发布候选版本？',
      description: '发布会替换该模型类型的当前生产版本。',
      confirm: '确认发布',
    },
    offline: {
      title: '确认下线当前版本？',
      description: '下线只改变生命周期状态，不删除模型版本。',
      confirm: '确认下线',
    },
    abnormal: {
      title: '确认标记版本异常？',
      description: '标记异常会创建该版本的告警，并由后端处理后续状态。',
      confirm: '确认标记异常',
    },
    rollback: {
      title: '确认回滚到目标版本？',
      description: '回滚会把目标健康版本切换为当前生产版本。',
      confirm: '确认回滚',
    },
  }
  const reasonLabel = action === 'publish' ? '发布原因（必填）' : action === 'rollback' ? '回滚原因（必填）' : '异常原因（必填）'
  const reasonId = action === 'publish' ? 'publish-reason' : action === 'rollback' ? 'rollback-reason' : 'abnormal-reason'

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel() }}>
    <section aria-labelledby="model-operation-confirm-title" aria-modal="true" className="confirm-dialog" role="dialog">
      <p className="eyebrow">二次确认</p>
      <h2 id="model-operation-confirm-title">{copy[action].title}</h2>
      <p>{copy[action].description}</p>
      <div className="confirm-model">
        {action === 'publish' && <><span>候选版本：<b>{model.version}</b></span><span>当前生产版本：<b>{current?.version ?? '无当前生产版本'}</b></span></>}
        {action === 'rollback' && <><span>当前版本：<b>{current?.version ?? UNKNOWN_TEXT}</b></span><span>目标版本：<b>{model.version}</b></span></>}
        {action === 'offline' && <span>版本：<b>{model.version}</b></span>}
        {action === 'abnormal' && <span>版本：<b>{model.version}</b></span>}
      </div>
      {needsReason && <label htmlFor={reasonId}>{reasonLabel}</label>}
      {needsReason && <textarea
        aria-describedby={reasonError ? `${reasonId}-error` : undefined}
        aria-invalid={Boolean(reasonError)}
        aria-required="true"
        id={reasonId}
        onChange={(event) => onReasonChange(event.target.value)}
        placeholder={action === 'publish' ? '请输入发布原因' : action === 'rollback' ? '请输入回滚原因' : '请输入异常原因'}
        value={reason}
      />}
      {reasonError && <div className="alert-box error" id={`${reasonId}-error`} role="alert">{reasonError}</div>}
      {operationError && <div className="alert-box error" role="alert">{operationError}</div>}
      <div className="dialog-actions">
        <button className="secondary-button action-button" disabled={busy} onClick={onCancel} type="button">取消</button>
        <button className={action === 'publish' ? 'primary-button action-button' : 'danger-button action-button'} disabled={busy} onClick={onConfirm} type="button">{busy ? '处理中…' : copy[action].confirm}</button>
      </div>
    </section>
  </div>
}

function VersionDetails({
  model,
  baseline,
  baselineLoading,
  trainingFinishedAt,
  scriptNames,
  alerts,
  loading,
  error,
  canPublish,
  canOffline,
  canAbnormal,
  canRollback,
  actionLoading,
  onAction,
}: {
  model: ModelVersionDetail | ModelVersionSummary
  baseline: ModelVersionSummary | null
  baselineLoading: boolean
  trainingFinishedAt: Record<string, string | null>
  scriptNames: Record<string, string | null>
  alerts: ModelAlert[]
  loading: boolean
  error: string | null
  canPublish: boolean
  canOffline: boolean
  canAbnormal: boolean
  canRollback: boolean
  actionLoading: boolean
  onAction: (action: ActionType) => void
}) {
  const source = model as ModelVersionSummary
  const previousId = source.previous_healthy_version_id ?? null
  const reason = modelReason(source, alerts)
  const currentText = typeof source.is_current === 'boolean' ? (source.is_current ? '是 / 生产' : '否') : NOT_PROVIDED_TEXT

  return <div aria-label="版本详情" className="registry-panel version-detail">
    <div className="panel-heading">
      <div><p className="eyebrow">SELECTED VERSION / {source.version}</p><h2>{source.version}</h2><p>{MODEL_TYPE_NAMES[source.model_type]}</p></div>
      {loading && <span className="panel-count">加载中…</span>}
    </div>
    {error && <div className="alert-box error" role="alert">{error}</div>}
    <div className="detail-statuses">
      <span className={`status-badge ${lifecycleClass(source)}`} data-lifecycle={lifecycleCode(source) ?? 'UNKNOWN'}>生命周期：{lifecycleValue(source)}</span>
      <span className={`status-badge ${healthCode(source) === 'HEALTHY' ? 'success' : healthCode(source) === 'ABNORMAL' ? 'danger' : 'neutral'}`} data-health-status={healthCode(source)}>健康：{healthValue(source)}</span>
      {source.is_current && <span className="status-badge success">当前有效</span>}
    </div>
    <div className="detail-grid">
      <div><small>MAE</small><b>{metricValue(source.metrics, 'mae')}</b></div>
      <div><small>RMSE</small><b>{metricValue(source.metrics, 'rmse')}</b></div>
      <div><small>MAPE</small><b>{metricValue(source.metrics, 'mape')}</b></div>
      <div><small>R²</small><b>{metricValue(source.metrics, 'r2')}</b></div>
    </div>
    <dl className="detail-list model-detail-list">
      <div><dt>训练脚本</dt><dd>{scriptDetail(source, 'train', scriptNames)}</dd></div>
      <div><dt>训练完成</dt><dd>{trainingFinished(source, trainingFinishedAt)}</dd></div>
      <div><dt>生命周期</dt><dd>{lifecycleValue(source)}</dd></div>
      <div><dt>健康状态</dt><dd>{healthValue(source)}</dd></div>
      <div><dt>预处理</dt><dd>{scriptDetail(source, 'preprocess', scriptNames)}</dd></div>
      <div><dt>训练任务</dt><dd>{source.training_job_id ?? NOT_PROVIDED_TEXT}</dd></div>
      <div><dt>当前有效</dt><dd>{currentText}</dd></div>
      <div><dt>回滚基线</dt><dd>{baselineLoading ? '加载中…' : previousId ? baseline?.version ?? previousId : '无基线'}</dd></div>
    </dl>
    {reason && <div className="alert-box warning" role="alert"><b>异常原因：</b><span>{reason}</span></div>}
    <div className="row-actions version-actions">
      <button className="primary-button action-button" disabled={!canPublish || actionLoading} onClick={() => onAction('publish')} title={!canPublish ? '发布需要 READY 且 HEALTHY' : undefined} type="button">发布版本</button>
      <button className="secondary-button action-button" disabled={!canOffline || actionLoading} onClick={() => onAction('offline')} title={!canOffline ? '下线仅适用于 PUBLISHED 且当前有效版本' : undefined} type="button">下线</button>
      <button className="danger-button action-button" disabled={!canAbnormal || actionLoading} onClick={() => onAction('abnormal')} title={!canAbnormal ? '该版本已经是 ABNORMAL' : undefined} type="button">标记异常</button>
      <button className="text-button action-link-button" disabled={!canRollback || actionLoading} onClick={() => onAction('rollback')} title={!canRollback ? '回滚目标必须不是当前版本且健康' : undefined} type="button">回滚到所选版本</button>
    </div>
  </div>
}

export function ModelVersionsPage() {
  const [searchParams] = useSearchParams()
  const location = useLocation()
  const queryType = searchParams.get('model_type')
  const queryVersionId = location.hash.replace(/^#/, '')
    ? decodeURIComponent(location.hash.replace(/^#/, ''))
    : null
  const [models, setModels] = useState<ModelVersionSummary[]>([])
  const [alerts, setAlerts] = useState<ModelAlert[]>([])
  const [selectedType, setSelectedType] = useState<ModelTypeCode>(() => isModelType(queryType) ? queryType : 'electric_load')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [trainingFinishedAt, setTrainingFinishedAt] = useState<Record<string, string | null>>({})
  const [scriptNames, setScriptNames] = useState<Record<string, string | null>>({})
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [detail, setDetail] = useState<ModelVersionDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [baseline, setBaseline] = useState<ModelVersionSummary | null>(null)
  const [baselineLoading, setBaselineLoading] = useState(false)
  const [confirm, setConfirm] = useState<ConfirmState | null>(null)
  const [reason, setReason] = useState('')
  const [reasonError, setReasonError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)

  const mounted = useRef(true)
  const loadRequest = useRef(0)
  const detailRequest = useRef(0)
  const operationInFlight = useRef(false)
  const trainingCache = useRef<Record<string, string | null>>({})
  const trainingRequests = useRef(new Map<string, Promise<string | null>>())
  const scriptCache = useRef<Record<string, string | null>>({})
  const scriptRequests = useRef(new Map<string, Promise<string | null>>())

  useEffect(() => {
    if (isModelType(queryType)) setSelectedType(queryType)
    if (queryVersionId) {
      setSelectedId(queryVersionId)
      setDetail(null)
      setDetailError(null)
    }
  }, [queryType, queryVersionId])

  const fetchTrainingFinishedAt = useCallback((model: ModelVersionSummary, force = false) => {
    const jobId = model.training_job_id
    if (!jobId) return Promise.resolve(null)
    if (!force && Object.prototype.hasOwnProperty.call(trainingCache.current, model.id)) {
      return Promise.resolve(trainingCache.current[model.id])
    }
    let request = trainingRequests.current.get(jobId)
    if (!request || force) {
      request = apiClient.getTrainingJob(jobId).then((job) => typeof job.finished_at === 'string' ? job.finished_at : null).catch(() => null)
      trainingRequests.current.set(jobId, request)
    }
    return request.then((finishedAt) => {
      trainingCache.current[model.id] = finishedAt
      return finishedAt
    })
  }, [])

  const fetchScriptName = useCallback((id: string) => {
    if (Object.prototype.hasOwnProperty.call(scriptCache.current, id)) return Promise.resolve(scriptCache.current[id])
    const existing = scriptRequests.current.get(id)
    if (existing) return existing
    const request = apiClient.getScript(id).then((script) => typeof script.name === 'string' && script.name.trim() ? script.name : null).catch(() => null)
    scriptRequests.current.set(id, request)
    return request.then((name) => {
      scriptCache.current[id] = name
      return name
    })
  }, [])

  const loadAuxiliary = useCallback(async (nextModels: ModelVersionSummary[]): Promise<AuxiliaryData> => {
    const visibleModels = nextModels.filter((model) => !model.is_baseline)
    const finishedEntries = await Promise.all(visibleModels.map(async (model) => [model.id, await fetchTrainingFinishedAt(model, true)] as const))
    const scriptIds = [...new Set(visibleModels.flatMap((model) => {
      const ids: string[] = []
      if (!nestedScriptName(model, 'train') && model.train_script_id) ids.push(model.train_script_id)
      if (!nestedScriptName(model, 'preprocess') && model.preprocess_script_id) ids.push(model.preprocess_script_id)
      return ids
    }))]
    const scriptEntries = await Promise.all(scriptIds.map(async (id) => [id, await fetchScriptName(id)] as const))
    return {
      trainingFinishedAt: Object.fromEntries(finishedEntries),
      scriptNames: Object.fromEntries(scriptEntries),
    }
  }, [fetchScriptName, fetchTrainingFinishedAt])

  const load = useCallback(async (initial = false): Promise<RefreshResult> => {
    const requestId = ++loadRequest.current
    if (initial) setLoading(true); else setRefreshing(true)
    setError(null)
    const [modelsResult, alertsResult, auditResult] = await Promise.allSettled([
      apiClient.listModels(),
      apiClient.listAlerts(),
      apiClient.listAuditEvents({ page: 1, page_size: 10 }),
    ])
    if (!mounted.current || requestId !== loadRequest.current) return { ok: false, stale: true }

    const failures: string[] = []
    let nextModels: ModelVersionSummary[] | null = null
    if (modelsResult.status === 'fulfilled') {
      nextModels = [...modelsResult.value]
      setModels(nextModels)
      try {
        const auxiliary = await loadAuxiliary(nextModels)
        if (!mounted.current || requestId !== loadRequest.current) return { ok: false, stale: true }
        trainingCache.current = { ...trainingCache.current, ...auxiliary.trainingFinishedAt }
        scriptCache.current = { ...scriptCache.current, ...auxiliary.scriptNames }
        setTrainingFinishedAt(auxiliary.trainingFinishedAt)
        setScriptNames(auxiliary.scriptNames)
      } catch (auxiliaryError) {
        // Auxiliary records are rendered as unknown when unavailable. The
        // primary model response remains usable and the page can be retried.
        failures.push(`关联资源刷新失败：${errorMessage(auxiliaryError)}`)
      }
    } else {
      failures.push(`模型刷新失败：${errorMessage(modelsResult.reason)}`)
    }
    if (alertsResult.status === 'fulfilled') setAlerts([...alertsResult.value])
    else failures.push(`告警刷新失败：${errorMessage(alertsResult.reason)}`)
    if (auditResult.status !== 'fulfilled') failures.push(`审计刷新失败：${errorMessage(auditResult.reason)}`)

    if (!mounted.current || requestId !== loadRequest.current) return { ok: false, stale: true }
    if (failures.length) setError(failures.join('；'))
    setLoading(false)
    setRefreshing(false)
    return { ok: failures.length === 0, stale: false }
  }, [loadAuxiliary])

  useEffect(() => {
    mounted.current = true
    void load(true)
    return () => {
      mounted.current = false
      loadRequest.current += 1
      detailRequest.current += 1
    }
  }, [load])

  const versions = useMemo(() => models.filter((model) => model.model_type === selectedType && !model.is_baseline), [models, selectedType])
  // Render the first source-backed version immediately after the list arrives;
  // an explicit selection still wins, while a type switch resets it.
  const selected = useMemo(() => versions.find((model) => model.id === selectedId) ?? versions[0] ?? null, [selectedId, versions])
  const activeSelectedId = selected?.id ?? null
  const current = useMemo(() => versions.find((model) => model.is_current) ?? null, [versions])

  const openDetails = useCallback(async (model: ModelVersionSummary) => {
    const requestId = ++detailRequest.current
    setSelectedId(model.id)
    setDetail(null)
    setDetailError(null)
    setBaseline(null)
    setBaselineLoading(false)
    setDetailLoading(true)
    const detailPromise = apiClient.getModel(model.id)
    const trainingPromise = fetchTrainingFinishedAt(model).then((finishedAt) => {
      if (mounted.current && requestId === detailRequest.current) setTrainingFinishedAt((previous) => ({ ...previous, [model.id]: finishedAt }))
    })
    try {
      const modelDetail = await detailPromise
      if (!mounted.current || requestId !== detailRequest.current) return
      setDetail(modelDetail)
      const previousId = modelDetail.previous_healthy_version_id ?? model.previous_healthy_version_id ?? null
      if (previousId) {
        setBaselineLoading(true)
        try {
          const baselineModel = await apiClient.getModel(previousId)
          if (mounted.current && requestId === detailRequest.current) setBaseline(baselineModel)
        } catch {
          // The ID itself remains visible as the known source value; a failed
          // second query must not make the selected model detail disappear.
        } finally {
          if (mounted.current && requestId === detailRequest.current) setBaselineLoading(false)
        }
      }
      await trainingPromise
    } catch (reasonValue) {
      if (mounted.current && requestId === detailRequest.current) setDetailError(errorMessage(reasonValue))
    } finally {
      if (mounted.current && requestId === detailRequest.current) setDetailLoading(false)
    }
  }, [fetchTrainingFinishedAt])

  const selectType = (value: string) => {
    if (!isModelType(value)) return
    setSelectedType(value)
    setSelectedId(null)
    setDetail(null)
    setBaseline(null)
    setDetailError(null)
  }

  const openAction = (action: ActionType) => {
    if (!selected || actionLoading) return
    if (action === 'publish' && !(lifecycleCode(selected) === 'READY' && isHealthy(selected))) return
    if (action === 'offline' && !(lifecycleCode(selected) === 'PUBLISHED' && selected.is_current === true)) return
    if (action === 'abnormal' && !canMarkAbnormal(selected)) return
    if (action === 'rollback' && !canRollbackTarget(selected, current)) return
    setReason('')
    setReasonError(null)
    setError(null)
    setConfirm({ action, model: selected, current })
  }

  const execute = async () => {
    const operation = confirm
    if (!operation || operationInFlight.current) return
    const needsReason = operation.action === 'publish' || operation.action === 'rollback' || operation.action === 'abnormal'
    const trimmedReason = reason.trim()
    if (needsReason && !trimmedReason) {
      setReasonError(operation.action === 'publish' ? '发布原因不能为空' : operation.action === 'rollback' ? '回滚原因不能为空' : '异常原因不能为空')
      return
    }
    if (operation.action === 'rollback' && !canRollbackTarget(operation.model, operation.current)) {
      setReasonError('回滚目标必须是已发布历史版本、已发布且健康')
      return
    }
    if (operation.action === 'abnormal' && !canMarkAbnormal(operation.model)) {
      setReasonError('只有 READY、PUBLISHED 或 RETIRED 的非基线版本可以标记异常')
      return
    }

    operationInFlight.current = true
    setActionLoading(true)
    setError(null)
    setSuccess(null)
    try {
      if (operation.action === 'publish') {
        await apiClient.publishModel(operation.model.id, {
          confirmed: true,
          reason: trimmedReason,
          idempotency_key: createIdempotencyKey('publish'),
        })
      } else if (operation.action === 'offline') {
        await apiClient.offlineModel(operation.model.id)
      } else if (operation.action === 'abnormal') {
        await apiClient.markModelAbnormal(operation.model.id, trimmedReason)
      } else {
        await apiClient.rollbackModel(operation.current!.id, {
          target_version_id: operation.model.id,
          reason: trimmedReason,
          idempotency_key: createIdempotencyKey('rollback'),
        })
      }
      if (!mounted.current) return
      const actionLabel = operation.action === 'publish' ? '发布' : operation.action === 'offline' ? '下线' : operation.action === 'abnormal' ? '标记异常' : '回滚'
      setConfirm(null)
      setReason('')
      setReasonError(null)
      setDetail(null)
      setBaseline(null)
      const refreshResult = await load()
      if (!mounted.current) return
      setSuccess(`${operation.model.version} ${actionLabel}成功，模型、告警和审计已刷新。`)
      if (!refreshResult.ok && !refreshResult.stale) setError('操作已成功，但部分状态刷新失败，请手动刷新。')
      dispatchRegistryUpdated(operation.action, operation.model.id)
    } catch (reasonValue) {
      if (mounted.current) {
        const message = errorMessage(reasonValue)
        if (operation.action === 'abnormal') {
          await load()
        }
        if (mounted.current) setError(message)
      }
    } finally {
      operationInFlight.current = false
      if (mounted.current) setActionLoading(false)
    }
  }

  const canPublish = Boolean(selected && lifecycleCode(selected) === 'READY' && isHealthy(selected))
  const canOffline = Boolean(selected && lifecycleCode(selected) === 'PUBLISHED' && selected.is_current === true)
  const canAbnormal = Boolean(selected && canMarkAbnormal(selected))
  const canRollback = Boolean(selected && canRollbackTarget(selected, current))

  return <div className="page-content model-registry-page">
    <div className="registry-header">
      <div><p className="eyebrow">MODEL REGISTRY / LIFECYCLE</p><h1 className="page-title">模型版本管理</h1><p className="page-description">按模型类型管理版本，生命周期和健康状态独立展示。</p></div>
      <button className="secondary-button action-button" disabled={loading || refreshing || actionLoading} onClick={() => void load()} type="button">{refreshing ? '同步中…' : '刷新列表 ↻'}</button>
    </div>
    {error && !confirm && <div className="alert-box error" role="alert"><span>{error}</span><button type="button" onClick={() => void load()}>重试</button></div>}
    {success && <div className="alert-box success" role="status"><span>{success}</span><button type="button" onClick={() => setSuccess(null)}>关闭</button></div>}

    <div className="model-type-tabs">
      <label className="tab-caption" htmlFor="model-type-select">模型类型</label>
      <select id="model-type-select" onChange={(event) => selectType(event.target.value)} value={selectedType}>
        {MODEL_TYPE_CODES.map((code) => <option key={code} value={code}>{MODEL_TYPE_NAMES[code]} / {code}</option>)}
      </select>
      <span className="sub">当前类型：<b className="mono">{selectedType}</b> · 可管理版本 <b className="mono">{versions.length}</b> 个</span>
    </div>

    <div className="registry-lower-grid model-version-layout">
      <section className="registry-panel version-list">
        <div className="panel-heading"><div><p className="eyebrow">{selectedType.toUpperCase()}</p><h2>{MODEL_TYPE_NAMES[selectedType]} · 版本列表</h2></div><span className="panel-count">{loading ? '加载中…' : `${versions.length} 个版本`}</span></div>
        {loading && <div className="loading-state"><span className="spinner" />正在加载模型版本…</div>}
        {!loading && !versions.length && <div className="empty-state"><strong>暂无{MODEL_TYPE_NAMES[selectedType]}版本</strong><span>完成训练并保存 READY 版本后会显示在这里。</span></div>}
        {!loading && versions.length > 0 && <div className="table-wrap registry-table-wrap"><table aria-label="版本列表" className="registry-table model-version-table"><thead><tr><th>版本号</th><th>版本标题</th><th>训练完成时间</th><th>训练脚本名称</th><th>生命周期</th><th>健康状态</th></tr></thead><tbody>{versions.map((model) => <tr className={model.id === activeSelectedId ? 'current-row' : ''} data-model-version-id={model.id} key={model.id} onClick={(event) => {
          // The version number remains an explicit keyboard target, while the
          // rest of the source-backed row is also selectable like the
          // prototype's version list. Do not let lifecycle action buttons
          // change the selected detail as a side effect of bubbling.
          const target = event.target as HTMLElement | null
          if (target?.closest('button, a, input, select, textarea')) return
          void openDetails(model)
        }}>
          <td><button aria-label={model.version} className="version-link" onClick={() => void openDetails(model)} type="button"><b>{model.version}</b></button></td>
          <td>{versionTitle(model)}</td>
          <td>{trainingFinished(model, trainingFinishedAt)}</td>
          <td>{scriptName(model, 'train', scriptNames)}</td>
          <td><span className={`status-badge ${lifecycleClass(model)}`} data-lifecycle={lifecycleCode(model) ?? 'UNKNOWN'}>{lifecycleValue(model)}</span></td>
          <td><span className={`health-indicator ${healthClass(model)}`} data-health-status={healthCode(model)}><i />{healthValue(model)}</span></td>
        </tr>)}</tbody></table></div>}
      </section>

      {selected ? <VersionDetails
        alerts={alerts}
        baseline={baseline}
        baselineLoading={baselineLoading}
        canAbnormal={canAbnormal}
        canOffline={canOffline}
        canPublish={canPublish}
        canRollback={canRollback}
        error={detailError}
        loading={detailLoading}
        model={detail ?? selected}
        onAction={openAction}
        actionLoading={actionLoading}
        scriptNames={scriptNames}
        trainingFinishedAt={trainingFinishedAt}
      /> : <section aria-label="版本详情" className="registry-panel version-detail"><div className="empty-state"><strong>请选择版本</strong><span>选择版本后查看详情和可用操作。</span></div></section>}
    </div>

    {confirm && <ConfirmDialog
      busy={actionLoading}
      onCancel={() => { if (!actionLoading) { setConfirm(null); setReason(''); setReasonError(null) } }}
      onConfirm={() => void execute()}
      onReasonChange={(value) => { setReason(value); if (reasonError) setReasonError(null); if (error) setError(null) }}
      operationError={error}
      reason={reason}
      reasonError={reasonError}
      state={confirm}
    />}
  </div>
}
