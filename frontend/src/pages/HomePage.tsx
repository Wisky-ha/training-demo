import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiClient, ApiError } from '../api'
import { MODEL_TYPE_CODES, MODEL_TYPE_NAMES, type ModelAlert, type ModelTypeCode, type ModelVersionSummary } from '../types/contracts'

function formatDate(value?: string | null) {
  if (!value) return '尚未训练'
  return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
}

function statusLabel(model?: ModelVersionSummary) {
  if (!model) return '未训练'
  if (model.is_abnormal || model.health_status?.toLowerCase() === 'abnormal') return '异常'
  return ({ PUBLISHED: '已发布', READY: '待发布', TRAINING: '训练中', FAILED: '失败', DRAFT: '草稿' } as Record<string, string>)[model.status] ?? model.status
}

export function HomePage() {
  const [models, setModels] = useState<ModelVersionSummary[]>([])
  const [alerts, setAlerts] = useState<ModelAlert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    Promise.all([apiClient.listModels(), apiClient.listAlerts({ page: 1, page_size: 100 })])
      .then(([versions, activeAlerts]) => {
        if (!alive) return
        setModels(versions)
        setAlerts(activeAlerts.filter((item) => item.status === 'ACTIVE'))
      })
      .catch((reason: unknown) => {
        if (!alive) return
        setError(reason instanceof ApiError ? reason.message : '无法加载模型概览，请稍后重试')
      })
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  const byType = useMemo(() => {
    const result = new Map<ModelTypeCode, ModelVersionSummary[]>()
    MODEL_TYPE_CODES.forEach((code) => result.set(code, []))
    models.forEach((model) => result.get(model.model_type)?.push(model))
    return result
  }, [models])

  return (
    <div className="page-content">
      <section className="hero dashboard-hero">
        <div className="hero-copy">
          <div className="hero-kicker"><span /> MODEL STUDIO / OVERVIEW</div>
          <p className="eyebrow">MODEL TRAINING VISUALIZATION PLATFORM</p>
          <h1 className="page-title">训练状态，一目了然。</h1>
          <p className="page-description">从数据上传到模型发布，在同一个工作台追踪每个模型的版本、健康状态与训练结果。</p>
          <Link className="workflow-link" to="/workflow/model-type">开始一次训练 <span aria-hidden="true">→</span></Link>
        </div>
      </section>

      {error && <div className="alert-box error" role="alert">{error}<button type="button" onClick={() => window.location.reload()}>重试</button></div>}
      {alerts.length > 0 && <div className="alert-box warning" role="status">⚠ 当前有 {alerts.length} 个模型异常告警，请检查模型健康状态。</div>}

      <div className="section-heading dashboard-heading">
        <div><h2>模型工作台</h2><p>Three model lines / production overview</p></div>
        <span className="muted-caption">{loading ? '同步中…' : `共 ${models.length} 个版本`}</span>
      </div>
      <section className="model-card-grid" aria-label="模型类型概览">
        {MODEL_TYPE_CODES.map((code) => {
          const versions = byType.get(code) ?? []
          const current = versions.find((version) => version.is_current) ?? versions.find((version) => version.status === 'PUBLISHED')
          const backup = current?.previous_healthy_version_id
            ? versions.find((version) => version.id === current.previous_healthy_version_id)
            : versions.find((version) => version.id !== current?.id && ['READY', 'PUBLISHED'].includes(version.status))
          const alert = alerts.find((item) => item.model_type === code)
          return (
            <article className={`model-card${alert ? ' model-card-alert' : ''}`} key={code}>
              <div className="model-card-top"><span className="model-symbol">{code === 'electric_load' ? '⚡' : code === 'heating_cooling_load' ? '◒' : '⌁'}</span><span className={`status-badge ${alert ? 'danger' : current ? 'success' : 'neutral'}`}>{alert ? '异常告警' : statusLabel(current)}</span></div>
              <h3>{MODEL_TYPE_NAMES[code]}</h3>
              <p className="model-code">{code}</p>
              <dl className="model-facts">
                <div><dt>当前版本</dt><dd>{current?.version ?? '—'}</dd></div>
                <div><dt>训练时间</dt><dd>{formatDate(current?.published_at ?? current?.created_at)}</dd></div>
                <div><dt>回滚备份</dt><dd>{backup?.version ?? '无可用备份'}</dd></div>
              </dl>
              {alert && <p className="card-alert">{alert.reason}</p>}
              <div className="card-actions"><Link className="secondary-button" to={`/workflow/model-type?model=${code}`}>开始训练</Link><Link className="text-button" to="/models">查看版本 <span aria-hidden="true">↗</span></Link></div>
            </article>
          )
        })}
      </section>
      {!loading && models.length === 0 && <div className="empty-state">还没有模型版本。选择一个模型类型开始训练。</div>}
    </div>
  )
}
