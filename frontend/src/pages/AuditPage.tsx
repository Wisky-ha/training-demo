import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, apiClient } from '../api'
import type { AuditEvent, ListAuditEventsParams } from '../types/contracts'

const PAGE_SIZE = 50
const UNKNOWN_TEXT = '未知'

export const AUDIT_EVENT_FILTERS = [
  { value: 'all', label: '全部' },
  { value: 'MODEL_PUBLISHED', label: '模型发布' },
  { value: 'MODEL_ROLLBACK_SUCCEEDED', label: '模型回滚' },
  { value: 'TRAINING_SUCCEEDED', label: '训练完成' },
  { value: 'DATASET_VALIDATED', label: '数据校验' },
  { value: 'MODEL_MARKED_ABNORMAL', label: '标记异常' },
  { value: 'ALERT_ACKNOWLEDGED', label: '告警确认' },
] as const

type AuditEventFilterValue = (typeof AUDIT_EVENT_FILTERS)[number]['value']

type ObjectLink = {
  href: string
  id: string
} | null

function errorMessage(reason: unknown): string {
  return reason instanceof ApiError
    ? reason.message
    : reason instanceof Error
      ? reason.message
      : '请求失败，请稍后重试'
}

function nonEmpty(value: string | null | undefined): string | null {
  const text = value?.trim()
  return text ? text : null
}

function displayValue(value: string | null | undefined): string {
  return nonEmpty(value) ?? UNKNOWN_TEXT
}

function operatorValue(event: AuditEvent): string {
  return nonEmpty(event.operator_name) ?? nonEmpty(event.operator_id) ?? UNKNOWN_TEXT
}

function metadataId(event: AuditEvent, key: string): string | null {
  const value = event.metadata?.[key]
  return typeof value === 'string' ? nonEmpty(value) : null
}

function objectIdValue(event: AuditEvent, objectType: string): string | null {
  const objectId = nonEmpty(event.object_id)
  if (objectId) return objectId
  if (objectType === 'MODEL_VERSION') return nonEmpty(event.model_version_id)
  if (objectType === 'TRAINING_JOB') return nonEmpty(event.training_job_id)
  return null
}

function objectLink(event: AuditEvent): ObjectLink {
  const objectType = nonEmpty(event.object_type)?.toUpperCase() ?? ''
  const id = objectIdValue(event, objectType)
  if (!id) return null

  const modelType = nonEmpty(event.model_type)
  if (objectType === 'MODEL_VERSION') {
    const query = modelType ? `?model_type=${encodeURIComponent(modelType)}` : ''
    return { href: `/models${query}#${encodeURIComponent(id)}`, id }
  }
  if (objectType === 'TRAINING_JOB') {
    return { href: `/workflow/train?training_job_id=${encodeURIComponent(id)}`, id }
  }
  if (objectType === 'DATASET') {
    return { href: `/workflow/upload?dataset_id=${encodeURIComponent(id)}`, id }
  }
  if (objectType === 'PREPROCESSING_TASK') {
    return { href: `/workflow/preprocess?preprocessing_task_id=${encodeURIComponent(id)}`, id }
  }
  if (objectType === 'DATASET_SPLIT') {
    const datasetId = metadataId(event, 'dataset_id')
    const datasetQuery = datasetId ? `&dataset_id=${encodeURIComponent(datasetId)}` : ''
    return { href: `/workflow/split?split_id=${encodeURIComponent(id)}${datasetQuery}`, id }
  }
  return null
}

function eventClass(eventType: string): string {
  const value = eventType.toUpperCase()
  if (value.includes('FAILED') || value.includes('ABNORMAL')) return 'danger'
  if (value.includes('OFFLINED')) return 'warning'
  return 'neutral'
}

function resultClass(result: string): string {
  const value = result.toUpperCase()
  if (value === 'SUCCEEDED' || value === 'SUCCESS' || value === 'PASSED') return 'success'
  if (value === 'FAILED' || value === 'FAILURE') return 'danger'
  return 'neutral'
}

function ObjectValue({ event }: { event: AuditEvent }) {
  const objectType = displayValue(event.object_type)
  const link = objectLink(event)
  const objectId = link?.id ?? objectIdValue(event, objectType.toUpperCase())

  return (
    <div className="audit-object">
      <span className="audit-object-type">{objectType}</span>
      <span className="audit-object-value">
        {objectId ? (
          link ? <Link className="audit-object-link" to={link.href}>{objectId}</Link> : <span className="audit-object-id">{objectId}</span>
        ) : <span className="audit-object-id">{UNKNOWN_TEXT}</span>}
      </span>
    </div>
  )
}

export function AuditPage() {
  const [eventFilter, setEventFilter] = useState<AuditEventFilterValue>('all')
  const [page, setPage] = useState(1)
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [total, setTotal] = useState(0)
  const [hasNext, setHasNext] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)

  const selectedEventType = useMemo(
    () => AUDIT_EVENT_FILTERS.find((item) => item.value === eventFilter)?.value === 'all'
      ? undefined
      : eventFilter,
    [eventFilter],
  )

  useEffect(() => {
    let active = true
    const params: ListAuditEventsParams = {
      page,
      page_size: PAGE_SIZE,
      ...(selectedEventType ? { event_type: selectedEventType } : {}),
    }

    setLoading(true)
    setError(null)
    setEvents([])
    setTotal(0)
    setHasNext(false)

    // Starting from a resolved promise also turns an unexpected synchronous
    // adapter error into the same page-level error state as a rejected fetch.
    void Promise.resolve()
      .then(() => apiClient.listAuditEvents(params))
      .then((response) => {
        if (!active) return
        setEvents(response.items)
        setTotal(response.total)
        setHasNext(response.has_next)
      })
      .catch((reason: unknown) => {
        if (active) setError(errorMessage(reason))
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => { active = false }
  }, [page, reloadKey, selectedEventType])

  const handleFilterChange = (value: string) => {
    if (!(AUDIT_EVENT_FILTERS as readonly { value: string }[]).some((item) => item.value === value)) return
    setEventFilter(value as AuditEventFilterValue)
    setPage(1)
  }

  const retry = () => setReloadKey((value) => value + 1)
  const previousDisabled = loading || Boolean(error) || page <= 1
  const nextDisabled = loading || Boolean(error) || !hasNext

  return (
    <div className="page-content audit-page">
      <div className="audit-header">
        <div>
          <p className="eyebrow">AUDIT TRAIL / APPEND ONLY</p>
          <h1 className="page-title">审计记录</h1>
        </div>
        <div className="audit-filter">
          <label htmlFor="audit-event-filter">事件类型</label>
          <select
            id="audit-event-filter"
            onChange={(event) => handleFilterChange(event.target.value)}
            value={eventFilter}
          >
            {AUDIT_EVENT_FILTERS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </div>
      </div>

      {error && (
        <div className="alert-box error audit-error" role="alert">
          <span>{error}</span>
          <button onClick={retry} type="button">重试</button>
        </div>
      )}

      <div aria-busy={loading} className="table-wrap audit-table-wrap">
        <table aria-label="审计记录表" className="audit-table">
          <thead>
            <tr><th scope="col">时间</th><th scope="col">事件类型</th><th scope="col">对象</th><th scope="col">操作人</th><th scope="col">结果</th></tr>
          </thead>
          <tbody>
            {loading && <tr><td colSpan={5}><div className="loading-state" role="status"><span className="spinner" />正在加载审计事件…</div></td></tr>}
            {!loading && error && <tr><td colSpan={5}><div className="audit-table-state">无法加载审计事件</div></td></tr>}
            {!loading && !error && events.length === 0 && <tr><td colSpan={5}><div className="empty-state compact" role="status">暂无审计事件</div></td></tr>}
            {!loading && !error && events.map((event) => {
              const eventType = displayValue(event.event_type)
              const result = displayValue(event.result)
              return (
                <tr data-event-id={event.id} key={event.id}>
                  <td><time className="audit-time" dateTime={nonEmpty(event.occurred_at) ?? undefined}>{displayValue(event.occurred_at)}</time></td>
                  <td>
                    <div className="audit-event-cell">
                      <span className={`audit-event-type ${eventClass(eventType)}`}>{eventType}</span>
                      {nonEmpty(event.message) && <small className="audit-message">{nonEmpty(event.message)}</small>}
                    </div>
                  </td>
                  <td><ObjectValue event={event} /></td>
                  <td>{operatorValue(event)}</td>
                  <td><span className={`audit-result ${resultClass(result)}`}>{result}</span></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <nav aria-label="审计记录分页" className="audit-pagination">
        <span aria-live="polite">共 {total} 条 · 第 {page} 页</span>
        <div className="audit-pagination-actions">
          <button disabled={previousDisabled} onClick={() => setPage((value) => value - 1)} type="button">上一页</button>
          <button disabled={nextDisabled} onClick={() => setPage((value) => value + 1)} type="button">下一页</button>
        </div>
      </nav>
    </div>
  )
}
