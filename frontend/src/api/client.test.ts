import { describe, expect, it, vi } from 'vitest'
import { ApiClient, ApiError, createIdempotencyKey } from './client'

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

const model = {
  id: 'model-1',
  model_type: 'electric_load',
  version: 'v1',
  status: 'READY',
  health_status: 'UNKNOWN',
  is_baseline: false,
  is_current: false,
  created_at: '2025-01-01T00:00:00Z',
}

function requestBody(call: unknown[]): Record<string, unknown> {
  const init = call[1] as RequestInit
  return JSON.parse(init.body as string) as Record<string, unknown>
}

describe('ApiClient contracts', () => {
  it('uses formal /health without duplicating /api', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' }))
    await new ApiClient({ fetchImpl }).getHealth()
    expect(fetchImpl.mock.calls[0][0]).toBe('/health')
  })

  it('derives MCP availability from both formal OpenAPI POST declarations', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({
      paths: {
        '/api/mcp/predict': { post: { responses: { '200': {} } } },
        '/api/mcp/mark_model_abnormal': { post: { responses: { '200': {} } } },
      },
    }))
    const capabilities = await new ApiClient({ fetchImpl }).getMcpCapabilities()
    expect(fetchImpl.mock.calls[0][0]).toBe('/openapi.json')
    expect(capabilities).toEqual({ available: true, predict: true, markModelAbnormal: true })

    const missingRoute = await new ApiClient({
      fetchImpl: vi.fn().mockResolvedValue(jsonResponse({ paths: {
        '/api/mcp/predict': { post: {} },
      } })),
    }).getMcpCapabilities()
    expect(missingRoute).toEqual({ available: false, predict: true, markModelAbnormal: false })
  })

  it('uploads only the declared multipart file field', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ id: 'dataset-1' }))
    const file = new File(['time,value\n2025-01-01,1'], 'load.csv', { type: 'text/csv' })
    await new ApiClient({ fetchImpl }).uploadDataset(file, { model_type: 'electric_load' })

    const body = fetchImpl.mock.calls[0][1].body as FormData
    expect([...body.keys()]).toEqual(['file'])
    expect(body.get('file')).toBe(file)
  })

  it('uses the preprocessing task resource endpoint', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ id: 'task-1', status: 'SKIPPED' }))
    await new ApiClient({ fetchImpl }).getPreprocessingTask('task/1')
    expect(fetchImpl.mock.calls[0][0]).toBe('/api/preprocessing-tasks/task%2F1')
  })

  it('passes incremental log since, limit, and cursor parameters', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ job_id: 'job-1', items: [], next_cursor: 'next' }))
    await new ApiClient({ fetchImpl }).getTrainingJobLogs('job-1', {
      since: '2025-01-01T00:00:00Z', limit: 25, cursor: 'cursor-1',
    })
    expect(fetchImpl.mock.calls[0][0]).toBe('/api/training-jobs/job-1/logs?since=2025-01-01T00%3A00%3A00Z&limit=25&cursor=cursor-1')
  })

  it('calls training cancellation and alert acknowledgement contracts', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ id: 'job-1', status: 'CANCELLED' }))
      .mockResolvedValueOnce(jsonResponse({ id: 'alert-1', model_type: 'electric_load', status: 'ACTIVE' }))
      .mockResolvedValueOnce(jsonResponse({ id: 'alert-1', model_type: 'electric_load', status: 'ACKNOWLEDGED', statistics: { total: 1, active: 0, acknowledged: 1, resolved: 0 } }))
    const client = new ApiClient({ fetchImpl })
    await client.cancelTrainingJob('job-1')
    await client.getAlert('alert-1')
    await client.acknowledgeAlert('alert-1')

    expect(fetchImpl.mock.calls[0][0]).toBe('/api/training-jobs/job-1/cancel')
    expect(fetchImpl.mock.calls[1][0]).toBe('/api/alerts/alert-1')
    expect(fetchImpl.mock.calls[2][0]).toBe('/api/alerts/alert-1/acknowledge')
    expect(requestBody(fetchImpl.mock.calls[2])).toEqual({ confirmed: true })
  })

  it('keeps list models and alerts array-compatible while exposing page metadata', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ items: [model], page: 1, page_size: 1, total: 2, has_next: true }))
      .mockResolvedValueOnce(jsonResponse({ items: [], page: 1, page_size: 10, total: 0, next_cursor: null, statistics: { total: 0, active: 0, acknowledged: 0, resolved: 0 } }))
    const client = new ApiClient({ fetchImpl })
    const models = await client.listModels({ page: 1, page_size: 1 })
    const alerts = await client.listAlerts({ page: 1, page_size: 10 })
    expect(Array.isArray(models)).toBe(true)
    expect(models.items).toBe(models)
    expect(models.total).toBe(2)
    expect(models.has_next).toBe(true)
    expect(alerts.items).toBe(alerts)
    expect(alerts.total).toBe(0)
  })

  it('adapts audit filters and canonical pagination while validating event fields', async () => {
    const event = {
      id: 'event-1', occurred_at: '2025-01-01T00:00:00Z', event_type: 'MODEL_SAVED',
      object_type: 'MODEL_VERSION', object_id: 'model-1', model_type: 'electric_load',
      model_version_id: 'model-1', training_job_id: null, operator_type: null,
      operator_id: null, operator_name: null, result: 'SUCCESS', message: 'saved',
      request_id: null, correlation_id: null, metadata: {},
    }
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ items: [event], page: 2, page_size: 10, total: 21, has_next: true }))
    const result = await new ApiClient({ fetchImpl }).listAuditEvents({
      page: 2, page_size: 10, from: '2025-01-01T00:00:00Z', to: '2025-01-02T00:00:00Z',
      model_type: 'electric_load', event_type: 'MODEL_SAVED', object_type: 'MODEL_VERSION',
      result: 'SUCCESS', query: 'saved',
    })
    expect(fetchImpl.mock.calls[0][0]).toContain('/api/audit-events?')
    expect(fetchImpl.mock.calls[0][0]).toContain('from=2025-01-01T00%3A00%3A00Z')
    expect(fetchImpl.mock.calls[0][0]).toContain('page=2')
    expect(result).toMatchObject({ page: 2, page_size: 10, total: 21, has_next: true })
    expect(result.items[0]).toMatchObject({ id: 'event-1', object_type: 'MODEL_VERSION' })
  })

  it('sends canonical lifecycle commands and rejects an empty rollback target', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ operation: 'publish', model, record: null }))
      .mockResolvedValueOnce(jsonResponse({ operation: 'rollback', model, rollback: null }))
    const client = new ApiClient({ fetchImpl })
    await client.publishModel('model-1', { confirmed: true, message: 'legacy', idempotency_key: 'publish-key' })
    await client.rollbackModel('model-1', { target_version_id: 'model-0', reason: '故障回滚', idempotency_key: 'rollback-key' })

    expect(requestBody(fetchImpl.mock.calls[0])).toEqual({
      confirmed: true, reason: 'legacy', idempotency_key: 'publish-key',
    })
    expect(requestBody(fetchImpl.mock.calls[1])).toEqual({
      target_version_id: 'model-0', reason: '故障回滚', idempotency_key: 'rollback-key',
    })
    await expect(client.rollbackModel('model-1', { reason: '缺少目标' })).rejects.toMatchObject({
      code: 'ROLLBACK_TARGET_REQUIRED', status: 400,
    })
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('preserves FastAPI status, code, and detail payloads', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({
      detail: { code: 'ROLLBACK_TARGET_INVALID', message: '目标版本不可用', details: { target: 'bad' } },
    }, 409))
    const error = await new ApiClient({ fetchImpl }).get('models/model-1').catch((reason) => reason as ApiError)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 409, code: 'ROLLBACK_TARGET_INVALID', message: '目标版本不可用' })
    expect((error as ApiError).details).toMatchObject({ detail: { details: { target: 'bad' } } })
  })

  it('generates explicit idempotency keys when callers omit them', () => {
    expect(createIdempotencyKey('publish')).toMatch(/^publish:/)
  })
})
