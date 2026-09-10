import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { apiClient, ApiError, type CompatibleArrayResponse } from '../api'
import type { AuditEvent, ModelAlert, ModelVersionSummary, TrainingJob } from '../types/contracts'
import { HomePage } from './HomePage'

const modelFixture = (overrides: Partial<ModelVersionSummary> = {}): ModelVersionSummary => ({
  id: 'model-1',
  model_type: 'electric_load',
  version: 'v1',
  status: 'READY',
  health_status: 'UNKNOWN',
  is_baseline: false,
  is_current: false,
  is_abnormal: false,
  is_rollback_available: false,
  metrics: null,
  training_job_id: null,
  train_script_id: null,
  train_script_version: null,
  preprocess_script_id: null,
  preprocess_script_version: null,
  previous_healthy_version_id: null,
  train_script: null,
  preprocess_script: null,
  preprocess_used: false,
  feature_columns: [],
  time_column: null,
  target_column: null,
  created_at: '2026-01-01T00:00:00Z',
  published_at: null,
  ...overrides,
})

const alertFixture = (overrides: Partial<ModelAlert> = {}): ModelAlert => ({
  id: 'alert-1',
  model_type: 'electric_load',
  model_version_id: 'model-1',
  reason: '健康检查异常',
  rollback_from: null,
  rollback_to: null,
  status: 'ACTIVE',
  created_at: '2026-01-02T00:00:00Z',
  acknowledged_at: null,
  resolved_at: null,
  ...overrides,
})

const trainingFixture = (finished_at: string | null): TrainingJob => ({
  id: 'job-1',
  model_type: 'electric_load',
  dataset_id: 'dataset-1',
  preprocess_script_id: null,
  preprocessing_task_id: null,
  train_script_id: 'script-1',
  split_strategy: 'time_ordered',
  split_ratio: 0.8,
  test_ratio: 0.2,
  status: 'SUCCEEDED',
  progress_stage: 'SUCCEEDED',
  current_stage: null,
  stage: null,
  stage_started_at: null,
  logs: [],
  error_message: null,
  config: {},
  config_summary: {},
  dataset_split: null,
  train_row_count: null,
  test_row_count: null,
  train_time_start: null,
  train_time_end: null,
  test_time_start: null,
  test_time_end: null,
  model_version_id: 'model-1',
  created_at: '2026-01-01T00:00:00Z',
  started_at: null,
  finished_at,
})

const auditFixture = (overrides: Partial<AuditEvent> = {}): AuditEvent => ({
  id: 'event-1',
  occurred_at: '2026-01-03T00:00:00Z',
  event_type: 'MODEL_PUBLISHED',
  object_type: 'MODEL_VERSION',
  object_id: 'model-1',
  model_type: 'electric_load',
  model_version_id: 'model-1',
  training_job_id: null,
  operator_type: null,
  operator_id: null,
  operator_name: null,
  result: 'SUCCESS',
  message: null,
  request_id: null,
  correlation_id: null,
  metadata: {},
  ...overrides,
})

function alertResponse(items: ModelAlert[], total = items.length): CompatibleArrayResponse<ModelAlert> {
  return Object.assign(items, { total }) as CompatibleArrayResponse<ModelAlert>
}

function renderHome() {
  return render(<MemoryRouter initialEntries={['/']}><HomePage /></MemoryRouter>)
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.localStorage.clear()
})

describe('HomePage', () => {
  it('deduplicates production statistics, excludes baselines, queries training completion, and summarizes audit data', async () => {
    const models = [
      modelFixture({ id: 'baseline', version: 'v0', is_baseline: true, is_current: true, health_status: 'HEALTHY' }),
      modelFixture({ id: 'model-1', version: 'v1', status: 'PUBLISHED', is_current: true, health_status: 'HEALTHY', training_job_id: 'job-1' }),
      modelFixture({ id: 'duplicate-current', version: 'v1b', status: 'PUBLISHED', is_current: true, health_status: 'HEALTHY' }),
      modelFixture({ id: 'model-2', model_type: 'heating_cooling_load', version: 'v2', status: 'PUBLISHED', is_current: true, health_status: null }),
      modelFixture({ id: 'model-3', model_type: 'integrated_energy', version: 'v3', status: 'PUBLISHED', is_current: true, health_status: 'ABNORMAL' }),
      modelFixture({ id: 'candidate', version: 'v4', status: 'READY' }),
      modelFixture({ id: 'baseline-ready', version: 'v0-ready', is_baseline: true, status: 'READY' }),
    ]
    const listModels = vi.spyOn(apiClient, 'listModels').mockResolvedValue(models)
    const listAlerts = vi.spyOn(apiClient, 'listAlerts').mockResolvedValue(alertResponse([alertFixture()], 2))
    const listAuditEvents = vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue({
      items: [auditFixture()], page: 1, page_size: 10, total: 1, has_next: false,
    })
    const getTrainingJob = vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(trainingFixture('2026-01-02T03:04:05Z'))

    renderHome()

    expect(await screen.findByText('3 个当前版本 / 1 个健康 / 1 个未知')).toBeTruthy()
    expect(screen.getByText('1 个 READY 版本')).toBeTruthy()
    expect(screen.getByText('2 条 ACTIVE 告警')).toBeTruthy()
    expect(screen.getByText('MODEL_PUBLISHED')).toBeTruthy()
    expect(screen.getByText(/model-1/)).toBeTruthy()
    expect(screen.getAllByText(/2026/).length).toBeGreaterThan(0)
    expect(listModels).toHaveBeenCalledTimes(1)
    expect(listAlerts).toHaveBeenCalledWith({ status: 'ACTIVE', page: 1, page_size: 100 })
    expect(listAuditEvents).toHaveBeenCalledWith({ page: 1, page_size: 10 })
    expect(getTrainingJob).toHaveBeenCalledWith('job-1')
    expect(screen.getByRole('link', { name: '查看全部审计事件' }).getAttribute('href')).toBe('/audit')
  })

  it('does not count an alert with a missing status as ACTIVE and renders it as unknown', async () => {
    const missingStatus = { ...alertFixture({ id: 'alert-unknown' }), status: undefined } as unknown as ModelAlert
    vi.spyOn(apiClient, 'listModels').mockResolvedValue([])
    vi.spyOn(apiClient, 'listAlerts').mockResolvedValue([missingStatus] as CompatibleArrayResponse<ModelAlert>)
    vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, has_next: false,
    })

    renderHome()

    expect(await screen.findByText('0 条 ACTIVE 告警')).toBeTruthy()
    expect(screen.getByText('状态：未知')).toBeTruthy()
  })

  it('shows version labels instead of ids and hides an alert with the close button', async () => {
    const models = [
      modelFixture({ id: 'model-4', version: 'v4', status: 'PUBLISHED', health_status: 'ABNORMAL' }),
      modelFixture({ id: 'model-5', version: 'v5', status: 'PUBLISHED', is_current: true, health_status: 'HEALTHY' }),
    ]
    vi.spyOn(apiClient, 'listModels').mockResolvedValue(models)
    vi.spyOn(apiClient, 'listAlerts').mockResolvedValue(alertResponse([
      alertFixture({
        id: 'ad05087f-72ab-49b2-b5ad-530a382ab807',
        model_version_id: 'model-4',
        rollback_from: 'model-4',
        rollback_to: 'model-5',
        reason: '漂移',
      }),
    ], 1))
    vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, has_next: false,
    })
    const acknowledge = vi.spyOn(apiClient, 'acknowledgeAlert')

    renderHome()

    expect(await screen.findByText('回滚：v4 → v5')).toBeTruthy()
    expect(screen.getByText('电力负荷预测 · 原因：漂移')).toBeTruthy()
    // No raw identifier may leak into the rendered alert row.
    expect(screen.queryByText('ad05087f-72ab-49b2-b5ad-530a382ab807')).toBeNull()
    expect(screen.queryByText('model-4')).toBeNull()
    expect(screen.queryByText('model-5')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: '关闭告警显示：v4' }))

    await waitFor(() => {
      expect(screen.queryByText('电力负荷预测 · 原因：漂移')).toBeNull()
    })
    // Closing removes the alert completely: no leftover row, no leftover
    // counter and no restore control.
    expect(screen.getByText('暂无活动告警')).toBeTruthy()
    expect(screen.getByText('0 条 ACTIVE 告警')).toBeTruthy()
    expect(screen.queryByRole('button', { name: '恢复显示' })).toBeNull()
    // Closing is a local display action; the alert itself must not be modified.
    expect(acknowledge).not.toHaveBeenCalled()
  })

  it('keeps a dismissed alert hidden across a remount', async () => {
    const models = [
      modelFixture({ id: 'model-4', version: 'v4', status: 'PUBLISHED', health_status: 'ABNORMAL' }),
      modelFixture({ id: 'model-5', version: 'v5', status: 'PUBLISHED', is_current: true, health_status: 'HEALTHY' }),
    ]
    vi.spyOn(apiClient, 'listModels').mockResolvedValue(models)
    vi.spyOn(apiClient, 'listAlerts').mockResolvedValue(alertResponse([
      alertFixture({ model_version_id: 'model-4', rollback_from: 'model-4', rollback_to: 'model-5' }),
    ], 1))
    vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, has_next: false,
    })

    const first = renderHome()
    fireEvent.click(await screen.findByRole('button', { name: '关闭告警显示：v4' }))
    await waitFor(() => expect(screen.getByText('暂无活动告警')).toBeTruthy())
    first.unmount()

    renderHome()
    await waitFor(() => expect(screen.queryByText('电力负荷预测 · 原因：健康检查异常')).toBeNull())
    expect(await screen.findByText('暂无活动告警')).toBeTruthy()
  })

  it('shows explicit empty states when the dashboard has no data', async () => {
    vi.spyOn(apiClient, 'listModels').mockResolvedValue([])
    vi.spyOn(apiClient, 'listAlerts').mockResolvedValue(alertResponse([], 0))
    vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, has_next: false,
    })

    renderHome()

    expect(await screen.findByText('暂无模型版本')).toBeTruthy()
    expect(screen.getByText('暂无活动告警')).toBeTruthy()
    expect(screen.getByText('暂无审计事件')).toBeTruthy()
  })

  it('keeps model, alert, and audit errors separate from their loading and empty states', async () => {
    vi.spyOn(apiClient, 'listModels').mockRejectedValue(new ApiError('模型服务不可用', { status: 503 }))
    vi.spyOn(apiClient, 'listAlerts').mockRejectedValue(new ApiError('告警服务不可用', { status: 503 }))
    vi.spyOn(apiClient, 'listAuditEvents').mockRejectedValue(new ApiError('审计服务不可用', { status: 503 }))

    renderHome()

    await waitFor(() => expect(screen.getAllByRole('alert')).toHaveLength(3))
    expect(screen.getByText('模型服务不可用')).toBeTruthy()
    expect(screen.getByText('告警服务不可用')).toBeTruthy()
    expect(screen.getByText('审计服务不可用')).toBeTruthy()
    expect(screen.queryByText('暂无模型版本')).toBeNull()
    expect(screen.queryByText('暂无活动告警')).toBeNull()
    expect(screen.queryByText('暂无审计事件')).toBeNull()
  })
})
