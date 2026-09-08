import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { apiClient, type CompatibleArrayResponse } from '../api'
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

  it('acknowledges an active alert and refreshes models, alerts, and audit events', async () => {
    const models = [modelFixture({ id: 'model-1', is_current: true, status: 'PUBLISHED', health_status: 'HEALTHY' })]
    const listModels = vi.spyOn(apiClient, 'listModels').mockResolvedValue(models)
    const listAlerts = vi.spyOn(apiClient, 'listAlerts').mockResolvedValue(alertResponse([alertFixture()], 1))
    const listAuditEvents = vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, has_next: false,
    })
    const acknowledge = vi.spyOn(apiClient, 'acknowledgeAlert').mockResolvedValue({
      ...alertFixture({ status: 'ACKNOWLEDGED' }),
      statistics: { total: 1, active: 0, acknowledged: 1, resolved: 0 },
    })

    renderHome()
    fireEvent.click(await screen.findByRole('button', { name: '确认告警 alert-1' }))

    await waitFor(() => expect(acknowledge).toHaveBeenCalledWith('alert-1'))
    await waitFor(() => {
      expect(listModels).toHaveBeenCalledTimes(2)
      expect(listAlerts).toHaveBeenCalledTimes(2)
      expect(listAuditEvents).toHaveBeenCalledTimes(2)
    })
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
})
