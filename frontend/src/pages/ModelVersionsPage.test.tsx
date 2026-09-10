import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { apiClient, ApiError } from '../api'
import type { AuditEventsResponse, ModelAlert, ModelVersionDetail, ModelVersionSummary, TrainingJob } from '../types/contracts'
import { ModelVersionsPage, MODEL_REGISTRY_UPDATED_EVENT } from './ModelVersionsPage'

const modelFixture = (overrides: Partial<ModelVersionSummary> = {}): ModelVersionSummary => ({
  id: 'model-candidate',
  model_type: 'electric_load',
  version: 'v2',
  status: 'READY',
  health_status: 'HEALTHY',
  is_baseline: false,
  is_current: false,
  is_abnormal: false,
  is_rollback_available: false,
  metrics: { mae: 1.2, rmse: 2.3, mape: 3.4, r2: 0.8 },
  preprocess_used: true,
  model_path: 'models/v2.joblib',
  preprocessor_path: null,
  training_job_id: 'job-v2',
  train_script_id: 'script-train-v2',
  train_script_version: '1.0.0',
  preprocess_script_id: 'script-pre-v2',
  preprocess_script_version: '2.0.0',
  previous_healthy_version_id: 'model-current',
  train_script: null,
  preprocess_script: null,
  feature_columns: ['feature'],
  time_column: 'time',
  target_column: 'target',
  created_at: '2026-01-01T00:00:00Z',
  published_at: null,
  ...overrides,
})

const currentFixture = (overrides: Partial<ModelVersionSummary> = {}): ModelVersionSummary => modelFixture({
  id: 'model-current',
  version: 'v1',
  status: 'PUBLISHED',
  is_current: true,
  training_job_id: 'job-v1',
  train_script_id: 'script-train-v1',
  preprocess_script_id: null,
  preprocess_used: false,
  previous_healthy_version_id: null,
  published_at: '2026-01-02T00:00:00Z',
  ...overrides,
})

const targetFixture = (overrides: Partial<ModelVersionSummary> = {}): ModelVersionSummary => modelFixture({
  id: 'model-target',
  version: 'v0',
  status: 'RETIRED',
  is_current: false,
  training_job_id: 'job-v0',
  train_script_id: 'script-train-v0',
  preprocess_script_id: null,
  preprocess_used: false,
  previous_healthy_version_id: null,
  published_at: '2025-12-01T00:00:00Z',
  ...overrides,
})

const trainingFixture = (id: string, finished_at: string | null): TrainingJob => ({
  id,
  model_type: 'electric_load',
  dataset_id: `dataset-${id}`,
  preprocess_script_id: null,
  preprocessing_task_id: null,
  train_script_id: 'script-train',
  split_strategy: 'time_ordered',
  split_ratio: 0.8,
  test_ratio: 0.2,
  status: 'SUCCEEDED',
  progress_stage: 'SUCCEEDED',
  current_stage: null,
  stage: null,
  stage_started_at: null,
  progress: null,
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
  model_version_id: null,
  created_at: '2026-01-01T00:00:00Z',
  started_at: null,
  finished_at,
})

const alertFixture = (overrides: Partial<ModelAlert> = {}): ModelAlert => ({
  id: 'alert-1',
  model_type: 'electric_load',
  model_version_id: 'model-candidate',
  reason: '指标异常',
  rollback_from: null,
  rollback_to: null,
  status: 'ACTIVE',
  created_at: '2026-01-03T00:00:00Z',
  acknowledged_at: null,
  resolved_at: null,
  ...overrides,
})

const auditResponse: AuditEventsResponse = { items: [], page: 1, page_size: 10, total: 0, has_next: false }

function detail(model: ModelVersionSummary): ModelVersionDetail {
  return { ...model, input_schema: {}, evaluation: null }
}

function renderPage(path = '/') {
  return render(<MemoryRouter initialEntries={[path]}><ModelVersionsPage /></MemoryRouter>)
}

function mockReads(models: ModelVersionSummary[]) {
  vi.spyOn(apiClient, 'listModels').mockResolvedValue(models)
  vi.spyOn(apiClient, 'listAlerts').mockResolvedValue([])
  vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue(auditResponse)
  vi.spyOn(apiClient, 'getTrainingJob').mockImplementation(async (id) => trainingFixture(id, `2026-01-0${id.endsWith('v2') ? '2' : '1'}T03:04:05Z`))
  vi.spyOn(apiClient, 'getScript').mockImplementation(async (id) => ({
    id,
    name: id.includes('pre') ? '预处理脚本.py' : '训练脚本.py',
    script_type: id.includes('pre') ? 'preprocessor' : 'trainer',
    version: '1.0.0',
    source_code: '',
    supported_model_types: ['electric_load'],
    status: 'ENABLED',
    created_at: '2026-01-01T00:00:00Z',
    uploaded_at: '2026-01-01T00:00:00Z',
  }))
  vi.spyOn(apiClient, 'getModel').mockImplementation(async (id) => {
    const model = models.find((item) => item.id === id) ?? modelFixture({ id })
    return detail(model)
  })
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ModelVersionsPage', () => {
  it('keeps the model selector and converges the list to source-backed fields', async () => {
    const candidate = modelFixture()
    const current = currentFixture()
    const target = targetFixture()
    mockReads([candidate, current, target, modelFixture({ id: 'unknown', version: 'v-unknown', health_status: null, training_job_id: null })])

    renderPage()

    expect(await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })).toBeTruthy()
    expect(screen.getByLabelText('模型类型')).toBeTruthy()
    for (const heading of ['版本号', '版本标题', '训练完成时间', '训练脚本名称', '生命周期', '健康状态']) {
      expect(screen.getAllByRole('columnheader', { name: heading }).length).toBeGreaterThan(0)
    }
    expect(screen.queryByRole('columnheader', { name: '评估指标' })).toBeNull()
    expect(screen.getByRole('button', { name: 'v2' })).toBeTruthy()
    expect(screen.getAllByText('训练脚本.py').length).toBeGreaterThan(0)
    expect(screen.getAllByText('未知').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: 'v2' }))
    expect(await screen.findByRole('heading', { name: 'v2' })).toBeTruthy()
    expect(screen.getByText('预处理')).toBeTruthy()
    expect(screen.getByText('训练任务')).toBeTruthy()
    expect(screen.getByText('1.2')).toBeTruthy()
    expect(screen.getByText('3.4')).toBeTruthy()
    expect(apiClient.getModel).toHaveBeenCalledWith('model-candidate')
    expect(apiClient.getModel).toHaveBeenCalledWith('model-current')
    expect(apiClient.getTrainingJob).toHaveBeenCalledWith('job-v2')
  })

  it('switches the detail panel when a different version is clicked', async () => {
    const candidate = modelFixture({ metrics: { mae: 1.2, rmse: 2.3, mape: 3.4, r2: 0.8 } })
    const current = currentFixture({ metrics: { mae: 9.8, rmse: 8.7, mape: 7.6, r2: 0.5 } })
    mockReads([candidate, current])

    renderPage()
    expect(await screen.findByRole('heading', { name: 'v2' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'v1' }))

    expect(await screen.findByRole('heading', { name: 'v1' })).toBeTruthy()
    expect(screen.getByText('9.8')).toBeTruthy()
    expect(screen.getByText('模型版本管理')).toBeTruthy()
    expect(apiClient.getModel).toHaveBeenCalledWith('model-current')
  })

  it('opens the clicked version when the version list row is selected', async () => {
    const candidate = modelFixture()
    const current = currentFixture()
    mockReads([candidate, current])

    renderPage()
    await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })

    fireEvent.click(screen.getByRole('row', { name: /v1/ }))

    expect(await screen.findByRole('heading', { name: 'v1' })).toBeTruthy()
    expect(apiClient.getModel).toHaveBeenCalledWith('model-current')
  })

  it('consumes the model type query and hash version deep link', async () => {
    const model = modelFixture({ id: 'heat-model', model_type: 'heating_cooling_load', version: 'h1' })
    mockReads([model])

    renderPage('/models?model_type=heating_cooling_load#heat-model')

    expect(await screen.findByRole('heading', { name: '冷热负荷预测 · 版本列表' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'h1' })).toBeTruthy()
  })

  it('enforces lifecycle and independent health gates without defaulting missing health to healthy', async () => {
    const candidate = modelFixture()
    const current = currentFixture()
    const target = targetFixture()
    const unknown = modelFixture({ id: 'model-unknown', version: 'v-unknown', health_status: null })
    mockReads([candidate, current, target, unknown])

    renderPage()
    await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })

    expect(screen.getByRole('button', { name: '发布版本' })).not.toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '下线' })).toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '标记异常' })).not.toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '回滚到所选版本' })).toHaveProperty('disabled', true)

    fireEvent.click(screen.getByRole('button', { name: 'v1' }))
    expect(screen.getByRole('button', { name: '发布版本' })).toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '下线' })).not.toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '回滚到所选版本' })).toHaveProperty('disabled', true)

    fireEvent.click(screen.getByRole('button', { name: 'v-unknown' }))
    expect(screen.getAllByText('UNKNOWN（未知）').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: '发布版本' })).toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '标记异常' })).not.toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '回滚到所选版本' })).toHaveProperty('disabled', true)
  })

  it('requires reasons, sends explicit publish/rollback contracts, and refreshes all read models', async () => {
    const candidate = modelFixture()
    const current = currentFixture()
    const target = targetFixture()
    mockReads([candidate, current, target])
    const publish = vi.spyOn(apiClient, 'publishModel').mockResolvedValue({ operation: 'publish', model: detail(current) })
    const listModels = vi.mocked(apiClient.listModels)
    const listAlerts = vi.mocked(apiClient.listAlerts)
    const listAuditEvents = vi.mocked(apiClient.listAuditEvents)
    const updated = currentFixture({ version: 'v2', id: 'model-candidate', status: 'PUBLISHED', is_current: true })
    listModels.mockResolvedValueOnce([candidate, current, target]).mockResolvedValueOnce([updated, target])
    listAlerts.mockResolvedValueOnce([]).mockResolvedValueOnce([])
    listAuditEvents.mockResolvedValueOnce(auditResponse).mockResolvedValueOnce(auditResponse)
    const dispatch = vi.spyOn(window, 'dispatchEvent')

    renderPage()
    await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })
    fireEvent.click(screen.getByRole('button', { name: '发布版本' }))
    expect(screen.getByRole('dialog')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '确认发布' }))
    expect(publish).not.toHaveBeenCalled()
    expect(screen.getByText('发布原因不能为空')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('发布原因（必填）'), { target: { value: '评估通过' } })
    fireEvent.click(screen.getByRole('button', { name: '确认发布' }))
    await waitFor(() => expect(publish).toHaveBeenCalledWith('model-candidate', {
      confirmed: true,
      reason: '评估通过',
      idempotency_key: expect.any(String),
    }))
    await waitFor(() => {
      expect(listModels).toHaveBeenCalledTimes(2)
      expect(listAlerts).toHaveBeenCalledTimes(2)
      expect(listAuditEvents).toHaveBeenCalledTimes(2)
    })
    expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: MODEL_REGISTRY_UPDATED_EVENT }))
  })

  it('uses the current version as the rollback path and requires a target reason', async () => {
    const candidate = modelFixture({ status: 'RETIRED', published_at: '2025-12-02T00:00:00Z' })
    const current = currentFixture()
    const target = targetFixture()
    mockReads([current, target, candidate])
    const rollback = vi.spyOn(apiClient, 'rollbackModel').mockResolvedValue({ operation: 'rollback', model: detail(target), rollback: null })

    renderPage()
    await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })
    fireEvent.click(screen.getByRole('button', { name: 'v0' }))
    fireEvent.click(screen.getByRole('button', { name: '回滚到所选版本' }))
    expect(screen.getByText('当前版本：').parentElement?.textContent).toContain('v1')
    expect(screen.getByText('目标版本：').parentElement?.textContent).toContain('v0')
    fireEvent.click(screen.getByRole('button', { name: '确认回滚' }))
    expect(rollback).not.toHaveBeenCalled()
    expect(screen.getByText('回滚原因不能为空')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('回滚原因（必填）'), { target: { value: '生产指标异常' } })
    fireEvent.click(screen.getByRole('button', { name: '确认回滚' }))
    await waitFor(() => expect(rollback).toHaveBeenCalledWith('model-current', {
      target_version_id: 'model-target',
      reason: '生产指标异常',
      idempotency_key: expect.any(String),
    }))
  })

  it('confirms offline and version-scoped abnormal operations, showing failures', async () => {
    const current = currentFixture()
    const unknown = modelFixture({ id: 'model-unknown', version: 'v-unknown', health_status: null })
    mockReads([current, unknown])
    const offline = vi.spyOn(apiClient, 'offlineModel').mockRejectedValue(new ApiError('下线失败', { status: 409 }))
    const abnormal = vi.spyOn(apiClient, 'markModelAbnormal').mockResolvedValue({ operation: 'abnormal', model: detail(unknown), alert: null })

    renderPage()
    await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })
    fireEvent.click(screen.getByRole('button', { name: '下线' }))
    fireEvent.click(screen.getByRole('button', { name: '确认下线' }))
    await waitFor(() => expect(offline).toHaveBeenCalledWith('model-current'))
    expect((await screen.findByRole('alert')).textContent).toContain('下线失败')

    fireEvent.click(screen.getByRole('button', { name: 'v-unknown' }))
    fireEvent.click(screen.getByRole('button', { name: '标记异常' }))
    fireEvent.click(screen.getByRole('button', { name: '确认标记异常' }))
    expect(abnormal).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('异常原因（必填）'), { target: { value: '健康检查失败' } })
    fireEvent.click(screen.getByRole('button', { name: '确认标记异常' }))
    await waitFor(() => expect(abnormal).toHaveBeenCalledWith('model-unknown', '健康检查失败'))
  })
})

describe('ModelVersionsPage training job link', () => {
  it('links to the training workflow instead of showing a raw job id', async () => {
    vi.spyOn(apiClient, 'listModels').mockResolvedValue([
      modelFixture({ id: 'model-1', version: 'v1', training_job_id: 'f7353a96-287d-42f0-bad3-eb9d4764c069' }),
    ])
    vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(
      { finished_at: null } as unknown as TrainingJob,
    )
    renderPage()

    const link = await screen.findByRole('link', { name: '查看训练任务' })
    expect(link.getAttribute('href')).toBe(
      '/workflow/train?training_job_id=f7353a96-287d-42f0-bad3-eb9d4764c069',
    )
    // The raw identifier must not be rendered as visible text.
    expect(screen.queryByText('f7353a96-287d-42f0-bad3-eb9d4764c069')).toBeNull()
  })
})
