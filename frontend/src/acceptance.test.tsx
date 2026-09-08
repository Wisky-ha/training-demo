import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from './App'
import { ApiClient, ApiError, apiClient } from './api'
import { McpPage } from './pages/McpPage'
import { ModelVersionsPage } from './pages/ModelVersionsPage'
import { useAppStore } from './store/useAppStore'
import type { DatasetSplitResult, DatasetUploadResult, ModelEvaluation, ModelVersionSummary, PreprocessTask, ScriptContract, TrainingJob } from './types/contracts'

const baseModel = (overrides: Partial<ModelVersionSummary> = {}): ModelVersionSummary => ({
  id: 'model-1',
  model_type: 'electric_load',
  version: 'v1',
  status: 'READY',
  health_status: 'HEALTHY',
  is_baseline: false,
  is_current: false,
  is_abnormal: false,
  is_rollback_available: false,
  metrics: { mae: 1.2, rmse: 2.3 },
  preprocess_used: false,
  model_path: 'models/v1.joblib',
  preprocessor_path: null,
  training_job_id: 'job-1',
  train_script_id: 'trainer-1',
  train_script_version: '1.0.0',
  preprocess_script_id: null,
  preprocess_script_version: null,
  previous_healthy_version_id: null,
  train_script: null,
  preprocess_script: null,
  feature_columns: ['feature'],
  time_column: 'time',
  target_column: 'target',
  created_at: '2026-01-01T00:00:00Z',
  published_at: null,
  ...overrides,
})

const datasetFixture = (overrides: Partial<DatasetUploadResult> = {}): DatasetUploadResult => ({
  id: 'dataset-1', dataset_id: 'dataset-1', file_name: 'load.csv', file_path: 'datasets/load.csv',
  file_size_bytes: 10, checksum_sha256: 'checksum', file_storage: null, row_count: 5, column_count: 3,
  columns: [
    { name: 'time', role: 'time', data_type: 'datetime', nullable: false, missing_count: 0, missing_ratio: 0 },
    { name: 'feature', role: 'feature', data_type: 'number', nullable: false, missing_count: 0, missing_ratio: 0 },
    { name: 'target', role: 'target', data_type: 'number', nullable: false, missing_count: 0, missing_ratio: 0 },
  ],
  column_names: ['time', 'feature', 'target'], field_roles: { time: 'time', feature: 'feature', target: 'target' },
  time_column: 'time', feature_columns: ['feature'], target_column: 'target',
  column_types: { time: 'datetime', feature: 'number', target: 'number' }, missing_value_counts: { time: 0, feature: 0, target: 0 },
  missing_values: {}, preview_rows: [], preview: [], numeric_columns: ['feature', 'target'],
  time_parse: { success: true, format: null, invalid_count: 0, min: '2026-01-01T00:00:00Z', max: '2026-01-05T00:00:00Z', message: null },
  time_range: { start: '2026-01-01T00:00:00Z', end: '2026-01-05T00:00:00Z' },
  validation: { valid: true, errors: [], warnings: [], checks: {} }, summary: {}, data_summary: {}, status: 'parsed',
  created_at: '2026-01-01T00:00:00Z', ...overrides,
})

const preprocessFixture = (overrides: Partial<PreprocessTask> = {}): PreprocessTask => ({
  id: 'task-1', model_type: 'electric_load', dataset_id: 'dataset-1', preprocess_script_id: null,
  preprocess_used: false, preprocess_status: 'unused', preprocess_message: '未使用预处理，后续使用原始特征',
  status: 'SKIPPED', stage: 'completed', progress_stage: 'completed', logs: ['未使用预处理，后续使用原始特征'],
  error_message: null, input_row_count: 5, output_row_count: 5, input_columns: ['time', 'feature', 'target'],
  output_columns: ['time', 'feature', 'target'], input_summary: {}, output_summary: {}, preprocessor_path: null,
  preprocessor_state: null, started_at: null, finished_at: '2026-01-01T00:00:01Z', created_at: '2026-01-01T00:00:00Z',
  next_step: 'dataset_split', data_source: 'raw', ...overrides,
})

const splitFixture = (overrides: Partial<DatasetSplitResult> = {}): DatasetSplitResult => ({
  id: 'split-1', dataset_id: 'dataset-1', preprocessing_task_id: 'task-1', data_source: 'raw',
  split_strategy: 'time_ordered', split_ratio: 0.8, test_ratio: 0.2, total_row_count: 5,
  train_row_count: 4, test_row_count: 1, train_time_range: { start: '2026-01-01T00:00:00Z', end: '2026-01-04T00:00:00Z' },
  test_time_range: { start: '2026-01-05T00:00:00Z', end: '2026-01-05T00:00:00Z' },
  train_time_start: '2026-01-01T00:00:00Z', train_time_end: '2026-01-04T00:00:00Z',
  test_time_start: '2026-01-05T00:00:00Z', test_time_end: '2026-01-05T00:00:00Z', created_at: '2026-01-01T00:00:00Z', ...overrides,
})

const trainingFixture = (overrides: Partial<TrainingJob> = {}): TrainingJob => ({
  id: 'job-1', model_type: 'electric_load', dataset_id: 'dataset-1', preprocess_script_id: null,
  preprocessing_task_id: 'task-1', train_script_id: 'trainer-1', split_strategy: 'time_ordered', split_ratio: 0.8,
  test_ratio: 0.2, status: 'SUCCEEDED', progress_stage: 'SUCCEEDED', current_stage: '进入评估', logs: [],
  error_message: null, model_version_id: 'model-1', train_row_count: 4, test_row_count: 1,
  train_time_start: '2026-01-01T00:00:00Z', train_time_end: '2026-01-04T00:00:00Z',
  test_time_start: '2026-01-05T00:00:00Z', test_time_end: '2026-01-05T00:00:00Z',
  created_at: '2026-01-01T00:00:00Z', started_at: '2026-01-01T00:00:01Z', finished_at: '2026-01-01T00:01:00Z', ...overrides,
})

const trainerFixture: ScriptContract = {
  id: 'trainer-1', name: '训练脚本', script_type: 'trainer', version: '1.0.0', source_code: 'return 0',
  supported_model_types: ['electric_load'], status: 'ENABLED', created_at: '2026-01-01T00:00:00Z', uploaded_at: '2026-01-01T00:00:00Z',
}

const evaluationFixture = (overrides: Partial<ModelEvaluation> = {}): ModelEvaluation => ({
  job_id: 'job-1', model_version_id: 'model-1',
  metrics: { mae: 1, rmse: 2, mape: 3, r2: 0.8, sample_count: 1, mape_valid_count: 1, mape_excluded_count: 0, mape_note: '' },
  chart_data: [], error_data: [], model_comparison: {
    candidate: { model_version_id: 'model-1', version: 'v1', metrics: { mae: 1, rmse: 2, mape: 3, r2: 0.8, sample_count: 1, mape_valid_count: 1, mape_excluded_count: 0, mape_note: '' } },
    baseline: { model_version_id: null, version: null, metrics: { mae: null, rmse: null, mape: null, r2: null, sample_count: 0, mape_valid_count: 0, mape_excluded_count: 0, mape_note: '' } },
    changes: { mae: null, rmse: null, mape: null, r2: null },
  }, chart_sampled: false, chart_total_count: 0, chart_sample_count: 0,
  test_time_series: [], timestamps: [], actual_values: [], candidate_predictions: [], baseline_predictions: [],
  candidate_errors: [], baseline_errors: [], error_series: [], ...overrides,
})

function mockWorkflowReads({ job = trainingFixture(), model = baseModel(), evaluation = evaluationFixture() }: {
  job?: TrainingJob
  model?: ModelVersionSummary
  evaluation?: ModelEvaluation
} = {}) {
  vi.spyOn(apiClient, 'getDataset').mockResolvedValue(datasetFixture())
  vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(preprocessFixture())
  vi.spyOn(apiClient, 'getDatasetSplit').mockResolvedValue(splitFixture())
  vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(job)
  vi.spyOn(apiClient, 'getModel').mockResolvedValue({ ...model, input_schema: {}, evaluation: null })
  const evaluationRead = vi.spyOn(apiClient, 'getTrainingJobEvaluation').mockResolvedValue(evaluation)
  return { evaluationRead }
}

function renderApp(path: string) {
  return render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>)
}

function seedWorkflow(context: Record<string, unknown>) {
  useAppStore.getState().resetWorkflow()
  useAppStore.getState().setWorkflowContext(context as never)
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  useAppStore.getState().resetWorkflow()
  window.localStorage.clear()
})

describe('API client browser integration', () => {
  it('keeps the browser fetch receiver when using the default adapter', async () => {
    const originalFetch = globalThis.fetch
    const fetchMock = vi.fn(function (this: unknown) {
      expect(this).toBe(globalThis)
      return Promise.resolve({
        status: 200,
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ status: 'ok' }),
      } as Response)
    })
    vi.stubGlobal('fetch', fetchMock)
    try {
      await expect(new ApiClient().getHealth()).resolves.toEqual({ status: 'ok' })
    } finally {
      vi.stubGlobal('fetch', originalFetch)
    }
  })
})

describe('workflow acceptance flows', () => {
  it('redirects to model type when navigating without a selected model type', async () => {
    renderApp('/workflow/upload')

    expect(await screen.findByRole('heading', { name: '选择要训练的模型类型' })).toBeTruthy()
    expect((screen.getByRole('button', { name: /× 上传数据 Dataset/ }) as HTMLButtonElement).className).toContain('locked')
  })

  it('shows a client-side error for an upload with an invalid extension', async () => {
    useAppStore.getState().setWorkflowContext({ modelType: 'electric_load' })
    const upload = vi.spyOn(apiClient, 'uploadDataset')
    renderApp('/workflow/upload')

    const input = screen.getByLabelText(/拖拽 CSV 文件到这里/)
    fireEvent.change(input, { target: { files: [new File(['not csv'], 'readme.txt', { type: 'text/plain' })] } })

    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('仅支持 .csv 文件'))
    expect(upload).not.toHaveBeenCalled()
  })

  it('renders upload metadata and the empty field-summary state from the response', async () => {
    useAppStore.getState().setWorkflowContext({ modelType: 'electric_load' })
    const dataset = datasetFixture({ columns: [], column_count: 0, file_size_bytes: 0, row_count: 0, time_range: { start: '', end: '' }, validation: { valid: false, errors: [], warnings: [], checks: {} } })
    vi.spyOn(apiClient, 'uploadDataset').mockResolvedValue(dataset)
    vi.spyOn(apiClient, 'getDataset').mockResolvedValue(dataset)
    renderApp('/workflow/upload')

    fireEvent.change(screen.getByLabelText(/拖拽 CSV 文件到这里/), { target: { files: [new File([''], 'empty.csv', { type: 'text/csv' })] } })
    expect(await screen.findAllByText('load.csv')).not.toHaveLength(0)
    expect(screen.getAllByText('dataset-1').length).toBeGreaterThan(0)
    expect(screen.getAllByText('失败').length).toBeGreaterThan(0)
    expect(screen.getByText('暂无字段摘要')).toBeTruthy()
  })

  it('shows the explicit unused-preprocessing state when skip is selected', async () => {
    seedWorkflow({
      modelType: 'electric_load',
      datasetId: 'dataset-1',
      dataset: datasetFixture(),
      preprocessTaskId: 'task-1',
      preprocessTask: preprocessFixture(),
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    mockWorkflowReads({ job: trainingFixture({ model_version_id: null }) })
    renderApp('/workflow/preprocess')

    expect(await screen.findByText('未使用预处理，后续使用原始特征')).toBeTruthy()
    expect(screen.getByText(/预处理.*已跳过/)).toBeTruthy()
  })

  it('uploads and selects an ENABLED preprocessor script', async () => {
    seedWorkflow({ modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture() })
    const script: ScriptContract = { ...trainerFixture, id: 'preprocess-1', name: '清洗脚本', script_type: 'preprocessor' }
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    const upload = vi.spyOn(apiClient, 'uploadScript').mockResolvedValue(script)
    mockWorkflowReads({ job: trainingFixture({ model_version_id: null }) })
    renderApp('/workflow/preprocess')

    fireEvent.change(screen.getByLabelText('上传预处理脚本'), { target: { files: [new File(['print(1)'], 'clean.py')] } })
    expect(await screen.findByText('清洗脚本')).toBeTruthy()
    expect(upload).toHaveBeenCalledWith(expect.objectContaining({ script_type: 'preprocessor', supported_model_types: ['electric_load'] }))
  })

  it('shows failed-training retry guidance and submits a retry', async () => {
    const failedJob = trainingFixture({ status: 'FAILED', model_version_id: null, progress_stage: 'FAILED', current_stage: '失败', logs: ['FAILED：训练脚本失败'], error_message: '训练脚本失败' })
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(),
      preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(), splitId: 'split-1', split: splitFixture(),
      trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: failedJob, modelVersionId: null,
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    mockWorkflowReads({ job: failedJob })
    const retry = vi.spyOn(apiClient, 'retryTrainingJob').mockResolvedValue({ ...failedJob, status: 'PENDING', error_message: null })
    renderApp('/workflow/train')

    expect(await screen.findByText('本次训练失败，不会影响生产模型。修复脚本或配置后可以重试。')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '失败重试' }))
    await waitFor(() => expect(retry).toHaveBeenCalledWith('job-1'))
  })

  it('merges incremental training logs and can cancel a running job', async () => {
    const runningJob = trainingFixture({ status: 'RUNNING', model_version_id: null, current_stage: '训练中', logs: [] })
    const logRead = vi.spyOn(apiClient, 'getTrainingJobLogs').mockResolvedValue({
      job_id: 'job-1', items: [{ timestamp: '2026-01-01T00:00:01Z', level: 'info', message: '第一条日志', stage: 'training' }], next_cursor: 'cursor-2',
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [trainerFixture], pagination: { page: 1, page_size: 20, total: 1, total_pages: 1 } })
    vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(runningJob)
    const cancel = vi.spyOn(apiClient, 'cancelTrainingJob').mockResolvedValue({ ...runningJob, status: 'CANCELLED' })
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(), preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(),
      splitId: 'split-1', split: splitFixture(), trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: runningJob,
    })
    mockWorkflowReads({ job: runningJob })
    renderApp('/workflow/train')

    expect(await screen.findByText('第一条日志')).toBeTruthy()
    expect(logRead).toHaveBeenCalledWith('job-1', expect.objectContaining({ limit: 100 }))
    fireEvent.click(screen.getByRole('button', { name: '取消训练' }))
    await waitFor(() => expect(cancel).toHaveBeenCalledWith('job-1'))
  })

  it('enables the evaluation transition after training succeeds with a model version ID', async () => {
    const succeededJob = trainingFixture()
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(),
      preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(), splitId: 'split-1', split: splitFixture(),
      trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: succeededJob, modelVersionId: 'model-1',
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    mockWorkflowReads({ job: succeededJob })
    renderApp('/workflow/train')

    const next = await screen.findByRole('button', { name: /查看评估结果/ }) as HTMLButtonElement
    expect(next.disabled).toBe(false)
    fireEvent.click(next)
    expect(await screen.findByRole('heading', { name: '评估结果与模型对比' })).toBeTruthy()
  })

  it('shows chart empty states when evaluation data is empty', async () => {
    const succeededJob = trainingFixture()
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(), preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(),
      splitId: 'split-1', split: splitFixture(), trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: succeededJob,
      modelVersionId: 'model-1', modelVersion: baseModel(),
    })
    mockWorkflowReads({ job: succeededJob, evaluation: evaluationFixture({ chart_data: [], error_data: [] }) })
    renderApp('/workflow/evaluate')
    expect(await screen.findAllByText('暂无可绘制的有效数据')).toHaveLength(2)
  })

  it('blocks evaluation when a succeeded training job has no model_version_id', async () => {
    const succeededJob = trainingFixture({ model_version_id: null })
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(),
      preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(), splitId: 'split-1', split: splitFixture(),
      trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: succeededJob, modelVersionId: null,
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    const { evaluationRead } = mockWorkflowReads({ job: succeededJob })
    renderApp('/workflow/evaluate')

    expect(await screen.findByRole('heading', { name: '选择训练脚本并启动' })).toBeTruthy()
    expect(screen.getAllByText(/缺少 model_version_id/).length).toBeGreaterThan(0)
    expect((screen.getByRole('button', { name: /查看评估结果/ }) as HTMLButtonElement).disabled).toBe(true)
    expect(evaluationRead).not.toHaveBeenCalled()
  })

  it('requires a reason in a custom confirmation dialog before publishing', async () => {
    const succeededJob = trainingFixture()
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(),
      preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(), splitId: 'split-1', split: splitFixture(),
      trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: succeededJob,
      modelVersionId: 'model-1', modelVersion: baseModel(), evaluation: evaluationFixture(),
    })
    mockWorkflowReads({ job: succeededJob })
    const publish = vi.spyOn(apiClient, 'publishModel')
    renderApp('/workflow/publish')

    fireEvent.click(await screen.findByRole('button', { name: '发布' }))
    expect(await screen.findByRole('dialog')).toBeTruthy()
    expect(publish).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: '确认发布' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('saves a candidate as READY and publishes with a required reason', async () => {
    const succeededJob = trainingFixture()
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(), preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(),
      splitId: 'split-1', split: splitFixture(), trainScriptId: 'trainer-1', trainingJobId: 'job-1', trainingJob: succeededJob,
      modelVersionId: 'model-1', modelVersion: baseModel({ status: 'DRAFT' }), evaluation: evaluationFixture(),
    })
    mockWorkflowReads({ job: succeededJob, model: baseModel({ status: 'DRAFT' }) })
    vi.spyOn(apiClient, 'listModels').mockResolvedValue([])
    const save = vi.spyOn(apiClient, 'saveModel').mockResolvedValue(baseModel({ status: 'DRAFT' }))
    const publish = vi.spyOn(apiClient, 'publishModel').mockResolvedValue({ operation: 'publish', model: baseModel({ status: 'PUBLISHED', is_current: true }) })
    renderApp('/workflow/publish')

    fireEvent.click(await screen.findByRole('button', { name: '保存候选版本' }))
    await waitFor(() => expect(save).toHaveBeenCalledWith('model-1', expect.objectContaining({ status: 'READY' })))
    fireEvent.click(await screen.findByRole('button', { name: '发布' }))
    fireEvent.change(screen.getByLabelText('发布原因'), { target: { value: '验证通过，切换生产版本' } })
    fireEvent.click(screen.getByRole('button', { name: '确认发布' }))
    await waitFor(() => expect(publish).toHaveBeenCalledWith('model-1', expect.objectContaining({
      confirmed: true, reason: '验证通过，切换生产版本', idempotency_key: expect.any(String),
    })))
  })

  it('persists workflow resource IDs and currentStep instead of treating objects as durable state', () => {
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(),
      preprocessScriptId: 'preprocess-1', preprocessTaskId: 'task-1', preprocessTask: preprocessFixture(),
      splitId: 'split-1', split: splitFixture(), trainScriptId: 'trainer-1', trainingJobId: 'job-1',
      trainingJob: trainingFixture(), modelVersionId: 'model-1', modelVersion: baseModel(),
      currentStep: 'evaluate',
    })
    const persisted = JSON.parse(window.localStorage.getItem('model-training-platform-ui') ?? '{}') as { state?: { workflow?: Record<string, unknown> } }
    expect(persisted.state?.workflow).toEqual({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessScriptId: 'preprocess-1',
      preprocessTaskId: 'task-1', splitId: 'split-1', trainScriptId: 'trainer-1', trainingJobId: 'job-1',
      modelVersionId: 'model-1', currentStep: 'evaluate',
    })
  })

  it('does not treat legacy persisted resource objects as hydrated state', async () => {
    window.localStorage.setItem('model-training-platform-ui', JSON.stringify({
      state: {
        theme: 'light',
        sidebarCollapsed: false,
        workflow: {
          modelType: 'electric_load',
          datasetId: 'dataset-1',
          dataset: datasetFixture({ file_name: '过期缓存.csv' }),
          currentStep: 'upload',
        },
      },
      version: 0,
    }))

    await act(async () => { await useAppStore.persist.rehydrate() })
    expect(useAppStore.getState().workflow.datasetId).toBe('dataset-1')
    expect(useAppStore.getState().workflow.dataset).toBeNull()
  })

  it('clears every downstream ID and cache when the model type changes', () => {
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessScriptId: 'preprocess-1',
      preprocessTaskId: 'task-1', splitId: 'split-1', trainScriptId: 'trainer-1', trainingJobId: 'job-1',
      modelVersionId: 'model-1', dataset: datasetFixture(), preprocessTask: preprocessFixture(), split: splitFixture(),
      trainingJob: trainingFixture(), modelVersion: baseModel(), evaluation: evaluationFixture(), currentStep: 'publish',
    })
    useAppStore.getState().setModelType('heating_cooling_load')
    expect(useAppStore.getState().workflow).toMatchObject({
      modelType: 'heating_cooling_load', datasetId: null, preprocessScriptId: null, preprocessTaskId: null,
      splitId: null, trainScriptId: null, trainingJobId: null, modelVersionId: null, currentStep: 'upload',
      dataset: null, preprocessTask: null, split: null, trainingJob: null, evaluation: null, modelVersion: null,
    })
  })

  it('applies a valid model_type query as a model switch and drops the old chain', async () => {
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessTaskId: 'task-1', splitId: 'split-1',
      trainingJobId: 'job-1', modelVersionId: 'model-1', dataset: datasetFixture(), preprocessTask: preprocessFixture(),
      split: splitFixture(), trainingJob: trainingFixture(), modelVersion: baseModel(), evaluation: evaluationFixture(),
    })
    renderApp('/workflow/upload?model_type=heating_cooling_load')
    expect(await screen.findByRole('heading', { name: '上传并检查 CSV 数据' })).toBeTruthy()
    expect(useAppStore.getState().workflow).toMatchObject({
      modelType: 'heating_cooling_load', datasetId: null, preprocessTaskId: null, splitId: null,
      trainingJobId: null, modelVersionId: null, evaluation: null,
    })
  })

  it('refreshes each cached resource by ID and restores the latest server objects', async () => {
    const oldJob = trainingFixture({ status: 'RUNNING', model_version_id: null, current_stage: '旧状态', finished_at: null })
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessTaskId: 'task-1', splitId: 'split-1',
      trainScriptId: 'trainer-1', trainingJobId: 'job-1', modelVersionId: 'model-1', currentStep: 'evaluate',
      dataset: datasetFixture({ file_name: '旧缓存.csv' }), preprocessTask: preprocessFixture({ logs: ['旧缓存'] }),
      split: splitFixture({ train_row_count: 3 }), trainingJob: oldJob, modelVersion: baseModel({ version: 'old' }),
      evaluation: null,
    })
    useAppStore.getState().setWorkflowContext({
      dataset: null, preprocessTask: null, split: null, trainingJob: null, modelVersion: null, evaluation: null,
      datasetId: 'dataset-1', preprocessTaskId: 'task-1', splitId: 'split-1', trainingJobId: 'job-1', modelVersionId: 'model-1',
    })
    const latestDataset = datasetFixture({ file_name: '最新.csv' })
    const latestTask = preprocessFixture({ logs: ['最新任务'] })
    const latestSplit = splitFixture({ train_row_count: 4 })
    const latestJob = trainingFixture({ current_stage: '最新状态' })
    const latestModel = baseModel({ version: 'latest' })
    const latestEvaluation = evaluationFixture()
    const reads = {
      dataset: vi.spyOn(apiClient, 'getDataset').mockResolvedValue(latestDataset),
      task: vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(latestTask),
      split: vi.spyOn(apiClient, 'getDatasetSplit').mockResolvedValue(latestSplit),
      job: vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(latestJob),
      model: vi.spyOn(apiClient, 'getModel').mockResolvedValue({ ...latestModel, input_schema: {}, evaluation: null }),
      evaluation: vi.spyOn(apiClient, 'getTrainingJobEvaluation').mockResolvedValue(latestEvaluation),
    }
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    renderApp('/workflow/evaluate')

    await waitFor(() => expect(reads.evaluation).toHaveBeenCalledWith('job-1'))
    expect(reads.dataset).toHaveBeenCalledWith('dataset-1')
    expect(reads.task).toHaveBeenCalledWith('task-1')
    expect(reads.split).toHaveBeenCalledWith('dataset-1')
    expect(reads.job).toHaveBeenCalledWith('job-1')
    expect(reads.model).toHaveBeenCalledWith('model-1')
    expect(useAppStore.getState().workflow).toMatchObject({
      dataset: latestDataset, preprocessTask: latestTask, split: latestSplit, trainingJob: latestJob,
      modelVersion: latestModel, evaluation: latestEvaluation,
    })
  })

  it('resumes polling for an unfinished training job after refresh', async () => {
    const runningJob = trainingFixture({ status: 'RUNNING', model_version_id: null, current_stage: '训练中', finished_at: null })
    const succeededJob = trainingFixture({ status: 'SUCCEEDED', model_version_id: 'model-1' })
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(), preprocessTaskId: 'task-1',
      preprocessTask: preprocessFixture(), splitId: 'split-1', split: splitFixture(), trainScriptId: 'trainer-1',
      trainingJobId: 'job-1', trainingJob: runningJob, modelVersionId: null, currentStep: 'train',
    })
    const jobRead = vi.spyOn(apiClient, 'getTrainingJob')
      .mockResolvedValueOnce(runningJob)
      .mockResolvedValueOnce(succeededJob)
      .mockResolvedValue(succeededJob)
    vi.spyOn(apiClient, 'getDataset').mockResolvedValue(datasetFixture())
    vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(preprocessFixture())
    vi.spyOn(apiClient, 'getDatasetSplit').mockResolvedValue(splitFixture())
    vi.spyOn(apiClient, 'getModel').mockResolvedValue({ ...baseModel(), input_schema: {}, evaluation: null })
    vi.spyOn(apiClient, 'getTrainingJobEvaluation').mockResolvedValue(evaluationFixture())
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    renderApp('/workflow/train')

    await waitFor(() => expect(useAppStore.getState().workflow.trainingJob?.status).toBe('SUCCEEDED'))
    expect(jobRead).toHaveBeenCalledWith('job-1')
    expect(useAppStore.getState().workflow.modelVersionId).toBe('model-1')
  })

  it('keeps a skipped preprocessing task in the split and training request chain', async () => {
    seedWorkflow({ modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture() })
    vi.spyOn(apiClient, 'getDataset').mockResolvedValue(datasetFixture())
    vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(preprocessFixture())
    vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(trainingFixture({ status: 'RUNNING', model_version_id: null }))
    vi.spyOn(apiClient, 'listScripts').mockImplementation(async (params) => params?.script_type === 'preprocessor'
      ? { items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } }
      : { items: [trainerFixture], pagination: { page: 1, page_size: 20, total: 1, total_pages: 1 } })
    const createPreprocess = vi.spyOn(apiClient, 'createPreprocessingTask').mockResolvedValue(preprocessFixture())
    const noSplit = new ApiError('尚未划分', { status: 404 })
    const createdSplit = splitFixture()
    vi.spyOn(apiClient, 'getDatasetSplit').mockImplementation(async () => {
      if (useAppStore.getState().workflow.splitId) return createdSplit
      throw noSplit
    })
    const createSplit = vi.spyOn(apiClient, 'splitDataset').mockResolvedValue(createdSplit)
    const createTraining = vi.spyOn(apiClient, 'createTrainingJob').mockResolvedValue(trainingFixture({ status: 'RUNNING', model_version_id: null }))
    renderApp('/workflow/preprocess')

    fireEvent.click(await screen.findByRole('button', { name: /执行并继续/ }))
    await waitFor(() => expect(createPreprocess).toHaveBeenCalledWith(expect.objectContaining({
      preprocess_script_id: null, mode: 'skip', skip: true,
    })))
    fireEvent.click(await screen.findByRole('button', { name: /生成 80\/20 划分/ }))
    await waitFor(() => expect(createSplit).toHaveBeenCalledWith('dataset-1', 'task-1'))
    fireEvent.click(await screen.findByRole('button', { name: /继续选择训练脚本/ }))
    fireEvent.click(await screen.findByRole('button', { name: /训练脚本/ }))
    fireEvent.click(await screen.findByRole('button', { name: /启动训练/ }))
    await waitFor(() => expect(createTraining).toHaveBeenCalledWith(expect.objectContaining({
      preprocess_script_id: null, preprocessing_task_id: 'task-1',
    })))
  })

  it('redirects to the first available step when a required ID is missing', async () => {
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture(),
      preprocessTask: preprocessFixture(), preprocessTaskId: 'task-1', currentStep: 'publish',
    })
    vi.spyOn(apiClient, 'getDataset').mockResolvedValue(datasetFixture())
    vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(preprocessFixture())
    renderApp('/workflow/publish')

    expect(await screen.findByRole('heading', { name: '数据集划分' })).toBeTruthy()
    expect(screen.getAllByText(/缺少 splitId/).length).toBeGreaterThan(0)
  })

  it('cleans the complete chain when an ID query returns 404', async () => {
    const job = trainingFixture()
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessTaskId: 'task-1', splitId: 'split-1',
      trainingJobId: 'job-1', modelVersionId: 'model-1', dataset: datasetFixture(), preprocessTask: preprocessFixture(),
      split: splitFixture(), trainingJob: job, modelVersion: baseModel(), evaluation: evaluationFixture(),
    })
    vi.spyOn(apiClient, 'getDataset').mockRejectedValue(new ApiError('不存在', { status: 404 }))
    vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(preprocessFixture())
    vi.spyOn(apiClient, 'getDatasetSplit').mockResolvedValue(splitFixture())
    vi.spyOn(apiClient, 'getTrainingJob').mockResolvedValue(job)
    renderApp('/workflow/publish')

    expect(await screen.findByRole('heading', { name: '上传并检查 CSV 数据' })).toBeTruthy()
    expect(useAppStore.getState().workflow).toMatchObject({
      datasetId: null, preprocessTaskId: null, splitId: null, trainingJobId: null, modelVersionId: null,
      dataset: null, preprocessTask: null, split: null, trainingJob: null, modelVersion: null, evaluation: null,
    })
  })

  it('ignores a stale hydration response after switching model type', async () => {
    let resolveDataset!: (value: DatasetUploadResult) => void
    const pendingDataset = new Promise<DatasetUploadResult>((resolve) => { resolveDataset = resolve })
    seedWorkflow({ modelType: 'electric_load', datasetId: 'dataset-1', dataset: datasetFixture() })
    vi.spyOn(apiClient, 'getDataset').mockReturnValue(pendingDataset)
    vi.spyOn(apiClient, 'getPreprocessingTask').mockResolvedValue(preprocessFixture())
    renderApp('/workflow/upload')
    await waitFor(() => expect(apiClient.getDataset).toHaveBeenCalledWith('dataset-1'))

    act(() => useAppStore.getState().setModelType('heating_cooling_load'))
    resolveDataset(datasetFixture({ file_name: '过期.csv' }))
    await waitFor(() => expect(useAppStore.getState().workflow.modelType).toBe('heating_cooling_load'))
    expect(useAppStore.getState().workflow.datasetId).toBeNull()
    expect(useAppStore.getState().workflow.dataset).toBeNull()
  })
})

describe('model registry and MCP acceptance content', () => {
  it('renders version query content and loads version details', async () => {
    const model = baseModel({ id: 'model-v2', version: 'v2' })
    vi.spyOn(apiClient, 'listModels').mockResolvedValue([model])
    vi.spyOn(apiClient, 'listAlerts').mockResolvedValue([])
    vi.spyOn(apiClient, 'getModel').mockResolvedValue({ ...model, previous_healthy_version_id: model.previous_healthy_version_id ?? null, input_schema: { columns: ['time', 'feature', 'target'] }, evaluation: null })
    vi.spyOn(apiClient, 'getModelRollbackRecords').mockResolvedValue([])
    render(<MemoryRouter><ModelVersionsPage /></MemoryRouter>)

    expect(await screen.findByRole('heading', { name: '电力负荷预测 · 版本列表' })).toBeTruthy()
    expect(screen.getByText('v2')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /^v2/ }))
    expect(await screen.findByRole('heading', { name: 'v2' })).toBeTruthy()
    expect(screen.getByText('输入与预处理')).toBeTruthy()
    expect(screen.getByText('暂无与此版本关联的回滚记录。')).toBeTruthy()
  })

  it('documents MCP default-version and error handling rules without claiming an endpoint', () => {
    render(<McpPage />)

    expect(screen.getByText(/当前有效 \+ 最新已发布 \+ 健康/)).toBeTruthy()
    expect(screen.getByText(/指定了版本但不可用时不会静默切换/)).toBeTruthy()
    expect(screen.getByText('MCP 路由待接入')).toBeTruthy()
    expect(screen.getByText('MISSING_FEATURE')).toBeTruthy()
    expect(screen.getByText(/独立 MCP 适配层尚未实现/)).toBeTruthy()
  })
})
