import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from './App'
import { apiClient } from './api'
import { McpPage } from './pages/McpPage'
import { ModelVersionsPage } from './pages/ModelVersionsPage'
import { useAppStore } from './store/useAppStore'
import type { ModelVersionSummary } from './types/contracts'

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
  training_job_id: null,
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

  it('shows the explicit unused-preprocessing state when skip is selected', async () => {
    seedWorkflow({
      modelType: 'electric_load',
      datasetId: 'dataset-1',
      preprocessTaskId: 'task-1',
      preprocessTask: {
        id: 'task-1', dataset_id: 'dataset-1', preprocess_script_id: null,
        preprocess_used: false, preprocess_status: 'unused', status: 'skipped',
        stage: 'completed', logs: ['未使用预处理，后续使用原始特征'],
        created_at: '2026-01-01T00:00:00Z', data_source: 'raw',
      },
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    renderApp('/workflow/preprocess')

    expect(await screen.findByText('未使用预处理，后续使用原始特征')).toBeTruthy()
    expect(screen.getByText('预处理已跳过')).toBeTruthy()
  })

  it('shows failed-training retry guidance and submits a retry', async () => {
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessTaskId: 'task-1',
      split: { id: 'split-1', dataset_id: 'dataset-1', preprocessing_task_id: 'task-1', data_source: 'raw', split_strategy: 'time_ordered', split_ratio: 0.8, test_ratio: 0.2, total_row_count: 5, train_row_count: 4, test_row_count: 1, train_time_range: { start: '2024-01-01', end: '2024-01-04' }, test_time_range: { start: '2024-01-05', end: '2024-01-05' }, train_time_start: '2024-01-01', train_time_end: '2024-01-04', test_time_start: '2024-01-05', test_time_end: '2024-01-05', created_at: '2026-01-01T00:00:00Z' },
      trainScriptId: 'trainer-1', trainingJobId: 'job-1',
      trainingJob: { id: 'job-1', model_type: 'electric_load', dataset_id: 'dataset-1', preprocess_script_id: null, train_script_id: 'trainer-1', status: 'FAILED', progress_stage: 'FAILED', current_stage: '失败', logs: ['FAILED：训练脚本失败'], error_message: '训练脚本失败', created_at: '2026-01-01T00:00:00Z', finished_at: '2026-01-01T00:01:00Z' },
    })
    vi.spyOn(apiClient, 'listScripts').mockResolvedValue({ items: [], pagination: { page: 1, page_size: 20, total: 0, total_pages: 0 } })
    const retry = vi.spyOn(apiClient, 'retryTrainingJob').mockResolvedValue({
      ...useAppStore.getState().workflow.trainingJob!, status: 'PENDING', error_message: null,
    })
    renderApp('/workflow/train')

    expect(await screen.findByText('本次训练失败，不会影响生产模型。修复脚本或配置后可以重试。')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '失败重试' }))
    await waitFor(() => expect(retry).toHaveBeenCalledWith('job-1'))
  })

  it('requires browser confirmation before publishing a candidate', async () => {
    seedWorkflow({
      modelType: 'electric_load', datasetId: 'dataset-1', preprocessTaskId: 'task-1',
      split: { id: 'split-1', dataset_id: 'dataset-1', preprocessing_task_id: 'task-1', data_source: 'raw', split_strategy: 'time_ordered', split_ratio: 0.8, test_ratio: 0.2, total_row_count: 5, train_row_count: 4, test_row_count: 1, train_time_range: { start: '2024-01-01', end: '2024-01-04' }, test_time_range: { start: '2024-01-05', end: '2024-01-05' }, train_time_start: '2024-01-01', train_time_end: '2024-01-04', test_time_start: '2024-01-05', test_time_end: '2024-01-05', created_at: '2026-01-01T00:00:00Z' },
      trainingJobId: 'job-1', trainingJob: { id: 'job-1', model_type: 'electric_load', dataset_id: 'dataset-1', preprocess_script_id: null, train_script_id: 'trainer-1', status: 'SUCCEEDED', logs: [], error_message: null, created_at: '2026-01-01T00:00:00Z', finished_at: '2026-01-01T00:01:00Z' },
      modelVersion: baseModel(),
    })
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    const publish = vi.spyOn(apiClient, 'publishModel')
    renderApp('/workflow/publish')

    fireEvent.click(screen.getByRole('button', { name: '发布模型' }))

    expect(confirm).toHaveBeenCalledWith('发布后将成为当前生产模型，是否确认发布？')
    expect(publish).not.toHaveBeenCalled()
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
