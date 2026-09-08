import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ApiError, apiClient } from '../api'
import type { AuditEvent, AuditEventsResponse } from '../types/contracts'
import { AuditPage } from './AuditPage'

const eventFixture = (overrides: Partial<AuditEvent> = {}): AuditEvent => ({
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
  result: 'SUCCEEDED',
  message: null,
  request_id: null,
  correlation_id: null,
  metadata: {},
  ...overrides,
})

const responseFixture = (overrides: Partial<AuditEventsResponse> = {}): AuditEventsResponse => ({
  items: [eventFixture()],
  page: 1,
  page_size: 50,
  total: 1,
  has_next: false,
  ...overrides,
})

function renderAudit() {
  return render(<MemoryRouter initialEntries={['/audit']}><AuditPage /></MemoryRouter>)
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('AuditPage', () => {
  it('requests the default page and renders the five audit columns with source values', async () => {
    const listAuditEvents = vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue(responseFixture({
      items: [eventFixture({
        occurred_at: '2026-01-03T04:05:06+08:00',
        operator_name: null,
        operator_id: null,
        message: '模型版本已发布到生产',
      })],
    }))

    renderAudit()

    expect(await screen.findByText('MODEL_PUBLISHED')).toBeTruthy()
    expect(listAuditEvents).toHaveBeenCalledWith({ page: 1, page_size: 50 })
    expect(screen.getByText('2026-01-03T04:05:06+08:00')).toBeTruthy()
    expect(screen.getByText('MODEL_VERSION')).toBeTruthy()
    expect(screen.getByText('model-1')).toBeTruthy()
    expect(screen.getByText('SUCCEEDED')).toBeTruthy()
    expect(screen.getByText('未知')).toBeTruthy()
    expect(screen.getByText('模型版本已发布到生产')).toBeTruthy()
    for (const column of ['时间', '事件类型', '对象', '操作人', '结果']) {
      expect(screen.getByRole('columnheader', { name: column })).toBeTruthy()
    }
    expect(screen.queryByRole('button', { name: /编辑|删除/ })).toBeNull()
  })

  it('maps event filters to backend event types and resets pagination to page one', async () => {
    const listAuditEvents = vi.spyOn(apiClient, 'listAuditEvents').mockImplementation(async (params = {}) => responseFixture({
      items: [eventFixture({ event_type: params.event_type ?? 'MODEL_PUBLISHED' })],
      page: params.page ?? 1,
      total: 101,
      has_next: (params.page ?? 1) === 1,
    }))

    renderAudit()
    await screen.findByText('MODEL_PUBLISHED')

    fireEvent.click(screen.getByRole('button', { name: '下一页' }))
    await waitFor(() => expect(listAuditEvents).toHaveBeenLastCalledWith({ page: 2, page_size: 50 }))

    fireEvent.change(screen.getByLabelText('事件类型'), { target: { value: 'MODEL_ROLLBACK_SUCCEEDED' } })
    await waitFor(() => expect(listAuditEvents).toHaveBeenLastCalledWith({
      page: 1,
      page_size: 50,
      event_type: 'MODEL_ROLLBACK_SUCCEEDED',
    }))
    expect((screen.getByRole('button', { name: '上一页' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('uses backend total and has_next to control pagination', async () => {
    const listAuditEvents = vi.spyOn(apiClient, 'listAuditEvents').mockImplementation(async (params = {}) => responseFixture({
      total: 123,
      has_next: (params.page ?? 1) === 1,
      page: params.page ?? 1,
    }))

    renderAudit()
    await screen.findByText((_, element) => element?.textContent === '共 123 条 · 第 1 页')
    const next = screen.getByRole('button', { name: '下一页' }) as HTMLButtonElement
    expect(next.disabled).toBe(false)

    fireEvent.click(next)
    await waitFor(() => expect(listAuditEvents).toHaveBeenLastCalledWith({ page: 2, page_size: 50 }))
    await waitFor(() => expect((screen.getByRole('button', { name: '下一页' }) as HTMLButtonElement).disabled).toBe(true))
    expect(screen.getByText((_, element) => element?.textContent === '共 123 条 · 第 2 页')).toBeTruthy()
  })

  it('links only known object resources with their real IDs', async () => {
    vi.spyOn(apiClient, 'listAuditEvents').mockResolvedValue(responseFixture({
      items: [
        eventFixture({ id: 'model-event', object_type: 'MODEL_VERSION', object_id: 'model-1', model_type: 'electric_load' }),
        eventFixture({ id: 'job-event', object_type: 'TRAINING_JOB', object_id: 'job-1', training_job_id: 'job-1' }),
        eventFixture({ id: 'dataset-event', object_type: 'DATASET', object_id: 'dataset-1' }),
        eventFixture({ id: 'task-event', object_type: 'PREPROCESSING_TASK', object_id: 'task-1' }),
        eventFixture({ id: 'split-event', object_type: 'DATASET_SPLIT', object_id: 'split-1' }),
        eventFixture({ id: 'alert-event', object_type: 'ALERT', object_id: 'alert-1' }),
        eventFixture({ id: 'other-event', object_type: 'OTHER', object_id: 'other-1' }),
      ],
      total: 7,
    }))

    renderAudit()

    expect(await screen.findByText('other-1')).toBeTruthy()
    expect(screen.getByRole('link', { name: 'model-1' }).getAttribute('href')).toBe('/models?model_type=electric_load#model-1')
    expect(screen.getByRole('link', { name: 'job-1' }).getAttribute('href')).toBe('/workflow/train?training_job_id=job-1')
    expect(screen.getByRole('link', { name: 'dataset-1' }).getAttribute('href')).toBe('/workflow/upload?dataset_id=dataset-1')
    expect(screen.getByRole('link', { name: 'task-1' }).getAttribute('href')).toBe('/workflow/preprocess?preprocessing_task_id=task-1')
    expect(screen.getByRole('link', { name: 'split-1' }).getAttribute('href')).toBe('/workflow/split?split_id=split-1')
    expect(screen.queryByRole('link', { name: 'alert-1' })).toBeNull()
    expect(screen.queryByRole('link', { name: 'other-1' })).toBeNull()
  })

  it('shows an explicit loading state, request errors, and empty state without throwing', async () => {
    let resolveRequest!: (value: AuditEventsResponse) => void
    const pending = new Promise<AuditEventsResponse>((resolve) => { resolveRequest = resolve })
    vi.spyOn(apiClient, 'listAuditEvents').mockReturnValueOnce(pending)
    renderAudit()
    expect(screen.getByRole('status').textContent).toContain('正在加载审计事件')
    resolveRequest(responseFixture({ items: [], total: 0 }))
    expect(await screen.findByText('暂无审计事件')).toBeTruthy()

    cleanup()
    vi.restoreAllMocks()
    vi.spyOn(apiClient, 'listAuditEvents').mockRejectedValue(new ApiError('审计服务不可用', { status: 503 }))
    renderAudit()
    expect((await screen.findByRole('alert')).textContent).toContain('审计服务不可用')
    expect((screen.getByRole('button', { name: '下一页' }) as HTMLButtonElement).disabled).toBe(true)
  })
})
