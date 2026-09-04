import type {
  ApiErrorCode,
  ApiErrorResponse,
  ApiResponse,
  DatasetUploadOptions,
  DatasetUploadResult,
  DatasetSplitResult,
  EntityId,
  HealthResponse,
  JsonValue,
  ListAlertsParams,
  ListModelsParams,
  ListScriptsParams,
  ModelAlert,
  ModelEvaluation,
  ModelTypeContract,
  ModelTypeCode,
  ModelVersionDetail,
  ModelVersionSummary,
  LifecycleOperationResponse,
  PaginatedResponse,
  PredictionRequest,
  PredictionResponse,
  PublishModelRequest,
  PublishModelResponse,
  PublishRecord,
  ModelSaveRequest,
  RollbackModelRequest,
  RollbackRecord,
  ScriptContract,
  ScriptUploadInput,
  TrainingJob,
  TrainingLogsResponse,
  CreateTrainingJobRequest,
  PreprocessTask,
} from '../types/contracts'

export const DEFAULT_API_BASE_URL = '/api'

export interface ApiClientOptions {
  /** A backend origin or an already prefixed API URL. */
  baseUrl?: string
  fetchImpl?: typeof fetch
}

export interface RequestOptions {
  query?: Record<string, string | number | boolean | null | undefined>
  signal?: AbortSignal
  headers?: HeadersInit
}

type JsonRequestBody = object | string | number | boolean | null

export class ApiError extends Error {
  readonly status: number
  readonly code: ApiErrorCode | string
  readonly details: JsonValue | null

  constructor(
    message: string,
    options: {
      status: number
      code?: ApiErrorCode | string
      details?: JsonValue | null
    },
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = options.status
    this.code = options.code ?? 'UNKNOWN_ERROR'
    this.details = options.details ?? null
  }
}

export function resolveApiBaseUrl(configuredUrl?: string): string {
  const value = configuredUrl?.trim()
  if (!value) return DEFAULT_API_BASE_URL

  const withoutTrailingSlash = value.replace(/\/+$/, '')
  return /\/api$/i.test(withoutTrailingSlash)
    ? withoutTrailingSlash
    : `${withoutTrailingSlash}/api`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  return (
    isRecord(value) &&
    value.success === false &&
    typeof value.message === 'string' &&
    typeof value.error_code === 'string'
  )
}

function isSuccessEnvelope<T>(value: unknown): value is { success: true; data: T } {
  return isRecord(value) && value.success === true && 'data' in value
}

function asJsonValue(value: unknown): JsonValue | null {
  if (value === null) return null
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  if (Array.isArray(value)) return value as JsonValue[]
  if (isRecord(value)) return value as JsonValue
  return null
}

function toErrorResponse(payload: unknown, status: number): ApiError {
  if (isApiErrorResponse(payload)) {
    return new ApiError(payload.message, {
      status,
      code: payload.error_code,
      details: payload.details,
    })
  }

  if (isRecord(payload)) {
    const detail = isRecord(payload.detail) ? payload.detail : null
    const nestedError = isRecord(payload.error) ? payload.error : null
    const message = (detail?.message ?? nestedError?.message ?? payload.detail)
    const code = detail?.code ?? nestedError?.code
    if (typeof message === 'string') {
      return new ApiError(message, {
        status,
        code: typeof code === 'string' ? code : 'VALIDATION_ERROR',
        details: asJsonValue(payload),
      })
    }
  }

  return new ApiError(status ? `API request failed with status ${status}` : 'Network request failed', {
    status,
    code: status ? 'UNKNOWN_ERROR' : 'NETWORK_ERROR',
    details: asJsonValue(payload),
  })
}

const MODEL_STATUSES: ModelVersionSummary['status'][] = [
  'DRAFT', 'TRAINING', 'READY', 'PUBLISHED', 'RETIRED', 'ABNORMAL', 'FAILED',
]

function normalizeModel(value: unknown): ModelVersionSummary | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_type !== 'string' || typeof value.version !== 'string') {
    return null
  }
  const status = String(value.status ?? 'READY').toUpperCase() as ModelVersionSummary['status']
  const healthStatus = typeof value.health_status === 'string' ? value.health_status.toUpperCase() : typeof value.healthStatus === 'string' ? value.healthStatus.toUpperCase() : 'HEALTHY'
  const metrics = value.metrics === null ? null : isRecord(value.metrics) ? value.metrics as Record<string, JsonValue> : {}
  const trainScript = isRecord(value.train_script) && typeof value.train_script.id === 'string' && typeof value.train_script.name === 'string' && typeof value.train_script.version === 'string'
    ? { id: value.train_script.id, name: value.train_script.name, version: value.train_script.version } : null
  const preprocessScript = isRecord(value.preprocess_script) && typeof value.preprocess_script.id === 'string' && typeof value.preprocess_script.name === 'string' && typeof value.preprocess_script.version === 'string'
    ? { id: value.preprocess_script.id, name: value.preprocess_script.name, version: value.preprocess_script.version } : null
  return {
    id: value.id,
    model_type: value.model_type as ModelVersionSummary['model_type'],
    version: value.version,
    status: MODEL_STATUSES.includes(status) ? status : 'READY',
    health_status: healthStatus,
    is_baseline: value.is_baseline === true,
    is_current: value.is_current === true,
    is_abnormal: value.is_abnormal === true || healthStatus === 'ABNORMAL',
    is_rollback_available: value.is_rollback_available === true,
    metrics,
    model_path: typeof value.model_path === 'string' ? value.model_path : null,
    preprocessor_path: typeof value.preprocessor_path === 'string' ? value.preprocessor_path : null,
    training_job_id: typeof value.training_job_id === 'string' ? value.training_job_id : null,
    train_script_id: typeof value.train_script_id === 'string' ? value.train_script_id : null,
    train_script_version: typeof value.train_script_version === 'string' ? value.train_script_version : null,
    preprocess_script_id: typeof value.preprocess_script_id === 'string' ? value.preprocess_script_id : null,
    preprocess_script_version: typeof value.preprocess_script_version === 'string' ? value.preprocess_script_version : null,
    previous_healthy_version_id: typeof value.previous_healthy_version_id === 'string' ? value.previous_healthy_version_id : null,
    train_script: trainScript,
    preprocess_script: preprocessScript,
    preprocess_used: value.preprocess_used === true,
    feature_columns: Array.isArray(value.feature_columns) ? value.feature_columns.filter((item): item is string => typeof item === 'string') : [],
    time_column: typeof value.time_column === 'string' ? value.time_column : null,
    target_column: typeof value.target_column === 'string' ? value.target_column : null,
    created_at: typeof value.created_at === 'string' ? value.created_at : '',
    published_at: typeof value.published_at === 'string' ? value.published_at : null,
  }
}

function normalizeDetail(value: unknown): ModelVersionDetail | null {
  const model = normalizeModel(value)
  if (!model) return null
  const source = isRecord(value) ? value : {}
  return {
    ...model,
    input_schema: isRecord(source.input_schema) ? source.input_schema as Record<string, JsonValue> : {},
    previous_healthy_version_id: model.previous_healthy_version_id ?? null,
    evaluation: isRecord(source.evaluation) ? source.evaluation as ModelEvaluation : null,
  }
}

function normalizeAlert(value: unknown): ModelAlert | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_type !== 'string') return null
  const status = String(value.status ?? 'ACTIVE').toUpperCase()
  return {
    id: value.id,
    model_type: value.model_type as ModelAlert['model_type'],
    model_version_id: typeof value.model_version_id === 'string' ? value.model_version_id : null,
    reason: typeof value.reason === 'string' ? value.reason : '未提供异常原因',
    rollback_from: typeof value.rollback_from === 'string' ? value.rollback_from : null,
    rollback_to: typeof value.rollback_to === 'string' ? value.rollback_to : null,
    status: status === 'RESOLVED' ? 'RESOLVED' : 'ACTIVE',
    created_at: typeof value.created_at === 'string' ? value.created_at : '',
    resolved_at: typeof value.resolved_at === 'string' ? value.resolved_at : null,
  }
}

function normalizePublishRecord(value: unknown): PublishRecord | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_version_id !== 'string' || typeof value.published_version !== 'string') return null
  return {
    id: value.id,
    model_version_id: value.model_version_id,
    published_version: value.published_version,
    previous_current_version_id: typeof value.previous_current_version_id === 'string' ? value.previous_current_version_id : null,
    published_at: typeof value.published_at === 'string' ? value.published_at : '',
    message: typeof value.message === 'string' ? value.message : null,
  }
}

function normalizeRollback(value: unknown): RollbackRecord | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_type !== 'string') return null
  const from = typeof value.rollback_from === 'string' ? value.rollback_from : typeof value.from_version_id === 'string' ? value.from_version_id : null
  const to = typeof value.rollback_to === 'string' ? value.rollback_to : typeof value.to_version_id === 'string' ? value.to_version_id : null
  return {
    id: value.id,
    model_type: value.model_type as RollbackRecord['model_type'],
    rollback_from: from,
    rollback_to: to,
    from_version_id: from,
    to_version_id: to,
    alert_id: typeof value.alert_id === 'string' ? value.alert_id : null,
    reason: typeof value.reason === 'string' ? value.reason : null,
    status: typeof value.status === 'string' ? value.status.toUpperCase() as RollbackRecord['status'] : undefined,
    created_at: typeof value.created_at === 'string' ? value.created_at : '',
    finished_at: typeof value.finished_at === 'string' ? value.finished_at : null,
  }
}

function normalizeOperation(value: unknown, fallbackOperation: string): LifecycleOperationResponse {
  const source = isRecord(value) ? value : {}
  const model = normalizeDetail(source.model ?? source.current_model ?? value)
  if (!model) throw new ApiError('API 返回的模型版本数据无效', { status: 200, code: 'INVALID_RESPONSE' })
  const operation = typeof source.operation === 'string' ? source.operation : fallbackOperation
  const rollbackSource = source.rollback ?? source.record
  const rollback = rollbackSource === null || rollbackSource === undefined ? null : normalizeRollback(rollbackSource)
  const alert = source.alert === null || source.alert === undefined ? null : normalizeAlert(source.alert)
  return { operation, model, rollback, alert }
}

export class ApiClient {
  readonly baseUrl: string
  private readonly fetchImpl: typeof fetch

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = resolveApiBaseUrl(
      options.baseUrl ?? import.meta.env.VITE_API_BASE_URL,
    )
    this.fetchImpl = options.fetchImpl ?? fetch
  }

  private buildUrl(path: string, query?: RequestOptions['query']): string {
    const rawUrl = `${this.baseUrl}/${path.replace(/^\/+/, '')}`
    const isAbsoluteUrl = /^https?:\/\//i.test(rawUrl)
    const url = new URL(rawUrl, 'http://localhost')
    if (query) {
      Object.entries(query).forEach(([key, value]) => {
        if (value !== undefined && value !== null) url.searchParams.set(key, String(value))
      })
    }
    // Keep same-origin requests relative so Vite's proxy can handle /api.
    return isAbsoluteUrl ? url.toString() : `${url.pathname}${url.search}`
  }

  private async request<T>(
    path: string,
    method: string,
    body?: BodyInit | JsonRequestBody,
    options: RequestOptions = {},
  ): Promise<T> {
    const isJsonBody = body !== undefined && !(body instanceof FormData) && !(body instanceof Blob)
    const headers = new Headers(options.headers)
    headers.set('Accept', 'application/json')
    if (isJsonBody) headers.set('Content-Type', 'application/json')

    let response: Response
    try {
      response = await this.fetchImpl(this.buildUrl(path, options.query), {
        method,
        headers,
        body: isJsonBody ? JSON.stringify(body) : body,
        signal: options.signal,
      })
    } catch (error) {
      if (error instanceof ApiError) throw error
      throw new ApiError(error instanceof Error ? error.message : 'Network request failed', {
        status: 0,
        code: 'NETWORK_ERROR',
      })
    }

    let payload: unknown = null
    if (response.status !== 204) {
      const contentType = response.headers.get('content-type') ?? ''
      payload = contentType.includes('application/json')
        ? await response.json().catch(() => null)
        : await response.text()
    }

    if (!response.ok || isApiErrorResponse(payload)) {
      throw toErrorResponse(payload, response.status)
    }

    if (isSuccessEnvelope<T>(payload)) return payload.data
    return payload as T
  }

  get<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, 'GET', undefined, options)
  }

  postJson<TResponse, TRequest extends JsonRequestBody = JsonRequestBody>(
    path: string,
    body?: TRequest,
    options?: RequestOptions,
  ): Promise<TResponse> {
    return this.request<TResponse>(path, 'POST', body, options)
  }

  postForm<T>(path: string, formData: FormData, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, 'POST', formData, options)
  }

  getHealth(options?: RequestOptions): Promise<HealthResponse> {
    return this.get<HealthResponse>('health', options)
  }

  listModelTypes(options?: RequestOptions): Promise<ModelTypeContract[]> {
    return this.get<ModelTypeContract[]>('model-types', options)
  }

  uploadDataset(file: File, options: DatasetUploadOptions = {}): Promise<DatasetUploadResult> {
    const formData = new FormData()
    formData.append('file', file)
    if (options.model_type) formData.append('model_type', options.model_type)
    return this.postForm<DatasetUploadResult>('datasets/upload', formData)
  }

  createPreprocessingTask(input: {
    model_type: ModelTypeCode
    dataset_id: EntityId
    preprocess_script_id?: EntityId | null
    mode?: 'use' | 'skip'
    skip?: boolean
    config?: Record<string, JsonValue>
  }): Promise<PreprocessTask> {
    return this.postJson<PreprocessTask>('preprocessing-tasks', input)
  }

  getPreprocessingTask(id: EntityId): Promise<PreprocessTask> {
    return this.get<PreprocessTask>(`preprocessing-tasks/${encodeURIComponent(id)}`)
  }

  splitDataset(datasetId: EntityId, preprocessingTaskId?: EntityId | null): Promise<DatasetSplitResult> {
    return this.postJson<DatasetSplitResult>(`datasets/${encodeURIComponent(datasetId)}/split`,
      preprocessingTaskId ? { preprocessing_task_id: preprocessingTaskId } : {})
  }

  getDatasetSplit(datasetId: EntityId): Promise<DatasetSplitResult> {
    return this.get<DatasetSplitResult>(`datasets/${encodeURIComponent(datasetId)}/split`)
  }

  listScripts(params?: ListScriptsParams): Promise<PaginatedResponse<ScriptContract>> {
    return this.get<PaginatedResponse<ScriptContract>>('scripts', {
      query: params as RequestOptions['query'],
    })
  }

  getScript(id: EntityId): Promise<ScriptContract> {
    return this.get<ScriptContract>(`scripts/${encodeURIComponent(id)}`)
  }

  enableScript(id: EntityId): Promise<ScriptContract> {
    return this.postJson<ScriptContract>(`scripts/${encodeURIComponent(id)}/enable`)
  }

  disableScript(id: EntityId): Promise<ScriptContract> {
    return this.postJson<ScriptContract>(`scripts/${encodeURIComponent(id)}/disable`)
  }

  uploadScript(input: ScriptUploadInput): Promise<ScriptContract> {
    const formData = new FormData()
    formData.append('file', input.file)
    formData.append('name', input.name)
    formData.append('script_type', input.script_type)
    formData.append('supported_model_types', JSON.stringify(input.supported_model_types))
    if (input.version) formData.append('version', input.version)
    return this.postForm<ScriptContract>('scripts/upload', formData)
  }

  createTrainingJob(input: CreateTrainingJobRequest): Promise<TrainingJob> {
    return this.postJson<TrainingJob, CreateTrainingJobRequest>('training-jobs', input)
  }

  retryTrainingJob(id: EntityId): Promise<TrainingJob> {
    return this.postJson<TrainingJob>(`training-jobs/${encodeURIComponent(id)}/retry`)
  }

  getTrainingJob(id: EntityId): Promise<TrainingJob> {
    return this.get<TrainingJob>(`training-jobs/${encodeURIComponent(id)}`)
  }

  getTrainingJobLogs(
    id: EntityId,
    params?: { since?: string; limit?: number },
  ): Promise<TrainingLogsResponse> {
    return this.get<TrainingLogsResponse>(`training-jobs/${encodeURIComponent(id)}/logs`, {
      query: params,
    })
  }

  getTrainingJobEvaluation(id: EntityId): Promise<ModelEvaluation> {
    return this.get<ModelEvaluation>(`training-jobs/${encodeURIComponent(id)}/evaluation`)
  }

  saveModel(id: EntityId, input: ModelSaveRequest): Promise<ModelVersionSummary> {
    return this.postJson<ModelVersionSummary, ModelSaveRequest>(
      `models/${encodeURIComponent(id)}/save`, input)
  }

  publishModel(
    id: EntityId,
    input: PublishModelRequest = {},
  ): Promise<PublishModelResponse> {
    return this.postJson<unknown>(
      `models/${encodeURIComponent(id)}/publish`,
      { ...input, confirmed: input.confirmed ?? input.confirm ?? false },
    ).then((payload) => {
      const operation = normalizeOperation(payload, 'publish')
      const source = isRecord(payload) ? payload : {}
      const record = normalizePublishRecord(source.record)
      return { model: operation.model, operation: operation.operation, ...(record ? { record } : {}) } satisfies PublishModelResponse
    })
  }

  listModels(params?: ListModelsParams): Promise<ModelVersionSummary[]> {
    return this.get<unknown>('models', {
      query: params as RequestOptions['query'],
    }).then((payload) => {
      const rows = Array.isArray(payload)
        ? payload
        : isRecord(payload) && Array.isArray(payload.items) ? payload.items : []
      return rows.map(normalizeModel).filter((item): item is ModelVersionSummary => item !== null)
    })
  }

  getModel(id: EntityId): Promise<ModelVersionDetail> {
    return this.get<unknown>(`models/${encodeURIComponent(id)}`).then((payload) => {
      const model = normalizeDetail(payload)
      if (!model) throw new ApiError('API 返回的模型版本详情无效', { status: 200, code: 'INVALID_RESPONSE' })
      return model
    })
  }

  /** The backend accepts either a target id/version or an empty body (path target). */
  rollbackModel(
    id: EntityId,
    input: RollbackModelRequest = {},
  ): Promise<LifecycleOperationResponse> {
    return this.postJson<unknown>(
      `models/${encodeURIComponent(id)}/rollback`,
      input,
    ).then((payload) => normalizeOperation(payload, 'rollback'))
  }

  /** Offline is the backend's name for retiring a published version. */
  offlineModel(id: EntityId): Promise<LifecycleOperationResponse> {
    return this.postJson<unknown>(`models/${encodeURIComponent(id)}/offline`)
      .then((payload) => normalizeOperation(payload, 'offline'))
  }

  getModelRollbackRecords(id: EntityId): Promise<RollbackRecord[]> {
    return this.get<unknown>(`models/${encodeURIComponent(id)}/rollback-records`).then((payload) => {
      const rows = Array.isArray(payload) ? payload : isRecord(payload) && Array.isArray(payload.items) ? payload.items : []
      return rows.map(normalizeRollback).filter((item): item is RollbackRecord => item !== null)
    })
  }

  getModelPublishRecords(id: EntityId): Promise<PublishRecord[]> {
    return this.get<unknown>(`models/${encodeURIComponent(id)}/publish-records`).then((payload) => {
      const rows = Array.isArray(payload) ? payload : isRecord(payload) && Array.isArray(payload.items) ? payload.items : []
      return rows.map(normalizePublishRecord).filter((item): item is PublishRecord => item !== null)
    })
  }

  /** Uses the implemented lifecycle endpoint, rather than the undocumented MCP alias. */
  markModelAbnormal(id: EntityId, reason = '健康检查异常'): Promise<LifecycleOperationResponse> {
    return this.postJson<unknown>(`models/${encodeURIComponent(id)}/abnormal`, { reason })
      .then((payload) => normalizeOperation(payload, 'abnormal'))
  }

  listAlerts(params?: ListAlertsParams): Promise<ModelAlert[]> {
    return this.get<unknown>('alerts', {
      query: params as RequestOptions['query'],
    }).then((payload) => {
      const rows = Array.isArray(payload)
        ? payload
        : isRecord(payload) && Array.isArray(payload.items) ? payload.items : []
      return rows.map(normalizeAlert).filter((item): item is ModelAlert => item !== null)
    })
  }

  predict(input: PredictionRequest): Promise<PredictionResponse> {
    return this.postJson<PredictionResponse, PredictionRequest>('predict', input)
  }
}

export const apiClient = new ApiClient()

export type ApiResult<T> = ApiResponse<T>
