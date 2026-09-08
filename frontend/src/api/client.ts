import type {
  ApiErrorCode,
  ApiErrorResponse,
  ApiResponse,
  AlertAcknowledgeRequest,
  AlertAcknowledgeResponse,
  AlertStatistics,
  AuditEvent,
  AuditEventsResponse,
  DatasetUploadOptions,
  DatasetUploadResult,
  DatasetSplitResult,
  EntityId,
  HealthResponse,
  HealthStatus,
  JsonValue,
  ListAlertsParams,
  ListAuditEventsParams,
  ListModelsParams,
  ListScriptsParams,
  ModelAlert,
  McpCapabilities,
  ModelEvaluation,
  ModelTypeCode,
  ModelVersionDetail,
  ModelVersionSummary,
  LifecycleOperationResponse,
  PaginatedResponse,
  PredictionRequest,
  PredictionResponse,
  PublishModelInput,
  PublishModelResponse,
  PublishRecord,
  ModelSaveRequest,
  RollbackModelInput,
  RollbackRecord,
  ScriptContract,
  ScriptUploadInput,
  TrainingJob,
  TrainingLogsParams,
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

const MCP_OPENAPI_PATHS = {
  predict: '/api/mcp/predict',
  markModelAbnormal: '/api/mcp/mark_model_abnormal',
} as const

function hasDeclaredPost(pathItem: unknown): boolean {
  return isRecord(pathItem) && isRecord(pathItem.post)
}

/** Only a formal POST declaration for both documented paths makes MCP available. */
export function inspectMcpOpenApi(document: unknown): McpCapabilities {
  const paths = isRecord(document) && isRecord(document.paths) ? document.paths : null
  const predict = Boolean(paths && hasDeclaredPost(paths[MCP_OPENAPI_PATHS.predict]))
  const markModelAbnormal = Boolean(paths && hasDeclaredPost(paths[MCP_OPENAPI_PATHS.markModelAbnormal]))
  return {
    available: predict && markModelAbnormal,
    predict,
    markModelAbnormal,
  }
}

function toErrorResponse(payload: unknown, status: number): ApiError {
  if (isApiErrorResponse(payload)) {
    return new ApiError(payload.message, {
      status,
      code: payload.error_code,
      details: payload.details ?? null,
    })
  }

  if (isRecord(payload)) {
    // FastAPI raises HTTPException with ``detail={code, message, ...}``,
    // whereas validation errors use a detail array. Keep the complete wire
    // payload in details so callers never lose status/code/field information.
    const detail = isRecord(payload.detail) ? payload.detail : null
    const nestedError = isRecord(payload.error) ? payload.error : null
    const message = detail?.message ?? nestedError?.message
      ?? (typeof payload.detail === 'string' ? payload.detail : payload.message)
    const code = detail?.code ?? nestedError?.code ?? payload.error_code ?? payload.code
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

/** Create a stable request key when a page does not provide one. */
export function createIdempotencyKey(prefix = 'model-operation'): string {
  const randomUuid = globalThis.crypto?.randomUUID
  if (typeof randomUuid === 'function') return `${prefix}:${randomUuid.call(globalThis.crypto)}`
  return `${prefix}:${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

const MODEL_STATUSES: Exclude<ModelVersionSummary['status'], null>[] = [
  'DRAFT', 'TRAINING', 'READY', 'PUBLISHED', 'RETIRED', 'FAILED',
]
const HEALTH_STATUSES = new Set<HealthStatus>(['HEALTHY', 'ABNORMAL', 'UNKNOWN'])

function normalizeModel(value: unknown): ModelVersionSummary | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_type !== 'string' || typeof value.version !== 'string') {
    return null
  }
  const rawStatus = typeof value.status === 'string' ? value.status.toUpperCase() : null
  const isCurrent = value.is_current === true
  // ABNORMAL was historically mixed into lifecycle. It is health evidence,
  // not enough evidence to invent a lifecycle state.
  const status = rawStatus === 'ABNORMAL'
    ? null
    : rawStatus as ModelVersionSummary['status']
  const rawHealth = typeof value.health_status === 'string'
    ? value.health_status.toUpperCase()
    : typeof value.healthStatus === 'string' ? value.healthStatus.toUpperCase() : 'UNKNOWN'
  const healthStatus = rawStatus === 'ABNORMAL'
    ? 'ABNORMAL'
    : HEALTH_STATUSES.has(rawHealth as HealthStatus)
      ? rawHealth as HealthStatus
      : 'UNKNOWN'
  const metrics = value.metrics === null ? null : isRecord(value.metrics) ? value.metrics as Record<string, JsonValue> : {}
  const trainScript = isRecord(value.train_script) && typeof value.train_script.id === 'string' && typeof value.train_script.name === 'string' && typeof value.train_script.version === 'string'
    ? { id: value.train_script.id, name: value.train_script.name, version: value.train_script.version } : null
  const preprocessScript = isRecord(value.preprocess_script) && typeof value.preprocess_script.id === 'string' && typeof value.preprocess_script.name === 'string' && typeof value.preprocess_script.version === 'string'
    ? { id: value.preprocess_script.id, name: value.preprocess_script.name, version: value.preprocess_script.version } : null
  return {
    id: value.id,
    model_type: value.model_type as ModelVersionSummary['model_type'],
    version: value.version,
    status: status && MODEL_STATUSES.includes(status) ? status : null,
    health_status: healthStatus,
    is_baseline: value.is_baseline === true,
    is_current: isCurrent,
    is_abnormal: value.is_abnormal === true || healthStatus === 'ABNORMAL' || rawStatus === 'ABNORMAL',
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
    evaluation: isRecord(source.evaluation) ? source.evaluation as unknown as ModelEvaluation : null,
  }
}

function normalizeAlertStatistics(value: unknown): AlertStatistics | null {
  if (!isRecord(value)) return null
  const values = ['total', 'active', 'acknowledged', 'resolved']
  if (!values.every((key) => typeof value[key] === 'number' && Number.isFinite(value[key]))) return null
  return {
    total: value.total as number,
    active: value.active as number,
    acknowledged: value.acknowledged as number,
    resolved: value.resolved as number,
  }
}

function normalizeAlert(value: unknown): ModelAlert | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_type !== 'string') return null
  const rawStatus = String(value.status ?? '').toUpperCase()
  const status = rawStatus === 'ACTIVE' || rawStatus === 'ACKNOWLEDGED' || rawStatus === 'RESOLVED'
    ? rawStatus
    : 'UNKNOWN'
  return {
    id: value.id,
    model_type: value.model_type as ModelAlert['model_type'],
    model_version_id: typeof value.model_version_id === 'string' ? value.model_version_id : null,
    reason: typeof value.reason === 'string' ? value.reason : '未提供异常原因',
    rollback_from: typeof value.rollback_from === 'string' ? value.rollback_from : null,
    rollback_to: typeof value.rollback_to === 'string' ? value.rollback_to : null,
    status,
    created_at: typeof value.created_at === 'string' ? value.created_at : '',
    acknowledged_at: typeof value.acknowledged_at === 'string' ? value.acknowledged_at : null,
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
    reason: typeof value.reason === 'string' ? value.reason : null,
    idempotency_key: typeof value.idempotency_key === 'string' ? value.idempotency_key : null,
    message: typeof value.message === 'string' ? value.message : null,
  }
}

function normalizeRollback(value: unknown): RollbackRecord | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.model_type !== 'string') return null
  const from = typeof value.rollback_from === 'string' ? value.rollback_from : typeof value.from_version_id === 'string' ? value.from_version_id : null
  const to = typeof value.rollback_to === 'string' ? value.rollback_to : typeof value.to_version_id === 'string' ? value.to_version_id : null
  const status = typeof value.status === 'string' ? value.status.toUpperCase() : ''
  if (status !== 'PENDING' && status !== 'SUCCEEDED' && status !== 'FAILED') return null
  return {
    id: value.id,
    model_type: value.model_type as RollbackRecord['model_type'],
    rollback_from: from,
    rollback_to: to,
    from_version_id: from,
    to_version_id: to,
    alert_id: typeof value.alert_id === 'string' ? value.alert_id : null,
    reason: typeof value.reason === 'string' ? value.reason : null,
    idempotency_key: typeof value.idempotency_key === 'string' ? value.idempotency_key : null,
    status,
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

function normalizeAuditEvent(value: unknown): AuditEvent | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.occurred_at !== 'string'
    || typeof value.event_type !== 'string' || typeof value.object_type !== 'string'
    || typeof value.result !== 'string') return null
  return {
    id: value.id,
    occurred_at: value.occurred_at,
    event_type: value.event_type,
    object_type: value.object_type,
    object_id: typeof value.object_id === 'string' ? value.object_id : null,
    model_type: typeof value.model_type === 'string' ? value.model_type as AuditEvent['model_type'] : null,
    model_version_id: typeof value.model_version_id === 'string' ? value.model_version_id : null,
    training_job_id: typeof value.training_job_id === 'string' ? value.training_job_id : null,
    operator_type: typeof value.operator_type === 'string' ? value.operator_type : null,
    operator_id: typeof value.operator_id === 'string' ? value.operator_id : null,
    operator_name: typeof value.operator_name === 'string' ? value.operator_name : null,
    result: value.result,
    message: typeof value.message === 'string' ? value.message : null,
    request_id: typeof value.request_id === 'string' ? value.request_id : null,
    correlation_id: typeof value.correlation_id === 'string' ? value.correlation_id : null,
    metadata: isRecord(value.metadata) ? value.metadata as AuditEvent['metadata'] : {},
  }
}

export type CompatibleArrayResponse<T> = T[] & {
  items?: T[]
  page?: number | null
  page_size?: number | null
  total?: number
  has_next?: boolean
  next_cursor?: string | null
  statistics?: AlertStatistics
}

function compatibleArrayResponse<T>(rows: T[], payload: unknown): CompatibleArrayResponse<T> {
  const result = rows as CompatibleArrayResponse<T>
  if (isRecord(payload) && Array.isArray(payload.items)) {
    const page = typeof payload.page === 'number' ? payload.page : null
    const pageSize = typeof payload.page_size === 'number' ? payload.page_size : null
    const total = typeof payload.total === 'number' ? payload.total : rows.length
    const nextCursor = typeof payload.next_cursor === 'string' ? payload.next_cursor : null
    const hasNext = typeof payload.has_next === 'boolean'
      ? payload.has_next
      : Boolean(nextCursor) || (page !== null && pageSize !== null && page * pageSize < total)
    // Non-enumerable properties keep old ``array.map`` consumers and their
    // equality expectations intact while exposing page metadata to newer UI.
    Object.defineProperties(result, {
      items: { value: result, enumerable: false },
      page: { value: page, enumerable: false },
      page_size: { value: pageSize, enumerable: false },
      total: { value: total, enumerable: false },
      has_next: { value: hasNext, enumerable: false },
      next_cursor: { value: nextCursor, enumerable: false },
      statistics: { value: normalizeAlertStatistics(payload.statistics) ?? undefined, enumerable: false },
    })
  }
  return result
}

export class ApiClient {
  readonly baseUrl: string
  private readonly fetchImpl: typeof fetch

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = resolveApiBaseUrl(
      options.baseUrl ?? import.meta.env.VITE_API_BASE_URL,
    )
    // `fetch` is a Web IDL method and must retain its global receiver in
    // browsers. Store a bound default while leaving injected test adapters
    // untouched.
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis)
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

  async getHealth(options?: RequestOptions): Promise<HealthResponse> {
    // /health is the declared contract. The fallback is only for deployments
    // that expose the legacy /api/health alias; it also avoids /api/api/health.
    try {
      return await this.get<HealthResponse>('../health', options)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return this.get<HealthResponse>('health', options)
      }
      throw error
    }
  }

  uploadDataset(file: File, _options: DatasetUploadOptions = {}): Promise<DatasetUploadResult> {
    // model_type belongs to the workflow context. The upload OpenAPI contract
    // declares only the multipart ``file`` field.
    const formData = new FormData()
    formData.append('file', file)
    return this.postForm<DatasetUploadResult>('datasets/upload', formData)
  }

  /** Read persisted dataset metadata; upload responses are not durable UI state. */
  getDataset(id: EntityId): Promise<DatasetUploadResult> {
    return this.get<DatasetUploadResult>(`datasets/${encodeURIComponent(id)}`)
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

  getTrainingJobLogs(id: EntityId, params?: TrainingLogsParams): Promise<TrainingLogsResponse> {
    return this.get<TrainingLogsResponse>(`training-jobs/${encodeURIComponent(id)}/logs`, {
      query: params as RequestOptions['query'],
    })
  }

  cancelTrainingJob(id: EntityId): Promise<TrainingJob> {
    return this.postJson<TrainingJob>(`training-jobs/${encodeURIComponent(id)}/cancel`)
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
    input: PublishModelInput = { confirmed: false },
  ): Promise<PublishModelResponse> {
    const compatibility = input as PublishModelInput & {
      confirm?: boolean
      confirmation?: boolean
      message?: string | null
    }
    const reason = input.reason ?? compatibility.message ?? null
    const body = {
      confirmed: input.confirmed ?? compatibility.confirm ?? compatibility.confirmation ?? false,
      reason,
      idempotency_key: input.idempotency_key ?? createIdempotencyKey('publish'),
    }
    return this.postJson<unknown>(
      `models/${encodeURIComponent(id)}/publish`,
      body,
    ).then((payload) => {
      const operation = normalizeOperation(payload, 'publish')
      const source = isRecord(payload) ? payload : {}
      const record = normalizePublishRecord(source.record)
      return { model: operation.model, operation: operation.operation, ...(record ? { record } : {}) } satisfies PublishModelResponse
    })
  }

  listModels(params?: ListModelsParams): Promise<CompatibleArrayResponse<ModelVersionSummary>> {
    return this.get<unknown>('models', {
      query: params as RequestOptions['query'],
    }).then((payload) => {
      const rows = Array.isArray(payload)
        ? payload
        : isRecord(payload) && Array.isArray(payload.items) ? payload.items : []
      const normalized = rows.map(normalizeModel).filter((item): item is ModelVersionSummary => item !== null)
      return compatibleArrayResponse(normalized, payload)
    })
  }

  getModel(id: EntityId): Promise<ModelVersionDetail> {
    return this.get<unknown>(`models/${encodeURIComponent(id)}`).then((payload) => {
      const model = normalizeDetail(payload)
      if (!model) throw new ApiError('API 返回的模型版本详情无效', { status: 200, code: 'INVALID_RESPONSE' })
      return model
    })
  }

  /** Compatibility aliases are adapted here; the path id is never a target. */
  rollbackModel(
    id: EntityId,
    input: RollbackModelInput = {},
  ): Promise<LifecycleOperationResponse> {
    const compatibility = input as RollbackModelInput & {
      version_id?: EntityId
      version?: string
    }
    const targetVersionId = input.target_version_id ?? compatibility.version_id ?? compatibility.version
    const reason = input.reason?.trim()
    if (!targetVersionId) {
      return Promise.reject(new ApiError('回滚必须显式提供 target_version_id', {
        status: 400,
        code: 'ROLLBACK_TARGET_REQUIRED',
        details: { field: 'target_version_id' },
      }))
    }
    if (!reason) {
      return Promise.reject(new ApiError('回滚必须提供 reason', {
        status: 400,
        code: 'VALIDATION_ERROR',
        details: { field: 'reason' },
      }))
    }
    const body = {
      target_version_id: targetVersionId,
      reason,
      idempotency_key: input.idempotency_key ?? createIdempotencyKey('rollback'),
    }
    return this.postJson<unknown>(
      `models/${encodeURIComponent(id)}/rollback`,
      body,
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
  markModelAbnormal(id: EntityId, reason: string): Promise<LifecycleOperationResponse> {
    if (!reason.trim()) {
      return Promise.reject(new ApiError('标记异常必须提供 reason', {
        status: 400,
        code: 'VALIDATION_ERROR',
        details: { field: 'reason' },
      }))
    }
    return this.postJson<unknown>(`models/${encodeURIComponent(id)}/abnormal`, { reason })
      .then((payload) => normalizeOperation(payload, 'abnormal'))
  }

  getAlert(id: EntityId): Promise<ModelAlert> {
    return this.get<unknown>(`alerts/${encodeURIComponent(id)}`).then((payload) => {
      const alert = normalizeAlert(payload)
      if (!alert) throw new ApiError('API 返回的告警数据无效', { status: 200, code: 'INVALID_RESPONSE' })
      return alert
    })
  }

  acknowledgeAlert(id: EntityId, input: AlertAcknowledgeRequest = { confirmed: true }): Promise<AlertAcknowledgeResponse> {
    const body = { confirmed: input.confirmed ?? true }
    return this.postJson<unknown>(`alerts/${encodeURIComponent(id)}/acknowledge`, body).then((payload) => {
      const source = isRecord(payload) ? payload : {}
      const alert = normalizeAlert(source)
      const statistics = normalizeAlertStatistics(source.statistics)
      if (!alert || !statistics) throw new ApiError('API 返回的告警数据无效', { status: 200, code: 'INVALID_RESPONSE' })
      return { ...alert, statistics }
    })
  }

  listAlerts(params?: ListAlertsParams): Promise<CompatibleArrayResponse<ModelAlert>> {
    return this.get<unknown>('alerts', {
      query: params as RequestOptions['query'],
    }).then((payload) => {
      // The server uses the canonical page envelope when pagination is
      // requested and retains an array for old no-parameter callers.
      const rows = Array.isArray(payload)
        ? payload
        : isRecord(payload) && Array.isArray(payload.items) ? payload.items : []
      const normalized = rows.map(normalizeAlert).filter((item): item is ModelAlert => item !== null)
      return compatibleArrayResponse(normalized, payload)
    })
  }

  listAuditEvents(params?: ListAuditEventsParams): Promise<AuditEventsResponse> {
    return this.get<unknown>('audit-events', {
      query: params as RequestOptions['query'],
    }).then((payload) => {
      const source = isRecord(payload) ? payload : {}
      const items = Array.isArray(payload)
        ? payload
        : Array.isArray(source.items) ? source.items : []
      const page = typeof source.page === 'number' ? source.page : params?.page ?? 1
      const pageSize = typeof source.page_size === 'number' ? source.page_size : params?.page_size ?? items.length
      const total = typeof source.total === 'number' ? source.total : items.length
      const hasNext = typeof source.has_next === 'boolean' ? source.has_next : page * pageSize < total
      return {
        items: items.map(normalizeAuditEvent).filter((item): item is AuditEvent => item !== null),
        page,
        page_size: pageSize,
        total,
        has_next: hasNext,
      }
    })
  }

  getMcpCapabilities(): Promise<McpCapabilities> {
    return this.get<unknown>('../openapi.json').then(inspectMcpOpenApi)
  }

  predict(input: PredictionRequest): Promise<PredictionResponse> {
    return this.postJson<PredictionResponse, PredictionRequest>('mcp/predict', input)
  }
}

export const apiClient = new ApiClient()

export type ApiResult<T> = ApiResponse<T>
