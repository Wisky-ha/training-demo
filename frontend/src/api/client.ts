import type {
  ApiErrorCode,
  ApiErrorResponse,
  ApiResponse,
  DatasetUploadOptions,
  DatasetUploadResult,
  EntityId,
  HealthResponse,
  JsonValue,
  ListAlertsParams,
  ListModelsParams,
  ListScriptsParams,
  MarkModelAbnormalRequest,
  MarkModelAbnormalResponse,
  ModelAlert,
  ModelEvaluation,
  ModelTypeContract,
  ModelVersionDetail,
  ModelVersionSummary,
  PaginatedResponse,
  PredictionRequest,
  PredictionResponse,
  PublishModelRequest,
  PublishModelResponse,
  RollbackModelRequest,
  RollbackResponse,
  ScriptContract,
  ScriptUploadInput,
  TrainingJob,
  TrainingLogsResponse,
  CreateTrainingJobRequest,
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

  if (isRecord(payload) && typeof payload.detail === 'string') {
    return new ApiError(payload.detail, { status, code: 'VALIDATION_ERROR' })
  }

  return new ApiError(status ? `API request failed with status ${status}` : 'Network request failed', {
    status,
    code: status ? 'UNKNOWN_ERROR' : 'NETWORK_ERROR',
    details: asJsonValue(payload),
  })
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

  publishModel(
    id: EntityId,
    input: PublishModelRequest = {},
  ): Promise<PublishModelResponse> {
    return this.postJson<PublishModelResponse, PublishModelRequest>(
      `models/${encodeURIComponent(id)}/publish`,
      input,
    )
  }

  listModels(params?: ListModelsParams): Promise<PaginatedResponse<ModelVersionSummary>> {
    return this.get<PaginatedResponse<ModelVersionSummary>>('models', {
      query: params as RequestOptions['query'],
    })
  }

  getModel(id: EntityId): Promise<ModelVersionDetail> {
    return this.get<ModelVersionDetail>(`models/${encodeURIComponent(id)}`)
  }

  rollbackModel(
    id: EntityId,
    input: RollbackModelRequest = {},
  ): Promise<RollbackResponse> {
    return this.postJson<RollbackResponse, RollbackModelRequest>(
      `models/${encodeURIComponent(id)}/rollback`,
      input,
    )
  }

  markModelAbnormal(
    input: MarkModelAbnormalRequest,
  ): Promise<MarkModelAbnormalResponse> {
    return this.postJson<MarkModelAbnormalResponse, MarkModelAbnormalRequest>('models/abnormal', input)
  }

  listAlerts(params?: ListAlertsParams): Promise<PaginatedResponse<ModelAlert>> {
    return this.get<PaginatedResponse<ModelAlert>>('alerts', {
      query: params as RequestOptions['query'],
    })
  }

  predict(input: PredictionRequest): Promise<PredictionResponse> {
    return this.postJson<PredictionResponse, PredictionRequest>('predict', input)
  }
}

export const apiClient = new ApiClient()

export type ApiResult<T> = ApiResponse<T>
