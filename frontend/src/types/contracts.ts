/**
 * Shared API contracts for the model-training platform.
 *
 * These types describe the JSON/multipart wire format used by the FastAPI
 * service. Field names intentionally follow the backend's snake_case naming
 * convention; the UI store can map them to its camelCase view model when a
 * page is implemented.
 */

export type EntityId = string
export type IsoDateTime = string
export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue }
export type JsonObject = { [key: string]: JsonValue }

export const MODEL_TYPE_CODES = [
  'electric_load',
  'heating_cooling_load',
  'integrated_energy',
] as const

export type ModelTypeCode = (typeof MODEL_TYPE_CODES)[number]

export const MODEL_TYPE_NAMES: Record<ModelTypeCode, string> = {
  electric_load: '电力负荷预测',
  heating_cooling_load: '冷热负荷预测',
  integrated_energy: '综合能耗预测',
}

/** Model-type alert status used by the current backend (uppercase values are canonical). */
export type AlertStatus = 'ACTIVE' | 'RESOLVED' | 'healthy' | 'active' | 'resolved'
export type HealthStatus = 'HEALTHY' | 'ABNORMAL' | 'healthy' | 'abnormal'
export type ModelTypeStatus =
  | 'untrained'
  | 'training'
  | 'ready'
  | 'published'
  | 'retired'
  | 'abnormal'
  | 'failed'

export interface ModelTypeContract {
  id: EntityId
  code: ModelTypeCode
  name: string
  description?: string
  current_version_id: EntityId | null
  current_version?: ModelVersionSummary | null
  status: ModelTypeStatus
  alert_status: AlertStatus
  backup_version_id?: EntityId | null
  last_trained_at: IsoDateTime | null
  created_at: IsoDateTime
  updated_at: IsoDateTime
}

export type ScriptType = 'preprocessor' | 'trainer'
export type ScriptStatus = 'enabled' | 'disabled' | 'ENABLED' | 'DISABLED'

export interface ScriptContract {
  id: EntityId
  name: string
  script_type: ScriptType
  version: string
  source_code?: string
  supported_model_types: ModelTypeCode[]
  status: ScriptStatus
  created_at: IsoDateTime
  updated_at?: IsoDateTime
}

export interface ScriptUploadInput {
  file: File
  name: string
  script_type: ScriptType
  supported_model_types: ModelTypeCode[]
  version?: string
}

export type DatasetStatus = 'uploaded' | 'parsed' | 'failed'
export type DatasetColumnRole = 'time' | 'feature' | 'target'
export type DatasetColumnDataType = 'datetime' | 'number' | 'string' | 'boolean' | 'unknown'

export interface DatasetColumn {
  name: string
  role: DatasetColumnRole
  data_type: DatasetColumnDataType
  nullable: boolean
  missing_count: number
  missing_ratio: number
}

export interface TimeColumnParseResult {
  success: boolean
  format?: string | null
  invalid_count: number
  min?: IsoDateTime | null
  max?: IsoDateTime | null
  message?: string | null
}

export interface DatasetPreviewRow {
  [column: string]: JsonPrimitive
}

export interface MissingValueSummary {
  missing_count: number
  missing_ratio: number
}

export interface DatasetUploadOptions {
  model_type?: ModelTypeCode
}

export interface DatasetUploadResult {
  id: EntityId
  file_name: string
  row_count: number
  column_count: number
  columns: DatasetColumn[]
  time_column: string
  feature_columns: string[]
  target_column: string
  time_parse: TimeColumnParseResult
  numeric_columns: string[]
  missing_values: Record<string, MissingValueSummary>
  preview_rows: DatasetPreviewRow[]
  status: DatasetStatus
  created_at: IsoDateTime
}

export interface PreprocessResultSummary {
  input_row_count: number
  output_row_count: number
  output_columns: string[]
  preprocess_used: boolean
  message?: string | null
}

export type WorkflowStageStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped'
export type PreprocessStage = 'waiting' | 'data_reading' | 'preprocessing' | 'validating' | 'completed'
export type TrainingStage =
  | 'preparing_data'
  | 'loading_script'
  | 'training'
  | 'saving_model'
  | 'entering_evaluation'

export interface StageProgress<TStage extends string = string> {
  stage: TStage
  status: WorkflowStageStatus
  message?: string | null
  started_at?: IsoDateTime | null
  finished_at?: IsoDateTime | null
}

export interface PreprocessTask {
  id: EntityId
  dataset_id: EntityId
  model_type?: ModelTypeCode
  script_id: EntityId | null
  preprocess_script_id?: EntityId | null
  preprocess_used?: boolean
  preprocess_status?: 'used' | 'unused'
  status: WorkflowStageStatus | 'waiting' | 'running' | 'succeeded' | 'skipped' | 'failed'
  current_stage?: PreprocessStage
  stage?: PreprocessStage
  progress_stage?: PreprocessStage
  stages?: StageProgress<PreprocessStage>[]
  summary?: PreprocessResultSummary | null
  input_row_count?: number | null
  output_row_count?: number | null
  input_columns?: string[]
  output_columns?: string[]
  input_summary?: Record<string, JsonValue>
  output_summary?: Record<string, JsonValue>
  logs: Array<TrainingLogEntry | string>
  error_message?: string | null
  created_at: IsoDateTime
  started_at?: IsoDateTime | null
  finished_at?: IsoDateTime | null
  next_step?: 'dataset_split' | null
  data_source?: 'raw' | 'preprocessed'
}

export type TrainingJobStatus =
  | 'PENDING'
  | 'RUNNING'
  | 'PREPROCESSING'
  | 'SPLITTING'
  | 'TRAINING'
  | 'EVALUATING'
  | 'SUCCEEDED'
  | 'FAILED'
  | 'CANCELLED'

export type TrainingJobStage = PreprocessStage | TrainingStage | 'splitting' | 'evaluating'

export interface DatasetSplitSummary {
  strategy?: 'time_ordered'
  split_strategy?: 'time_ordered'
  train_ratio?: 0.8
  split_ratio?: 0.8
  test_ratio: 0.2
  total_row_count?: number
  train_row_count: number
  test_row_count: number
  train_time_range: [IsoDateTime, IsoDateTime] | null | { start: IsoDateTime; end: IsoDateTime }
  test_time_range: [IsoDateTime, IsoDateTime] | null | { start: IsoDateTime; end: IsoDateTime }
}

export interface DatasetSplitResult {
  id: EntityId
  dataset_id: EntityId
  preprocessing_task_id: EntityId | null
  data_source: 'raw' | 'preprocessed'
  split_strategy: 'time_ordered'
  split_ratio: 0.8
  test_ratio: 0.2
  total_row_count: number
  train_row_count: number
  test_row_count: number
  train_time_range: { start: IsoDateTime; end: IsoDateTime }
  test_time_range: { start: IsoDateTime; end: IsoDateTime }
  train_time_start: IsoDateTime
  train_time_end: IsoDateTime
  test_time_start: IsoDateTime
  test_time_end: IsoDateTime
  created_at: IsoDateTime
}

export interface TrainingLogEntry {
  timestamp: IsoDateTime
  level: 'debug' | 'info' | 'warning' | 'error'
  message: string
  stage?: TrainingJobStage | PreprocessStage | null
}

export interface CreateTrainingJobRequest {
  model_type: ModelTypeCode
  dataset_id: EntityId
  preprocess_script_id: EntityId | null
  preprocessing_task_id?: EntityId | null
  train_script_id: EntityId
  config?: Record<string, JsonValue>
}

export interface TrainingJob {
  id: EntityId
  model_type: ModelTypeCode
  dataset_id: EntityId
  preprocess_script_id: EntityId | null
  preprocessing_task_id?: EntityId | null
  train_script_id: EntityId
  split_strategy?: 'time_ordered'
  split_ratio: 0.8
  test_ratio?: 0.2
  status: TrainingJobStatus
  progress_stage?: TrainingJobStage | string | null
  current_stage?: TrainingJobStage | string | null
  stage?: string | null
  progress?: StageProgress<TrainingJobStage> | null
  stages?: StageProgress<TrainingJobStage>[]
  logs: Array<TrainingLogEntry | string>
  error_message: string | null
  dataset_split?: DatasetSplitSummary | null
  train_row_count?: number | null
  test_row_count?: number | null
  train_time_start?: IsoDateTime | null
  train_time_end?: IsoDateTime | null
  test_time_start?: IsoDateTime | null
  test_time_end?: IsoDateTime | null
  model_version_id?: EntityId | null
  created_at: IsoDateTime
  started_at?: IsoDateTime | null
  finished_at: IsoDateTime | null
}

export interface TrainingLogsResponse {
  job_id: EntityId
  items: Array<TrainingLogEntry | string>
  next_cursor?: string | null
}

export type MetricName = 'mae' | 'rmse' | 'mape' | 'r2'

export interface EvaluationMetrics {
  mae: number | null
  rmse: number | null
  mape: number | null
  r2: number | null
  sample_count?: number
  mape_valid_count: number
  mape_excluded_count?: number
  mape_note?: string | null
}

export interface MetricComparison {
  metric: MetricName
  baseline: number | null
  candidate: number | null
  delta: number | null
}

export interface EvaluationSeriesPoint {
  /** The service currently returns `time`; `timestamp` is retained for compatibility. */
  timestamp?: IsoDateTime
  time?: IsoDateTime
  actual: number
  /** Newer clients may call this candidate_prediction; the service uses predicted. */
  predicted?: number
  candidate_prediction?: number
  baseline_prediction?: number | null
  error: number
  signed_error?: number
  absolute_error?: number
  percentage_error?: number | null
}

export interface EvaluationChartData {
  actual_vs_prediction?: EvaluationSeriesPoint[]
  error_series?: EvaluationSeriesPoint[]
  metric_comparison?: MetricComparison[]
  sampled?: boolean
  source_row_count?: number
}

export interface ModelEvaluation {
  id?: EntityId
  job_id?: EntityId
  model_version_id?: EntityId
  candidate?: EvaluationMetrics
  baseline?: EvaluationMetrics
  metrics?: EvaluationMetrics | Record<string, JsonValue>
  comparison?: MetricComparison[] | Record<string, JsonValue>
  chart_data?: EvaluationChartData | Array<Record<string, JsonValue>>
  error_data?: Array<Record<string, JsonValue>>
  chart_sampled?: boolean
  chart_total_count?: number
  chart_sample_count?: number
  model_comparison?: Record<string, JsonValue>
  created_at?: IsoDateTime
}

export interface InputFieldSchema {
  name: string
  role: 'time' | 'feature'
  data_type: DatasetColumnDataType
  required: true
}

export interface InputSchema {
  time_column: string
  feature_columns: string[]
  target_column: string
  fields: InputFieldSchema[]
}

export interface DataSummary {
  row_count: number
  time_range: [IsoDateTime, IsoDateTime] | null
  columns: string[]
}

export type ModelVersionStatus =
  | 'DRAFT'
  | 'TRAINING'
  | 'READY'
  | 'PUBLISHED'
  | 'RETIRED'
  | 'ABNORMAL'
  | 'FAILED'

export interface ModelVersionSummary {
  id: EntityId
  model_type: ModelTypeCode
  version: string
  status: ModelVersionStatus
  health_status?: HealthStatus | string
  is_baseline: boolean
  is_current: boolean
  is_abnormal?: boolean
  is_rollback_available?: boolean
  metrics: EvaluationMetrics | Record<string, JsonValue> | null
  preprocess_used?: boolean
  model_path?: string | null
  preprocessor_path?: string | null
  training_job_id?: EntityId | null
  train_script_id?: EntityId | null
  train_script_version?: string | null
  preprocess_script_id?: EntityId | null
  preprocess_script_version?: string | null
  previous_healthy_version_id?: EntityId | null
  train_script?: Pick<ScriptContract, 'id' | 'name' | 'version'> | null
  preprocess_script?: Pick<ScriptContract, 'id' | 'name' | 'version'> | null
  input_schema?: InputSchema | Record<string, JsonValue>
  feature_columns?: string[]
  time_column?: string | null
  target_column?: string | null
  created_at: IsoDateTime
  published_at: IsoDateTime | null
}

export interface ModelVersionDetail extends ModelVersionSummary {
  model_path?: string | null
  preprocessor_path?: string | null
  train_script_snapshot?: ScriptSnapshot | null
  preprocess_script_snapshot?: ScriptSnapshot | null
  input_schema?: InputSchema | Record<string, JsonValue>
  split?: DatasetSplitSummary | null
  train_data_summary?: DataSummary | null
  test_data_summary?: DataSummary | null
  previous_healthy_version_id: EntityId | null
  evaluation?: ModelEvaluation | null
}

export interface ScriptSnapshot {
  name: string
  version: string
  source_code: string
  script_type: ScriptType
}

export interface PublishRecord {
  id: EntityId
  model_version_id: EntityId
  published_version: string
  previous_current_version_id: EntityId | null
  published_at: IsoDateTime
  message?: string | null
}

export interface PublishModelRequest {
  message?: string
  confirm?: boolean
  confirmed?: boolean
}

export interface ModelSaveRequest {
  model_type: ModelTypeCode
  status?: 'DRAFT' | 'READY'
  training_job_id?: EntityId
  train_script_id?: EntityId
  preprocess_script_id?: EntityId | null
  preprocess_used?: boolean
  input_schema?: Record<string, JsonValue>
  time_column?: string
  feature_columns?: string[]
  target_column?: string
  metrics?: Record<string, JsonValue>
}

export interface PublishModelResponse {
  model: ModelVersionDetail | ModelVersionSummary
  record?: PublishRecord
  operation?: string
}

export type RollbackStatus = 'PENDING' | 'SUCCEEDED' | 'FAILED' | 'pending' | 'succeeded' | 'failed'

/** Wire-compatible with the backend's rollback_from/rollback_to response. */
export interface RollbackRecord {
  id: EntityId
  model_type: ModelTypeCode
  rollback_from: EntityId | null
  rollback_to: EntityId | null
  alert_id?: EntityId | null
  reason?: string | null
  status?: RollbackStatus
  created_at: IsoDateTime
  finished_at?: IsoDateTime | null
  /** Legacy aliases retained for older API responses. */
  from_version_id?: EntityId | null
  to_version_id?: EntityId | null
}

export interface RollbackModelRequest {
  target_version_id?: EntityId
  reason?: string
}

/** Compatibility shape for older rollback consumers and current lifecycle responses. */
export interface RollbackResponse {
  operation?: string
  model?: ModelVersionDetail | ModelVersionSummary
  rollback?: RollbackRecord | null
  current_model?: ModelVersionSummary
  record?: RollbackRecord
}

export type ModelAlertState = 'ACTIVE' | 'RESOLVED'

export interface ModelAlert {
  id: EntityId
  model_type: ModelTypeCode
  model_version_id: EntityId | null
  reason: string
  rollback_from: EntityId | null
  rollback_to: EntityId | null
  status: ModelAlertState
  created_at: IsoDateTime
  resolved_at: IsoDateTime | null
}

export interface MarkModelAbnormalRequest {
  model_type: ModelTypeCode
  model_version: string
  abnormal: true
  reason: string
}

export interface MarkModelAbnormalResponse {
  alert: ModelAlert | null
  current_model: ModelVersionSummary
  rollback: RollbackRecord | null
}

export interface PredictionRequest {
  model_type: ModelTypeCode
  model_version?: string
  data: JsonObject[]
}

export interface PredictionResponse {
  success: true
  model_type: ModelTypeCode
  model_version: string
  preprocess_used: boolean
  predictions: number[]
}

export interface HealthResponse {
  status: 'ok'
  /** Present in the older health contract; current backend returns environment/app_name. */
  service?: string
  database?: 'ok'
  environment?: string
  app_name?: string
}

export interface PageParams {
  page?: number
  page_size?: number
}

export interface PaginationMeta {
  page: number
  page_size: number
  total: number
  total_pages: number
}

export interface PaginatedResponse<T> {
  items: T[]
  pagination: PaginationMeta
}

export interface ListScriptsParams extends PageParams {
  script_type?: ScriptType
  model_type?: ModelTypeCode
  status?: ScriptStatus
}

export interface ListModelsParams extends PageParams {
  model_type?: ModelTypeCode
  status?: ModelVersionStatus
}

export interface ListAlertsParams extends PageParams {
  model_type?: ModelTypeCode
  status?: ModelAlertState
  active_only?: boolean
}

export type ApiErrorCode =
  | 'MODEL_TYPE_NOT_FOUND'
  | 'MODEL_VERSION_NOT_FOUND'
  | 'MODEL_VERSION_UNAVAILABLE'
  | 'MISSING_TIME_FIELD'
  | 'MISSING_FEATURE'
  | 'INVALID_FIELD_TYPE'
  | 'INVALID_TIME_FORMAT'
  | 'PREPROCESS_FAILED'
  | 'PREDICTION_FAILED'
  | 'NO_HEALTHY_BACKUP'
  | 'VALIDATION_ERROR'
  | 'NETWORK_ERROR'
  | 'UNKNOWN_ERROR'

export interface LifecycleOperationResponse {
  operation: string
  model: ModelVersionDetail | ModelVersionSummary
  rollback?: RollbackRecord | null
  alert?: ModelAlert | null
  [key: string]: JsonValue | ModelVersionDetail | ModelVersionSummary | RollbackRecord | ModelAlert | undefined
}

export interface ApiErrorResponse {
  success: false
  error_code: ApiErrorCode | (string & {})
  message: string
  details?: JsonValue
}

export interface ApiSuccessEnvelope<T> {
  success: true
  data: T
}

export type ApiResponse<T> = T | ApiSuccessEnvelope<T> | ApiErrorResponse
