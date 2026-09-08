/** Shared JSON/API contracts for the model-training platform.
 *
 * Response types describe the canonical wire shape. Compatibility types are
 * deliberately named and kept at the request/normalizer boundary; pages
 * should not need to understand historical field aliases.
 */

/** Resource IDs are opaque, non-empty server identifiers. */
export type EntityId = string
export type ResourceId = EntityId
export type IsoDateTime = string

export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue }
export type JsonRecord = { [key: string]: JsonValue }
/** Compatibility alias used by existing page code. */
export type JsonObject = JsonRecord

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

/** Public model-version lifecycle. */
export const MODEL_LIFECYCLE_STATUSES = ['READY', 'PUBLISHED', 'RETIRED', 'FAILED'] as const
export type ModelLifecycleStatus = (typeof MODEL_LIFECYCLE_STATUSES)[number]
/** Historical values accepted only when reading/normalizing old rows. */
export type LegacyModelLifecycleStatus = 'DRAFT' | 'TRAINING' | 'ABNORMAL'
export type ModelVersionStatus = ModelLifecycleStatus | LegacyModelLifecycleStatus

export type HealthStatus = 'HEALTHY' | 'ABNORMAL' | 'UNKNOWN'
export type LegacyHealthStatus = 'healthy' | 'abnormal'
export type HealthStatusWire = HealthStatus | LegacyHealthStatus

export type ScriptType = 'preprocessor' | 'trainer'
export type ScriptStatus = 'ENABLED' | 'DISABLED'
export type LegacyScriptStatus = 'enabled' | 'disabled'
export type ScriptStatusWire = ScriptStatus | LegacyScriptStatus

export interface ScriptContract {
  id: EntityId
  name: string
  script_type: ScriptType
  version: string
  source_code: string
  supported_model_types: ModelTypeCode[]
  status: ScriptStatus
  created_at: IsoDateTime
  uploaded_at: IsoDateTime
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

export interface TimeRange {
  start: IsoDateTime
  end: IsoDateTime
}

export interface TimeColumnParseResult {
  success: boolean
  format: string | null
  invalid_count: number
  min: IsoDateTime | null
  max: IsoDateTime | null
  is_sorted?: boolean
  out_of_order_count?: number
  duplicate_count?: number
  message: string | null
}

export interface DatasetPreviewRow {
  [column: string]: JsonPrimitive
}

export interface MissingValueSummary {
  missing_count: number
  missing_ratio: number
}

export interface FileStorageMetadata {
  artifact_type: string
  relative_path: string
  size_bytes: number
  checksum_sha256: string
}

export interface ValidationIssue {
  code: string
  field: string
  message: string
  column?: string
  row_numbers?: number[]
  count?: number
}

export interface DatasetValidationChecks {
  [check: string]: JsonValue
}

export interface DatasetValidationResult {
  valid: boolean
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
  checks: DatasetValidationChecks
}

/** Raw upload response returned by POST /api/datasets/upload. */
export interface DatasetUploadWireResponse {
  id: EntityId
  dataset_id?: EntityId
  file_name: string
  file_path: string | null
  file_size_bytes: number
  checksum_sha256: string
  file_storage: FileStorageMetadata | null
  row_count: number
  column_count: number
  columns: DatasetColumn[]
  column_names: string[]
  field_roles: Record<string, DatasetColumnRole>
  time_column: string
  feature_columns: string[]
  target_column: string
  column_types: Record<string, DatasetColumnDataType>
  missing_value_counts: Record<string, number>
  missing_values: Record<string, MissingValueSummary>
  preview_rows: DatasetPreviewRow[]
  preview: DatasetPreviewRow[]
  numeric_columns: string[]
  time_parse: TimeColumnParseResult
  time_range: TimeRange
  validation: DatasetValidationResult
  summary: JsonRecord
  data_summary: JsonRecord
  status: DatasetStatus
  created_at: IsoDateTime
}

/** Page-facing upload model; aliases are retained until normalizers land. */
export interface DatasetUploadResult extends DatasetUploadWireResponse {}
export interface DatasetUploadOptions {
  model_type?: ModelTypeCode
}

export type PreprocessingTaskStatus = 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'SKIPPED' | 'FAILED'
export type PreprocessingStage = 'waiting' | 'data_reading' | 'preprocessing' | 'validating' | 'completed' | 'failed'
export type WorkflowStageStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped'
export type PreprocessStage = PreprocessingStage
export type PreprocessStatus = PreprocessingTaskStatus

export interface PreprocessResultSummary {
  input_row_count: number | null
  output_row_count: number | null
  output_columns: string[]
  preprocess_used: boolean
  message: string | null
}

export interface QualityCheck {
  code: string
  valid: boolean
  message: string | null
  details: JsonRecord
}

/** Optional because old preprocessing responses did not persist quality checks. */
export interface QualityChecks {
  valid: boolean
  checks: QualityCheck[]
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
}

export interface StageProgress<TStage extends string = string> {
  stage: TStage
  status: WorkflowStageStatus
  message: string | null
  started_at: IsoDateTime | null
  finished_at: IsoDateTime | null
}

export interface PreprocessTask {
  id: EntityId
  model_type: ModelTypeCode
  dataset_id: EntityId
  preprocess_script_id: EntityId | null
  preprocess_used: boolean
  preprocess_status: 'used' | 'unused'
  preprocess_message: string
  status: PreprocessingTaskStatus
  stage: PreprocessingStage
  progress_stage: PreprocessingStage
  logs: Array<TrainingLogEntry | string>
  error_message: string | null
  error_code?: string | null
  error_details?: JsonRecord
  config?: JsonRecord
  input_row_count: number | null
  output_row_count: number | null
  input_columns: string[]
  output_columns: string[]
  input_summary: JsonRecord
  output_summary: JsonRecord
  preprocessor_path: string | null
  preprocessor_state: JsonRecord | null
  quality_checks?: QualityChecks | null
  started_at: IsoDateTime | null
  finished_at: IsoDateTime | null
  stage_started_at?: IsoDateTime | null
  created_at: IsoDateTime
  next_step: 'dataset_split' | null
  data_source: 'raw' | 'preprocessed'
  /** Compatibility aliases from the first frontend contract. */
  script_id?: EntityId | null
  current_stage?: PreprocessingStage
  stages?: StageProgress<PreprocessingStage>[]
  summary?: PreprocessResultSummary | null
}

export interface PreprocessingTransformRequest {
  dataset_id: EntityId
  config?: JsonRecord | null
}

export interface PreprocessingTransformResponse {
  task_id: EntityId
  preprocess_used: boolean
  data_source: 'raw' | 'preprocessed'
  row_count: number
  columns: string[]
  summary: JsonRecord
}

export interface DatasetSplitSummary {
  strategy?: 'time_ordered'
  split_strategy?: 'time_ordered'
  train_ratio?: 0.8
  split_ratio?: 0.8
  test_ratio: 0.2
  total_row_count?: number
  train_row_count: number
  test_row_count: number
  train_time_range: TimeRange | null
  test_time_range: TimeRange | null
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
  train_time_range: TimeRange
  test_time_range: TimeRange
  train_time_start: IsoDateTime
  train_time_end: IsoDateTime
  test_time_start: IsoDateTime
  test_time_end: IsoDateTime
  sort_order?: 'ascending'
  rounding_rule?: 'floor(total_row_count * 0.8)'
  sorted_before_split?: boolean
  created_at: IsoDateTime
}

export interface TrainingLogEntry {
  timestamp: IsoDateTime
  level: 'debug' | 'info' | 'warning' | 'error'
  message: string
  stage: string | null
}
export type TrainingLog = TrainingLogEntry | string

export type TrainingJobStatus = 'PENDING' | 'RUNNING' | 'PREPROCESSING' | 'SPLITTING' | 'TRAINING' | 'EVALUATING' | 'SUCCEEDED' | 'FAILED' | 'CANCELLED'
export type TrainingJobStage = string

export interface TrainingProgress {
  stage: string | null
  status: WorkflowStageStatus
  message: string | null
  started_at: IsoDateTime | null
  finished_at: IsoDateTime | null
}

export interface CreateTrainingJobRequest {
  model_type: ModelTypeCode
  dataset_id: EntityId
  preprocess_script_id: EntityId | null
  preprocessing_task_id?: EntityId | null
  train_script_id: EntityId
  config?: JsonRecord
}

export interface TrainingJob {
  id: EntityId
  model_type: ModelTypeCode
  dataset_id: EntityId
  preprocess_script_id: EntityId | null
  preprocessing_task_id: EntityId | null
  train_script_id: EntityId
  split_strategy: 'time_ordered'
  split_ratio: 0.8
  test_ratio: 0.2
  status: TrainingJobStatus
  progress_stage: string | null
  current_stage: string | null
  stage?: string | null
  stage_started_at?: IsoDateTime | null
  progress?: TrainingProgress | null
  logs: TrainingLog[]
  error_message: string | null
  error_code?: string | null
  error_details?: JsonRecord
  config?: JsonRecord
  config_summary?: JsonRecord
  dataset_split?: DatasetSplitSummary | null
  train_row_count: number | null
  test_row_count: number | null
  train_time_start: IsoDateTime | null
  train_time_end: IsoDateTime | null
  test_time_start: IsoDateTime | null
  test_time_end: IsoDateTime | null
  model_version_id: EntityId | null
  created_at: IsoDateTime
  started_at?: IsoDateTime | null
  finished_at: IsoDateTime | null
  stages?: StageProgress<TrainingJobStage>[]
}

export interface TrainingLogsParams {
  since?: IsoDateTime
  limit?: number
  cursor?: string
}
export interface TrainingLogsResponse {
  job_id: EntityId
  items: TrainingLogEntry[]
  next_cursor: string | null
}

export type MetricName = 'mae' | 'rmse' | 'mape' | 'r2'

/** Complete metric object returned by the evaluation endpoint. */
export interface MetricSet {
  mae: number | null
  rmse: number | null
  mape: number | null
  r2: number | null
  sample_count: number
  mape_valid_count: number
  mape_excluded_count: number
  mape_note: string
}
/** Partial metric maps are used only for old saved model rows. */
export type EvaluationMetrics = Partial<MetricSet>

export interface EvaluationChartPoint {
  time: IsoDateTime
  timestamp?: IsoDateTime
  actual: number
  predicted: number
  candidate_prediction?: number
  baseline_prediction?: number | null
  error: number
  signed_error?: number
  absolute_error?: number
  percentage_error?: number | null
}

export interface EvaluationErrorPoint {
  time: IsoDateTime
  timestamp?: IsoDateTime
  candidate_error?: number
  baseline_error?: number
  error: number
  absolute_error?: number
  percentage_error?: number | null
}

/** Older clients represented chart data as an envelope; the list is canonical. */
export interface EvaluationChartDataEnvelope {
  actual_vs_prediction: EvaluationChartPoint[]
  error_series: EvaluationErrorPoint[]
  metric_comparison?: MetricComparison[]
  sampled?: boolean
  source_row_count?: number
}
export type EvaluationChartData = EvaluationChartPoint[] | EvaluationChartDataEnvelope

export interface MetricComparison {
  metric: MetricName
  baseline: number | null
  candidate: number | null
  delta: number | null
}

export interface ModelComparisonEntry {
  model_version_id: EntityId | null
  version: string | null
  metrics: MetricSet
}

export interface ModelComparison {
  candidate: ModelComparisonEntry
  baseline: ModelComparisonEntry
  changes: Record<MetricName, number | null>
  new_model?: ModelComparisonEntry
  production?: ModelComparisonEntry
  current_model?: ModelComparisonEntry
}

export interface ModelMetricsEnvelope {
  candidate: MetricSet
  baseline: MetricSet
  difference: Record<MetricName, number | null>
  differences?: Record<MetricName, number | null>
}

export interface ModelEvaluation {
  job_id: EntityId
  model_version_id: EntityId
  metrics: MetricSet
  chart_data: EvaluationChartData
  error_data: EvaluationErrorPoint[]
  model_comparison: ModelComparison
  chart_sampled: boolean
  chart_total_count: number
  chart_sample_count: number
  test_time_series: IsoDateTime[]
  timestamps: IsoDateTime[]
  actual_values: number[]
  candidate_predictions: number[]
  baseline_predictions: number[]
  candidate_errors: number[]
  baseline_errors: number[]
  error_series: number[]
  model_metrics?: ModelMetricsEnvelope
  metric_differences?: Record<MetricName, number | null>
  metrics_difference?: Record<MetricName, number | null>
  /** Historical aliases accepted by evaluation normalizers. */
  candidate?: EvaluationMetrics
  baseline?: EvaluationMetrics
  comparison?: ModelComparison | JsonRecord
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
  fields?: InputFieldSchema[]
  [key: string]: JsonValue | InputFieldSchema[] | undefined
}
export interface DataSummary {
  row_count: number
  time_range: TimeRange | null
  columns: string[]
}

export interface ScriptSnapshot {
  name: string
  version: string
  source_code: string
  script_type: ScriptType
}

export interface ModelVersionSummary {
  id: EntityId
  model_type: ModelTypeCode
  version: string
  status: ModelVersionStatus | null
  health_status?: HealthStatus | null
  is_baseline: boolean
  is_current: boolean
  is_abnormal?: boolean
  is_rollback_available?: boolean
  metrics: EvaluationMetrics | JsonRecord | null
  preprocess_used?: boolean
  model_path?: string | null
  model_artifact_id?: EntityId | null
  preprocessor_path?: string | null
  preprocessor_artifact_id?: EntityId | null
  training_job_id?: EntityId | null
  train_script_id?: EntityId | null
  train_script_version?: string | null
  train_script_source?: string | null
  preprocess_script_id?: EntityId | null
  preprocess_script_version?: string | null
  preprocess_script_source?: string | null
  previous_healthy_version_id?: EntityId | null
  train_script?: Pick<ScriptContract, 'id' | 'name' | 'version'> | null
  preprocess_script?: Pick<ScriptContract, 'id' | 'name' | 'version'> | null
  input_schema?: InputSchema | JsonRecord
  feature_columns?: string[]
  time_column?: string | null
  target_column?: string | null
  split_strategy?: 'time_ordered'
  split_ratio?: 0.8
  test_ratio?: 0.2
  train_data_summary?: JsonRecord
  test_data_summary?: JsonRecord
  created_at: IsoDateTime
  published_at: IsoDateTime | null
}

export interface ModelVersionDetail extends ModelVersionSummary {
  train_script_snapshot?: ScriptSnapshot | null
  preprocess_script_snapshot?: ScriptSnapshot | null
  split?: DatasetSplitSummary | null
  evaluation?: ModelEvaluation | null
}

/** Full model response returned by the backend; normalizers may project it. */
export interface ModelVersionResponse extends ModelVersionSummary {
  status: ModelVersionStatus
  health_status: HealthStatus
  model_path: string
  input_schema: JsonRecord
  feature_columns: string[]
  train_data_summary: JsonRecord
  test_data_summary: JsonRecord
  split_strategy: 'time_ordered'
  split_ratio: 0.8
  test_ratio: 0.2
  metrics: JsonRecord
}

export interface ModelSaveRequest {
  id?: EntityId | null
  model_type: ModelTypeCode
  version?: string | null
  model_path?: string | null
  model_content_base64?: string | null
  preprocessor_path?: string | null
  training_job_id?: EntityId | null
  train_script_id?: EntityId | null
  train_script_version?: string | null
  train_script_source?: string | null
  preprocess_script_id?: EntityId | null
  preprocess_script_version?: string | null
  preprocess_script_source?: string | null
  preprocess_used?: boolean
  preprocessor_state?: JsonRecord | null
  input_schema?: JsonRecord
  time_column?: string | null
  feature_columns?: string[]
  target_column?: string | null
  split_strategy?: 'time_ordered'
  split_ratio?: 0.8
  test_ratio?: 0.2
  train_data_summary?: JsonRecord
  test_data_summary?: JsonRecord
  metrics?: JsonRecord
  health_status?: HealthStatus | null
  status?: 'DRAFT' | 'READY'
}

export interface PublishRecord {
  id: EntityId
  model_version_id: EntityId
  published_version: string
  previous_current_version_id: EntityId | null
  published_at: IsoDateTime
  reason: string | null
  idempotency_key: string | null
  /** Compatibility response alias. */
  message?: string | null
}

/** Canonical publication command. `confirmed` is intentionally explicit. */
export interface PublishModelRequest {
  confirmed: boolean
  reason?: string | null
  idempotency_key?: string | null
}
/** Old names are accepted only by the API-client compatibility boundary. */
export interface PublishModelCompatibilityRequest {
  confirmed?: boolean
  confirm?: boolean
  confirmation?: boolean
  reason?: string | null
  message?: string | null
  idempotency_key?: string | null
}
export type PublishModelInput = PublishModelRequest | PublishModelCompatibilityRequest

export interface PublishModelResponse {
  operation: 'publish' | string
  model: ModelVersionDetail | ModelVersionSummary
  record?: PublishRecord | null
}

export type RollbackStatus = 'PENDING' | 'SUCCEEDED' | 'FAILED'
export interface RollbackRecord {
  id: EntityId
  model_type: ModelTypeCode
  rollback_from: EntityId | null
  rollback_to: EntityId | null
  alert_id: EntityId | null
  reason: string | null
  idempotency_key: string | null
  status: RollbackStatus
  created_at: IsoDateTime
  finished_at: IsoDateTime | null
  /** Compatibility aliases for older responses. */
  from_version_id?: EntityId | null
  to_version_id?: EntityId | null
}

/** Canonical rollback requires an explicit target, reason, and idempotency key. */
export interface RollbackModelRequest {
  target_version_id: EntityId
  reason: string
  idempotency_key: string
}
/** Compatibility aliases are not part of the canonical rollback request. */
export interface RollbackModelCompatibilityRequest {
  target_version_id?: EntityId
  version_id?: EntityId
  version?: string
  reason?: string
  idempotency_key?: string
}
export type RollbackModelInput = RollbackModelRequest | RollbackModelCompatibilityRequest

export interface RollbackResponse {
  operation?: 'rollback' | string
  model?: ModelVersionDetail | ModelVersionSummary
  rollback?: RollbackRecord | null
  current_model?: ModelVersionSummary
  record?: RollbackRecord
}

export const MODEL_ALERT_STATUSES = ['ACTIVE', 'ACKNOWLEDGED', 'RESOLVED'] as const
export type ModelAlertState = (typeof MODEL_ALERT_STATUSES)[number]
export type AlertStatus = ModelAlertState

export interface ModelAlert {
  id: EntityId
  model_type: ModelTypeCode
  model_version_id: EntityId | null
  reason: string
  rollback_from: EntityId | null
  rollback_to: EntityId | null
  status: ModelAlertState
  created_at: IsoDateTime
  acknowledged_at: IsoDateTime | null
  resolved_at: IsoDateTime | null
}

export interface AlertStatistics {
  total: number
  active: number
  acknowledged: number
  resolved: number
}

/** Backend alert response, including cursor compatibility fields. */
export interface AlertListResponse {
  items: ModelAlert[]
  total: number
  page: number | null
  page_size: number | null
  limit: number | null
  next_cursor: string | null
  statistics: AlertStatistics
  has_next?: boolean
}
export interface PaginatedAlertsResponse extends CanonicalPaginatedResponse<ModelAlert> {
  statistics: AlertStatistics
  next_cursor: string | null
}
export interface AlertAcknowledgeRequest {
  confirmed?: boolean
}
export interface AlertAcknowledgeResponse extends ModelAlert {
  statistics: AlertStatistics
}

export interface MarkModelAbnormalRequest {
  model_type: ModelTypeCode
  model_version: string
  abnormal: true
  reason: string
}

export interface PredictionRequest {
  model_type: ModelTypeCode
  model_version?: string
  data: JsonRecord[]
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
  service?: string
  database?: 'ok'
  environment?: string
  app_name?: string
}

export interface PageParams {
  page?: number
  page_size?: number
}

/** Canonical page shape shared by scripts, alerts, and audit events. */
export interface PaginationMeta {
  page: number
  page_size: number
  total: number
  has_next: boolean
}
export interface CanonicalPaginatedResponse<T> extends PaginationMeta {
  items: T[]
}
/** Legacy script-list response; only the client compatibility adapter uses it. */
export interface LegacyPaginationMeta {
  page: number
  page_size: number
  total: number
  total_pages: number
}
export interface LegacyPaginatedResponse<T> {
  items: T[]
  pagination: LegacyPaginationMeta
}
/** Explicit wire compatibility while new endpoints use the canonical shape. */
export type PaginatedResponse<T> = CanonicalPaginatedResponse<T> | LegacyPaginatedResponse<T>

export interface ListScriptsParams extends PageParams {
  script_type?: ScriptType
  model_type?: ModelTypeCode
  status?: ScriptStatusWire
}
export interface ListModelsParams extends PageParams {
  model_type?: ModelTypeCode
  status?: ModelVersionStatus
  health_status?: HealthStatus
}
export interface ListAlertsParams extends PageParams {
  model_type?: ModelTypeCode
  status?: ModelAlertState
  active_only?: boolean
  limit?: number
  cursor?: string
}

export interface AuditEvent {
  id: EntityId
  occurred_at: IsoDateTime
  event_type: string
  object_type: string
  object_id: EntityId | null
  model_type: ModelTypeCode | null
  model_version_id: EntityId | null
  training_job_id: EntityId | null
  operator_type: string | null
  operator_id: EntityId | null
  operator_name: string | null
  result: string
  message: string | null
  request_id: string | null
  correlation_id: string | null
  metadata: JsonRecord
}
export interface AuditEventFilters {
  from?: IsoDateTime
  to?: IsoDateTime
  model_type?: ModelTypeCode
  event_type?: string
  object_type?: string
  result?: string
  query?: string
}
export interface ListAuditEventsParams extends PageParams, AuditEventFilters {}
export interface AuditEventsResponse extends CanonicalPaginatedResponse<AuditEvent> {}

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
  record?: PublishRecord | null
  rollback?: RollbackRecord | null
  alert?: ModelAlert | null
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
