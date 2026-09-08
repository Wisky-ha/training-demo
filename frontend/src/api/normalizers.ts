import type {
  DatasetColumnDataType,
  DatasetColumnRole,
  HealthStatus,
  MetricName,
  ModelAlertState,
  ModelTypeCode,
  PreprocessingStage,
  PreprocessingTaskStatus,
  TrainingJobStatus,
  TrainingLogEntry,
} from '../types/contracts'

/** Values used at the wire/view-model boundary when the server has no fact. */
export const UNKNOWN = 'UNKNOWN' as const
export const UNKNOWN_TEXT = '未知' as const
export const NOT_PROVIDED_TEXT = '未提供' as const

export type UnknownState = typeof UNKNOWN
export type ValidationStatus = 'VALID' | 'INVALID' | UnknownState
export type NormalizedHealthStatus = HealthStatus
export type NormalizedColumnRole = DatasetColumnRole | UnknownState
export type NormalizedColumnDataType = DatasetColumnDataType | UnknownState
export type NormalizedPreprocessingStatus = PreprocessingTaskStatus | UnknownState
export type NormalizedTrainingStatus = TrainingJobStatus | UnknownState
export type NormalizedAlertStatus = ModelAlertState | UnknownState

/** Deliberately unknown-valued wire record; no page code should consume it directly. */
export interface WireRecord {
  [key: string]: unknown
}

function asRecord(value: unknown): WireRecord | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as WireRecord
    : null
}

function firstValue(source: WireRecord, keys: readonly string[]): unknown {
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) return source[key]
  }
  return null
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null
}

function firstString(source: WireRecord, keys: readonly string[]): string | null {
  return stringValue(firstValue(source, keys))
}

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function firstNumber(source: WireRecord, keys: readonly string[]): number | null {
  return numberValue(firstValue(source, keys))
}

function booleanValue(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function firstBoolean(source: WireRecord, keys: readonly string[]): boolean | null {
  return booleanValue(firstValue(source, keys))
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

function firstStringList(source: WireRecord, keys: readonly string[]): string[] {
  for (const key of keys) {
    if (Array.isArray(source[key])) return stringList(source[key])
  }
  return []
}

function firstArray(source: WireRecord, keys: readonly string[]): unknown[] {
  for (const key of keys) {
    if (Array.isArray(source[key])) return source[key]
  }
  return []
}

function recordValue(value: unknown): WireRecord | null {
  return asRecord(value)
}

function firstRecord(source: WireRecord, keys: readonly string[]): WireRecord | null {
  for (const key of keys) {
    const value = recordValue(source[key])
    if (value) return value
  }
  return null
}

function upper(value: unknown): string | null {
  const text = stringValue(value)
  return text ? text.toUpperCase() : null
}

function lower(value: unknown): string | null {
  const text = stringValue(value)
  return text ? text.toLowerCase() : null
}

function textOr(value: unknown, fallback: string): string {
  return stringValue(value) ?? fallback
}

function idOrNull(value: unknown): string | null {
  return stringValue(value)
}

function normalizeStatus<T extends string>(value: unknown, allowed: readonly T[]): T | UnknownState {
  const candidate = upper(value)
  return candidate && allowed.includes(candidate as T) ? candidate as T : UNKNOWN
}

function normalizeStage(value: unknown, allowed: readonly string[]): string | UnknownState {
  const candidate = lower(value)
  return candidate && allowed.includes(candidate) ? candidate : UNKNOWN
}

function firstKnownStage(source: WireRecord, keys: readonly string[], allowed: readonly string[]): string | UnknownState {
  for (const key of keys) {
    const stage = normalizeStage(source[key], allowed)
    if (stage !== UNKNOWN) return stage
  }
  return UNKNOWN
}

export interface NormalizedValidationIssue {
  code: string
  field: string
  message: string
  column: string | null
  rowNumbers: number[]
  count: number | null
}

export interface NormalizedValidation {
  valid: boolean | null
  errors: NormalizedValidationIssue[]
  warnings: NormalizedValidationIssue[]
  checks: WireRecord
}

function normalizeIssue(value: unknown): NormalizedValidationIssue {
  const source = asRecord(value) ?? {}
  return {
    code: textOr(source.code, UNKNOWN_TEXT),
    field: textOr(source.field, UNKNOWN_TEXT),
    message: textOr(source.message, NOT_PROVIDED_TEXT),
    column: stringValue(source.column),
    rowNumbers: Array.isArray(source.row_numbers)
      ? source.row_numbers.filter((item): item is number => typeof item === 'number' && Number.isFinite(item))
      : [],
    count: numberValue(source.count),
  }
}

function normalizeIssues(value: unknown): NormalizedValidationIssue[] {
  return Array.isArray(value) ? value.map(normalizeIssue) : []
}

function hasInvalidCheck(checks: WireRecord): boolean {
  return Object.values(checks).some((check) => asRecord(check)?.valid === false)
}

function normalizeValidation(source: WireRecord): NormalizedValidation {
  const validation = firstRecord(source, ['validation', 'validation_result'])
  const value = validation ?? {}
  const errors = normalizeIssues(value.errors ?? source.errors)
  const warnings = normalizeIssues(value.warnings ?? source.warnings)
  const checks = recordValue(value.checks ?? source.checks) ?? {}
  const valid = booleanValue(value.valid)
  return { valid, errors, warnings, checks }
}

function validationStatus(validation: NormalizedValidation, sourceStatus: unknown): ValidationStatus {
  if (validation.valid !== null) return validation.valid ? 'VALID' : 'INVALID'
  if (validation.errors.length > 0 || hasInvalidCheck(validation.checks)) return 'INVALID'
  if (upper(sourceStatus) === 'FAILED') return 'INVALID'
  return UNKNOWN
}

export interface NormalizedDatasetColumn {
  name: string
  role: NormalizedColumnRole
  roleLabel: string
  dataType: NormalizedColumnDataType
  dataTypeLabel: string
  nullable: boolean | null
  missingCount: number | null
  missingRatio: number | null
}

export interface NormalizedTimeRange {
  start: string | null
  end: string | null
}

export interface NormalizedDatasetUpload {
  id: string | null
  fileName: string
  filePath: string | null
  fileSize: number | null
  checksumSha256: string | null
  rowCount: number | null
  columnCount: number | null
  columns: NormalizedDatasetColumn[]
  columnNames: string[]
  timeColumn: string
  featureColumns: string[]
  targetColumn: string
  columnTypes: WireRecord
  missingValueCounts: WireRecord
  missingValues: WireRecord
  previewRows: WireRecord[]
  timeRange: NormalizedTimeRange | null
  validation: NormalizedValidation
  validationStatus: ValidationStatus
  status: string | UnknownState
  createdAt: string | null
}

const COLUMN_ROLES: readonly string[] = ['time', 'feature', 'target']
const COLUMN_TYPES: readonly string[] = ['datetime', 'number', 'string', 'boolean', 'unknown']
const DATASET_STATUSES: readonly string[] = ['UPLOADED', 'PARSED', 'FAILED']

function normalizeColumnRole(value: unknown): NormalizedColumnRole {
  const candidate = lower(value)
  return candidate && COLUMN_ROLES.includes(candidate) ? candidate as DatasetColumnRole : UNKNOWN
}

function normalizeColumnType(value: unknown): NormalizedColumnDataType {
  const candidate = lower(value)
  return candidate && COLUMN_TYPES.includes(candidate) && candidate !== 'unknown'
    ? candidate as DatasetColumnDataType
    : UNKNOWN
}

function columnNames(source: WireRecord): string[] {
  for (const key of ['column_names', 'columnNames']) {
    if (Array.isArray(source[key]) && source[key].length > 0) {
      return source[key].map((name) => textOr(name, UNKNOWN_TEXT))
    }
  }
  const columns = Array.isArray(source.columns) ? source.columns : []
  if (columns.length > 0) {
    return columns.map((column) => textOr(asRecord(column)?.name, UNKNOWN_TEXT))
  }
  const names = new Set<string>()
  for (const field of ['field_roles', 'fieldRoles', 'column_types', 'columnTypes', 'missing_value_counts', 'missingValueCounts']) {
    const record = asRecord(source[field])
    if (record) Object.keys(record).forEach((name) => names.add(name))
  }
  return [...names]
}

function normalizeColumns(source: WireRecord, names: string[], missingCounts: WireRecord, types: WireRecord): NormalizedDatasetColumn[] {
  const rawColumns = Array.isArray(source.columns) ? source.columns : []
  const roleMap = firstRecord(source, ['field_roles', 'fieldRoles']) ?? {}
  const missingValues = firstRecord(source, ['missing_values', 'missingValues']) ?? {}
  return names.map((name, index) => {
    const raw = asRecord(rawColumns[index]) ?? (rawColumns.find((item) => stringValue(asRecord(item)?.name) === name) as WireRecord | undefined) ?? {}
    const missingRecord = asRecord(missingValues[name])
    const role = normalizeColumnRole(firstValue(raw, ['role', 'field_role']) ?? roleMap[name])
    const dataType = normalizeColumnType(firstValue(raw, ['data_type', 'dataType']) ?? types[name])
    const missingCount = firstNumber(raw, ['missing_count', 'missingCount'])
      ?? numberValue(missingCounts[name])
      ?? firstNumber(missingRecord ?? {}, ['missing_count', 'missingCount'])
    const missingRatio = firstNumber(raw, ['missing_ratio', 'missingRatio'])
      ?? firstNumber(missingRecord ?? {}, ['missing_ratio', 'missingRatio'])
    return {
      name,
      role,
      roleLabel: role === UNKNOWN ? UNKNOWN_TEXT : role,
      dataType,
      dataTypeLabel: dataType === UNKNOWN ? UNKNOWN_TEXT : dataType,
      // A missing-count is not evidence of the nullable schema flag. Keep
      // the distinction so the page never presents an inferred fact.
      nullable: firstBoolean(raw, ['nullable']),
      missingCount,
      missingRatio,
    }
  })
}

export function normalizeDatasetUpload(value: unknown): NormalizedDatasetUpload {
  const source = asRecord(value) ?? {}
  const storage = firstRecord(source, ['file_storage', 'fileStorage'])
  const names = columnNames(source)
  const missingCounts = firstRecord(source, ['missing_value_counts', 'missingValueCounts']) ?? {}
  const types = firstRecord(source, ['column_types', 'columnTypes']) ?? {}
  const validation = normalizeValidation(source)
  const range = firstRecord(source, ['time_range', 'timeRange'])
  const timeRange = range
    ? { start: stringValue(firstValue(range, ['start', 'min'])), end: stringValue(firstValue(range, ['end', 'max'])) }
    : null
  const statusText = upper(firstValue(source, ['status', 'state']))
  const status = statusText && DATASET_STATUSES.includes(statusText) ? statusText : UNKNOWN
  const previewValue = firstValue(source, ['preview_rows', 'previewRows', 'preview'])
  const previewRows = Array.isArray(previewValue) ? previewValue.map((row: unknown) => asRecord(row) ?? {}) : []
  return {
    id: idOrNull(firstValue(source, ['dataset_id', 'id'])),
    fileName: textOr(firstValue(source, ['file_name', 'fileName', 'filename']), NOT_PROVIDED_TEXT),
    filePath: firstString(source, ['file_path', 'filePath']) ?? stringValue(storage?.relative_path),
    fileSize: firstNumber(source, ['file_size_bytes', 'fileSize']) ?? numberValue(storage?.size_bytes),
    checksumSha256: firstString(source, ['checksum_sha256', 'checksumSha256']) ?? stringValue(storage?.checksum_sha256),
    rowCount: firstNumber(source, ['row_count', 'rowCount']),
    columnCount: firstNumber(source, ['column_count', 'columnCount']) ?? (names.length || null),
    columns: normalizeColumns(source, names, missingCounts, types),
    columnNames: names,
    timeColumn: textOr(firstValue(source, ['time_column', 'timeColumn']), UNKNOWN_TEXT),
    featureColumns: firstStringList(source, ['feature_columns', 'featureColumns']),
    targetColumn: textOr(firstValue(source, ['target_column', 'targetColumn']), UNKNOWN_TEXT),
    columnTypes: types,
    missingValueCounts: missingCounts,
    missingValues: firstRecord(source, ['missing_values', 'missingValues']) ?? {},
    previewRows,
    timeRange,
    validation,
    validationStatus: validationStatus(validation, statusText),
    status,
    createdAt: firstString(source, ['created_at', 'createdAt']),
  }
}

export interface NormalizedQualityChecks {
  valid: boolean | null
  checks: WireRecord[]
  errors: NormalizedValidationIssue[]
  warnings: NormalizedValidationIssue[]
}

export interface NormalizedPreprocessTask {
  id: string | null
  modelType: string
  datasetId: string | null
  preprocessScriptId: string | null
  preprocessUsed: boolean | null
  preprocessStatus: 'USED' | 'UNUSED' | UnknownState
  preprocessMessage: string
  status: NormalizedPreprocessingStatus
  stage: string | UnknownState
  progressStage: string | UnknownState
  currentStage: string | UnknownState
  stageLabel: string
  logs: Array<TrainingLogEntry | string>
  errorMessage: string | null
  inputRowCount: number | null
  outputRowCount: number | null
  inputColumns: string[]
  outputColumns: string[]
  qualityChecks: NormalizedQualityChecks | null
  qualityChecksLabel: string
  startedAt: string | null
  finishedAt: string | null
  createdAt: string | null
  nextStep: string | null
  dataSource: string | UnknownState
}

const PREPROCESS_STAGES = ['waiting', 'data_reading', 'preprocessing', 'validating', 'completed', 'failed']

function normalizeLogs(value: unknown): Array<TrainingLogEntry | string> {
  if (!Array.isArray(value)) return []
  return value.flatMap((entry): Array<TrainingLogEntry | string> => {
    if (typeof entry === 'string') return [entry]
    const source = asRecord(entry)
    if (!source) return []
    const message = stringValue(source.message)
    if (!message) return []
    const level = lower(source.level)
    const normalizedLevel = level === 'debug' || level === 'info' || level === 'warning' || level === 'error' ? level : 'info'
    return [{
      timestamp: textOr(source.timestamp, NOT_PROVIDED_TEXT),
      level: normalizedLevel,
      message,
      stage: stringValue(source.stage),
    }]
  })
}

function normalizeQualityChecks(value: unknown): NormalizedQualityChecks | null {
  const source = asRecord(value)
  if (!source) return null
  const checks = Array.isArray(source.checks) ? source.checks.map((item) => asRecord(item) ?? {}) : []
  return {
    valid: booleanValue(source.valid),
    checks,
    errors: normalizeIssues(source.errors),
    warnings: normalizeIssues(source.warnings),
  }
}

function normalizedStageValue(source: WireRecord): string | UnknownState {
  return firstKnownStage(source, ['current_stage', 'currentStage', 'stage', 'progress_stage', 'progressStage'], PREPROCESS_STAGES)
}

export function normalizePreprocessTask(value: unknown): NormalizedPreprocessTask {
  const source = asRecord(value) ?? {}
  const stage = normalizedStageValue(source)
  const preprocessStatus = upper(firstValue(source, ['preprocess_status', 'preprocessStatus']))
  const qualityChecks = normalizeQualityChecks(firstValue(source, ['quality_checks', 'qualityChecks']))
  const used = firstBoolean(source, ['preprocess_used', 'preprocessUsed'])
  const dataSource = lower(firstValue(source, ['data_source', 'dataSource']))
  return {
    id: idOrNull(source.id),
    modelType: textOr(firstValue(source, ['model_type', 'modelType']), UNKNOWN),
    datasetId: idOrNull(firstValue(source, ['dataset_id', 'datasetId'])),
    preprocessScriptId: idOrNull(firstValue(source, ['preprocess_script_id', 'preprocessScriptId', 'script_id'])),
    preprocessUsed: used,
    preprocessStatus: preprocessStatus === 'USED' || preprocessStatus === 'UNUSED' ? preprocessStatus : UNKNOWN,
    preprocessMessage: textOr(firstValue(source, ['preprocess_message', 'preprocessMessage']), NOT_PROVIDED_TEXT),
    status: normalizeStatus(firstValue(source, ['status', 'state']), ['WAITING', 'RUNNING', 'SUCCEEDED', 'SKIPPED', 'FAILED']),
    stage,
    progressStage: stage,
    currentStage: stage,
    stageLabel: stage === UNKNOWN ? NOT_PROVIDED_TEXT : stage,
    logs: normalizeLogs(firstArray(source, ['logs', 'log_entries'])),
    errorMessage: stringValue(firstValue(source, ['error_message', 'errorMessage'])),
    inputRowCount: firstNumber(source, ['input_row_count', 'inputRowCount']),
    outputRowCount: firstNumber(source, ['output_row_count', 'outputRowCount']),
    inputColumns: firstStringList(source, ['input_columns', 'inputColumns']),
    outputColumns: firstStringList(source, ['output_columns', 'outputColumns']),
    qualityChecks,
    qualityChecksLabel: qualityChecks ? (qualityChecks.valid === null ? UNKNOWN_TEXT : qualityChecks.valid ? '有效' : '无效') : NOT_PROVIDED_TEXT,
    startedAt: firstString(source, ['started_at', 'startedAt']),
    finishedAt: firstString(source, ['finished_at', 'finishedAt']),
    createdAt: firstString(source, ['created_at', 'createdAt']),
    nextStep: firstString(source, ['next_step', 'nextStep']),
    dataSource: dataSource === 'raw' || dataSource === 'preprocessed' ? dataSource : UNKNOWN,
  }
}

export interface NormalizedTrainingJob {
  id: string | null
  modelType: string
  datasetId: string | null
  preprocessScriptId: string | null
  preprocessingTaskId: string | null
  trainScriptId: string | null
  status: NormalizedTrainingStatus
  stage: string | UnknownState
  progressStage: string | UnknownState
  currentStage: string | UnknownState
  progress: number | null
  logs: Array<TrainingLogEntry | string>
  errorMessage: string | null
  modelVersionId: string | null
  trainRowCount: number | null
  testRowCount: number | null
  createdAt: string | null
  startedAt: string | null
  finishedAt: string | null
}

function trainingStage(source: WireRecord): string | UnknownState {
  const stage = firstString(source, ['progress_stage', 'progressStage', 'current_stage', 'currentStage', 'stage'])
  return stage ? stage.toLowerCase() : UNKNOWN
}

export function normalizeTrainingJob(value: unknown): NormalizedTrainingJob {
  const source = asRecord(value) ?? {}
  const stage = trainingStage(source)
  const progressRecord = firstRecord(source, ['progress'])
  return {
    id: idOrNull(source.id),
    modelType: textOr(firstValue(source, ['model_type', 'modelType']), UNKNOWN),
    datasetId: idOrNull(firstValue(source, ['dataset_id', 'datasetId'])),
    preprocessScriptId: idOrNull(firstValue(source, ['preprocess_script_id', 'preprocessScriptId'])),
    preprocessingTaskId: idOrNull(firstValue(source, ['preprocessing_task_id', 'preprocessingTaskId'])),
    trainScriptId: idOrNull(firstValue(source, ['train_script_id', 'trainScriptId'])),
    status: normalizeStatus(firstValue(source, ['status', 'state']), ['PENDING', 'RUNNING', 'PREPROCESSING', 'SPLITTING', 'TRAINING', 'EVALUATING', 'SUCCEEDED', 'FAILED', 'CANCELLED']),
    stage,
    progressStage: stage,
    currentStage: stage,
    progress: firstNumber(source, ['progress', 'progress_percent', 'progressPercentage'])
      ?? firstNumber(progressRecord ?? {}, ['percent', 'percentage', 'progress']),
    logs: normalizeLogs(firstArray(source, ['logs', 'log_entries'])),
    errorMessage: stringValue(firstValue(source, ['error_message', 'errorMessage'])),
    modelVersionId: idOrNull(firstValue(source, ['model_version_id', 'modelVersionId'])),
    trainRowCount: firstNumber(source, ['train_row_count', 'trainRowCount']),
    testRowCount: firstNumber(source, ['test_row_count', 'testRowCount']),
    createdAt: firstString(source, ['created_at', 'createdAt']),
    startedAt: firstString(source, ['started_at', 'startedAt']),
    finishedAt: firstString(source, ['finished_at', 'finishedAt']),
  }
}

export interface NormalizedMetricSet {
  mae: number | null
  rmse: number | null
  mape: number | null
  r2: number | null
  sampleCount: number | null
  mapeValidCount: number | null
  mapeExcludedCount: number | null
  mapeNote: string | null
}

const METRIC_NAMES: readonly MetricName[] = ['mae', 'rmse', 'mape', 'r2']

function normalizeMetricSet(value: unknown): NormalizedMetricSet {
  const source = asRecord(value) ?? {}
  const metric = (name: MetricName): number | null => numberValue(source[name])
  return {
    mae: metric('mae'), rmse: metric('rmse'), mape: metric('mape'), r2: metric('r2'),
    sampleCount: firstNumber(source, ['sample_count', 'sampleCount']),
    mapeValidCount: firstNumber(source, ['mape_valid_count', 'mapeValidCount', 'mape_valid_sample_count']),
    mapeExcludedCount: firstNumber(source, ['mape_excluded_count', 'mapeExcludedCount']),
    mapeNote: stringValue(firstValue(source, ['mape_note', 'mapeNote'])),
  }
}

export interface NormalizedEvaluationModel {
  modelVersionId: string | null
  version: string | null
  metrics: NormalizedMetricSet
}

export interface NormalizedEvaluationChartPoint {
  time: string | null
  actual: number | null
  predicted: number | null
  candidatePrediction: number | null
  baselinePrediction: number | null
  error: number | null
  signedError: number | null
  absoluteError: number | null
  percentageError: number | null
}

export interface NormalizedEvaluationErrorPoint {
  time: string | null
  candidateError: number | null
  baselineError: number | null
  error: number | null
  absoluteError: number | null
  percentageError: number | null
}

export interface NormalizedEvaluationComparison {
  candidate: NormalizedEvaluationModel
  baseline: NormalizedEvaluationModel
  changes: Record<MetricName, number | null>
}

export interface NormalizedEvaluation {
  jobId: string | null
  modelVersionId: string | null
  candidate: NormalizedEvaluationModel
  baseline: NormalizedEvaluationModel
  metrics: { candidate: NormalizedMetricSet; baseline: NormalizedMetricSet; difference: Record<MetricName, number | null> }
  chartData: NormalizedEvaluationChartPoint[]
  errorData: NormalizedEvaluationErrorPoint[]
  modelComparison: NormalizedEvaluationComparison
  sampleCount: number | null
  mapeValidCount: number | null
  mapeExcludedCount: number | null
  chartSampled: boolean | null
  chartTotalCount: number | null
  chartSampleCount: number | null
  testTimeSeries: unknown[]
  timestamps: unknown[]
  actualValues: number[]
  candidatePredictions: number[]
  baselinePredictions: number[]
  candidateErrors: number[]
  baselineErrors: number[]
  errorSeries: number[]
}

function metricDifference(value: unknown): Record<MetricName, number | null> {
  const source = asRecord(value)
  const result: Record<MetricName, number | null> = Object.fromEntries(
    METRIC_NAMES.map((name) => [name, null]),
  ) as Record<MetricName, number | null>
  if (source) {
    METRIC_NAMES.forEach((name) => { result[name] = numberValue(source[name]) })
    return result
  }
  // The first evaluation contract exposed comparison as a list of metric
  // rows. Preserve its real deltas without deriving values from metrics.
  if (Array.isArray(value)) {
    value.forEach((item) => {
      const row = asRecord(item)
      const name = lower(row?.metric)
      if (name && METRIC_NAMES.includes(name as MetricName)) {
        result[name as MetricName] = numberValue(row?.delta)
      }
    })
  }
  return result
}

function evaluationModel(value: unknown, fallbackId: unknown = null): NormalizedEvaluationModel {
  const source = asRecord(value) ?? {}
  return {
    modelVersionId: idOrNull(firstValue(source, ['model_version_id', 'modelVersionId'])) ?? idOrNull(fallbackId),
    version: stringValue(firstValue(source, ['version'])),
    metrics: normalizeMetricSet(source.metrics ?? source),
  }
}

function normalizeChartPoint(value: unknown): NormalizedEvaluationChartPoint {
  const source = asRecord(value) ?? {}
  return {
    time: firstString(source, ['time', 'timestamp']),
    actual: firstNumber(source, ['actual', 'actual_value', 'actualValue']),
    predicted: firstNumber(source, ['predicted', 'candidate_prediction', 'candidatePrediction']),
    candidatePrediction: firstNumber(source, ['candidate_prediction', 'candidatePrediction', 'predicted']),
    baselinePrediction: firstNumber(source, ['baseline_prediction', 'baselinePrediction']),
    error: firstNumber(source, ['error']),
    signedError: firstNumber(source, ['signed_error', 'signedError']),
    absoluteError: firstNumber(source, ['absolute_error', 'absoluteError']),
    percentageError: firstNumber(source, ['percentage_error', 'percentageError']),
  }
}

function normalizeErrorPoint(value: unknown): NormalizedEvaluationErrorPoint {
  const source = asRecord(value) ?? {}
  return {
    time: firstString(source, ['time', 'timestamp']),
    candidateError: firstNumber(source, ['candidate_error', 'candidateError']),
    baselineError: firstNumber(source, ['baseline_error', 'baselineError']),
    error: firstNumber(source, ['error']),
    absoluteError: firstNumber(source, ['absolute_error', 'absoluteError']),
    percentageError: firstNumber(source, ['percentage_error', 'percentageError']),
  }
}

function numericList(value: unknown): number[] {
  return Array.isArray(value) ? value.filter((item): item is number => numberValue(item) !== null) : []
}

export function normalizeEvaluation(value: unknown): NormalizedEvaluation {
  const source = asRecord(value) ?? {}
  const metricEnvelope = firstRecord(source, ['metrics'])
  const modelMetrics = firstRecord(source, ['model_metrics', 'modelMetrics'])
  const comparisonValue = firstValue(source, ['model_comparison', 'modelComparison', 'comparison'])
  const comparison = asRecord(comparisonValue) ?? {}
  const candidateBlock = firstRecord(metricEnvelope ?? {}, ['candidate'])
    ?? firstRecord(modelMetrics ?? {}, ['candidate'])
    ?? firstRecord(source, ['candidate'])
    ?? firstRecord(comparison, ['candidate'])
  const baselineBlock = firstRecord(metricEnvelope ?? {}, ['baseline'])
    ?? firstRecord(modelMetrics ?? {}, ['baseline'])
    ?? firstRecord(source, ['baseline'])
    ?? firstRecord(comparison, ['baseline'])
  // Older responses put the candidate metric set directly in `metrics`.
  const flatMetricBlock = metricEnvelope && !('candidate' in metricEnvelope) && !('baseline' in metricEnvelope)
    ? metricEnvelope
    : null
  const candidate = evaluationModel(candidateBlock ?? flatMetricBlock, firstValue(source, ['model_version_id', 'modelVersionId']))
  const baseline = evaluationModel(baselineBlock)
  const difference = metricDifference(
    metricEnvelope?.difference
      ?? metricEnvelope?.differences
      ?? modelMetrics?.difference
      ?? modelMetrics?.differences
      ?? source.metric_differences
      ?? source.metricDifferences
      ?? source.metrics_difference
      ?? comparison.changes
      ?? comparisonValue,
  )
  const chartEnvelope = firstRecord(source, ['chart_data', 'chartData', 'chart'])
  const chartRows = Array.isArray(source.chart_data) || Array.isArray(source.chartData)
    ? (Array.isArray(source.chart_data) ? source.chart_data : source.chartData)
    : chartEnvelope?.actual_vs_prediction
  const errorRows = Array.isArray(source.error_data) || Array.isArray(source.errorData)
    ? (Array.isArray(source.error_data) ? source.error_data : source.errorData)
    : chartEnvelope?.error_series
  const candidateMetrics = candidate.metrics
  return {
    jobId: idOrNull(firstValue(source, ['job_id', 'jobId'])),
    modelVersionId: idOrNull(firstValue(source, ['model_version_id', 'modelVersionId'])),
    candidate,
    baseline,
    metrics: { candidate: candidateMetrics, baseline: baseline.metrics, difference },
    chartData: Array.isArray(chartRows) ? chartRows.map(normalizeChartPoint) : [],
    errorData: Array.isArray(errorRows) ? errorRows.map(normalizeErrorPoint) : [],
    modelComparison: { candidate, baseline, changes: difference },
    sampleCount: candidateMetrics.sampleCount ?? firstNumber(source, ['sample_count', 'sampleCount']),
    mapeValidCount: candidateMetrics.mapeValidCount ?? firstNumber(source, ['mape_valid_count', 'mapeValidCount', 'mape_valid_sample_count']),
    mapeExcludedCount: candidateMetrics.mapeExcludedCount ?? firstNumber(source, ['mape_excluded_count', 'mapeExcludedCount']),
    chartSampled: firstBoolean(source, ['chart_sampled', 'chartSampled']) ?? booleanValue(chartEnvelope?.sampled),
    chartTotalCount: firstNumber(source, ['chart_total_count', 'chartTotalCount']) ?? firstNumber(chartEnvelope ?? {}, ['source_row_count', 'sourceRowCount']),
    chartSampleCount: firstNumber(source, ['chart_sample_count', 'chartSampleCount']),
    testTimeSeries: Array.isArray(source.test_time_series) ? source.test_time_series : [],
    timestamps: Array.isArray(source.timestamps) ? source.timestamps : [],
    actualValues: numericList(source.actual_values),
    candidatePredictions: numericList(source.candidate_predictions),
    baselinePredictions: numericList(source.baseline_predictions),
    candidateErrors: numericList(source.candidate_errors),
    baselineErrors: numericList(source.baseline_errors),
    errorSeries: numericList(source.error_series),
  }
}

export interface NormalizedModel {
  id: string | null
  modelType: string
  version: string
  lifecycleStatus: string | UnknownState
  healthStatus: NormalizedHealthStatus
  status: string | UnknownState
  compatibilityStatus: string | null
  compatibilityHealthStatus: string | null
  isBaseline: boolean | null
  isCurrent: boolean | null
  isAbnormal: boolean | null
  isRollbackAvailable: boolean | null
  metrics: WireRecord | null
  modelPath: string | null
  preprocessorPath: string | null
  trainingJobId: string | null
  modelVersionId: string | null
  trainScriptId: string | null
  preprocessScriptId: string | null
  featureColumns: string[]
  timeColumn: string | null
  targetColumn: string | null
  createdAt: string | null
  publishedAt: string | null
}

const MODEL_LIFECYCLE = ['READY', 'PUBLISHED', 'RETIRED', 'FAILED']
const HEALTH = ['HEALTHY', 'ABNORMAL', 'UNKNOWN']

export function normalizeModel(value: unknown): NormalizedModel {
  const source = asRecord(value) ?? {}
  const rawStatus = upper(firstValue(source, ['status', 'state']))
  const rawHealth = upper(firstValue(source, ['health_status', 'healthStatus']))
  const lifecycleStatus = rawStatus && MODEL_LIFECYCLE.includes(rawStatus) ? rawStatus : UNKNOWN
  const healthStatus: NormalizedHealthStatus = rawStatus === 'ABNORMAL'
    ? 'ABNORMAL'
    : rawHealth && HEALTH.includes(rawHealth) ? rawHealth as NormalizedHealthStatus : 'UNKNOWN'
  return {
    id: idOrNull(source.id),
    modelType: textOr(firstValue(source, ['model_type', 'modelType']), UNKNOWN),
    version: textOr(source.version, UNKNOWN_TEXT),
    lifecycleStatus,
    healthStatus,
    status: lifecycleStatus,
    compatibilityStatus: rawStatus && !MODEL_LIFECYCLE.includes(rawStatus) ? rawStatus : null,
    compatibilityHealthStatus: rawHealth && !HEALTH.includes(rawHealth) ? rawHealth : null,
    isBaseline: firstBoolean(source, ['is_baseline', 'isBaseline']),
    isCurrent: firstBoolean(source, ['is_current', 'isCurrent']),
    isAbnormal: firstBoolean(source, ['is_abnormal', 'isAbnormal']) ?? (healthStatus === 'ABNORMAL' ? true : null),
    isRollbackAvailable: firstBoolean(source, ['is_rollback_available', 'isRollbackAvailable']),
    metrics: firstRecord(source, ['metrics']),
    modelPath: firstString(source, ['model_path', 'modelPath']),
    preprocessorPath: firstString(source, ['preprocessor_path', 'preprocessorPath']),
    trainingJobId: idOrNull(firstValue(source, ['training_job_id', 'trainingJobId'])),
    modelVersionId: idOrNull(firstValue(source, ['model_version_id', 'modelVersionId'])),
    trainScriptId: idOrNull(firstValue(source, ['train_script_id', 'trainScriptId'])),
    preprocessScriptId: idOrNull(firstValue(source, ['preprocess_script_id', 'preprocessScriptId'])),
    featureColumns: firstStringList(source, ['feature_columns', 'featureColumns']),
    timeColumn: firstString(source, ['time_column', 'timeColumn']),
    targetColumn: firstString(source, ['target_column', 'targetColumn']),
    createdAt: firstString(source, ['created_at', 'createdAt']),
    publishedAt: firstString(source, ['published_at', 'publishedAt']),
  }
}

export interface NormalizedAlert {
  id: string | null
  modelType: string
  modelVersionId: string | null
  reason: string
  rollbackFrom: string | null
  rollbackTo: string | null
  status: NormalizedAlertStatus
  compatibilityStatus: string | null
  createdAt: string | null
  acknowledgedAt: string | null
  resolvedAt: string | null
}

export function normalizeAlert(value: unknown): NormalizedAlert {
  const source = asRecord(value) ?? {}
  const rawStatus = upper(firstValue(source, ['status', 'state']))
  const status = rawStatus === 'ACTIVE' || rawStatus === 'ACKNOWLEDGED' || rawStatus === 'RESOLVED'
    ? rawStatus as ModelAlertState
    : UNKNOWN
  return {
    id: idOrNull(source.id),
    modelType: textOr(firstValue(source, ['model_type', 'modelType']), UNKNOWN),
    // Deliberately do not use version labels or nested model objects as IDs.
    modelVersionId: idOrNull(firstValue(source, ['model_version_id', 'modelVersionId'])),
    reason: textOr(source.reason, NOT_PROVIDED_TEXT),
    rollbackFrom: idOrNull(firstValue(source, ['rollback_from', 'rollbackFrom', 'from_version_id', 'fromVersionId'])),
    rollbackTo: idOrNull(firstValue(source, ['rollback_to', 'rollbackTo', 'to_version_id', 'toVersionId'])),
    status,
    compatibilityStatus: rawStatus && status === UNKNOWN ? rawStatus : null,
    createdAt: firstString(source, ['created_at', 'createdAt']),
    acknowledgedAt: firstString(source, ['acknowledged_at', 'acknowledgedAt']),
    resolvedAt: firstString(source, ['resolved_at', 'resolvedAt']),
  }
}

/** Explicit alias for callers that name the resource rather than its page. */
export const normalizeModelVersion = normalizeModel
export const normalizeEvaluationResult = normalizeEvaluation
