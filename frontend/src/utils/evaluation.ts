import type { EvaluationMetrics, MetricName, ModelEvaluation } from '../types/contracts'

export const MAX_RENDER_POINTS = 600

export interface NormalizedEvaluationPoint {
  timestamp: string
  actual: number
  candidate: number
  baseline: number | null
  error: number
  percentageError: number | null
}

export interface EvaluationViewData {
  points: NormalizedEvaluationPoint[]
  sourceCount: number
  serverSampled: boolean
  clientSampled: boolean
  invalidCount: number
}

export interface EvaluationModelResult {
  version: string
  metrics: Partial<EvaluationMetrics>
  source?: string
}

const METRIC_NAMES: MetricName[] = ['mae', 'rmse', 'mape', 'r2']

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function stringValue(value: unknown): string | null {
  if (typeof value === 'string' && value.trim()) return value
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.toISOString()
  return value == null ? null : String(value)
}

function chartRows(evaluation: ModelEvaluation): unknown[] {
  const chart = evaluation.chart_data
  if (Array.isArray(chart)) return chart
  const chartRecord = record(chart)
  if (chartRecord && Array.isArray(chartRecord.actual_vs_prediction)) return chartRecord.actual_vs_prediction
  return []
}

function evenSample<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items
  const indexes = new Set<number>()
  for (let index = 0; index < limit; index += 1) {
    indexes.add(Math.round(index * (items.length - 1) / (limit - 1)))
  }
  return [...indexes].sort((a, b) => a - b).map((index) => items[index])
}

/**
 * Converts the endpoint's historical `time`/`predicted` names as well as the
 * newer contract names into one safe chart shape. Invalid chart rows are
 * omitted from rendering only; metric values still come from the backend.
 */
export function normalizeEvaluationData(evaluation: ModelEvaluation): EvaluationViewData {
  const rows = chartRows(evaluation)
  let invalidCount = 0
  const points = rows.flatMap((value) => {
    const row = record(value)
    if (!row) {
      invalidCount += 1
      return []
    }
    const timestamp = stringValue(row.timestamp ?? row.time)
    const actual = numberValue(row.actual)
    const candidate = numberValue(row.candidate_prediction ?? row.predicted ?? row.prediction)
    if (!timestamp || actual === null || candidate === null) {
      invalidCount += 1
      return []
    }
    const baseline = numberValue(row.baseline_prediction ?? row.baseline_predicted)
    const error = numberValue(row.error ?? row.signed_error) ?? actual - candidate
    const percentageError = numberValue(row.percentage_error)
      ?? (actual === 0 ? null : Math.abs(error / actual) * 100)
    return [{ timestamp, actual, candidate, baseline, error, percentageError }]
  })
  const sourceCount = numberValue(evaluation.chart_total_count)
    ?? numberValue(record(evaluation.metrics)?.sample_count)
    ?? rows.length
  const serverSampled = Boolean(evaluation.chart_sampled) || sourceCount > rows.length
  const rendered = evenSample(points, MAX_RENDER_POINTS)
  return {
    points: rendered,
    sourceCount: Math.max(sourceCount, points.length),
    serverSampled,
    clientSampled: rendered.length < points.length,
    invalidCount,
  }
}

export function evaluationMetrics(evaluation: ModelEvaluation | null): Partial<EvaluationMetrics> {
  if (!evaluation) return {}
  const metrics = record(evaluation.metrics)
  if (metrics) return metrics as Partial<EvaluationMetrics>
  return evaluation.candidate ?? {}
}

function metricsFrom(value: unknown): Partial<EvaluationMetrics> | null {
  const source = record(value)
  if (!source) return null
  return source.metrics && record(source.metrics)
    ? source.metrics as Partial<EvaluationMetrics>
    : source as Partial<EvaluationMetrics>
}

/** Reads both the current `model_comparison` envelope and older baseline fields. */
export function evaluationModels(evaluation: ModelEvaluation | null): {
  candidate: EvaluationModelResult | null
  production: EvaluationModelResult | null
} {
  if (!evaluation) return { candidate: null, production: null }
  const comparison = record(evaluation.model_comparison) ?? record(evaluation.comparison)
  const candidateMetrics = metricsFrom(comparison?.candidate)
    ?? metricsFrom(comparison?.new_model)
    ?? evaluation.candidate
    ?? evaluationMetrics(evaluation)
  const productionValue = comparison?.production ?? comparison?.current_model ?? comparison?.baseline ?? evaluation.baseline
  const productionMetrics = metricsFrom(productionValue)
  const candidate = candidateMetrics
    ? { version: String(record(comparison?.candidate)?.version ?? record(comparison?.new_model)?.version ?? '本次新模型'), metrics: candidateMetrics, source: 'evaluation' }
    : null
  const production = productionMetrics
    ? { version: String(record(productionValue)?.version ?? '当前生产模型'), metrics: productionMetrics, source: String(record(productionValue)?.source ?? 'evaluation') }
    : null
  return { candidate, production }
}

export function metricDelta(candidate: number | null | undefined, production: number | null | undefined): number | null {
  return candidate != null && production != null && Number.isFinite(candidate) && Number.isFinite(production)
    ? candidate - production
    : null
}

export function metricLabel(metric: MetricName): string {
  return ({ mae: 'MAE', rmse: 'RMSE', mape: 'MAPE', r2: 'R²' })[metric]
}

export { METRIC_NAMES }
