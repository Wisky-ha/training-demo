import { describe, expect, it } from 'vitest'
import {
  NOT_PROVIDED_TEXT,
  UNKNOWN,
  UNKNOWN_TEXT,
  normalizeAlert,
  normalizeDatasetUpload,
  normalizeEvaluation,
  normalizeModel,
  normalizePreprocessTask,
  normalizeTrainingJob,
} from './normalizers'

describe('API normalizers', () => {
  it('maps upload metadata, validation, missing counts and column fallbacks', () => {
    const result = normalizeDatasetUpload({
      dataset_id: 'dataset-1',
      file_name: 'load.csv',
      file_size_bytes: 42,
      missing_value_counts: { temperature: 2 },
      time_range: { start: '2024-01-01', end: '2024-01-02' },
      validation: {
        valid: true,
        errors: [],
        warnings: [{ code: 'TIME_NOT_SORTED', field: 'time', message: 'sort' }],
        checks: { columns: { valid: true } },
      },
      column_names: ['time', 'temperature'],
    })

    expect(result.id).toBe('dataset-1')
    expect(result.fileSize).toBe(42)
    expect(result.missingValueCounts).toEqual({ temperature: 2 })
    expect(result.timeRange).toEqual({ start: '2024-01-01', end: '2024-01-02' })
    expect(result.validationStatus).toBe('VALID')
    expect(result.validation.warnings).toHaveLength(1)
    expect(result.columns[0]).toMatchObject({ role: UNKNOWN, dataType: UNKNOWN, nullable: null })
    expect(result.columns[0].roleLabel).toBe(UNKNOWN_TEXT)
  })

  it('normalizes preprocessing aliases without inventing quality checks', () => {
    const result = normalizePreprocessTask({
      id: 'task-1',
      dataset_id: 'dataset-1',
      status: 'succeeded',
      current_stage: 'not-a-stage',
      stage: 'COMPLETED',
      progress_stage: 'failed',
      preprocess_status: 'unused',
      logs: ['done'],
    })

    expect(result.status).toBe('SUCCEEDED')
    expect(result.stage).toBe('completed')
    expect(result.progressStage).toBe('completed')
    expect(result.currentStage).toBe('completed')
    expect(result.preprocessStatus).toBe('UNUSED')
    expect(result.qualityChecks).toBeNull()
    expect(result.qualityChecksLabel).toBe(NOT_PROVIDED_TEXT)
  })

  it('preserves training null semantics and normalizes unknown status', () => {
    const result = normalizeTrainingJob({
      id: 'job-1',
      status: 'future-state',
      progress_stage: 'TRAINING',
      logs: ['running'],
      model_version_id: null,
    })

    expect(result.status).toBe(UNKNOWN)
    expect(result.stage).toBe('training')
    expect(result.modelVersionId).toBeNull()
    expect(result.logs).toEqual(['running'])
  })

  it('maps the flat legacy metric block and keeps evaluation blocks empty', () => {
    const result = normalizeEvaluation({
      job_id: 'job-1',
      model_version_id: 'model-1',
      metrics: { mae: 1, sample_count: 3, mape_valid_count: 2, mape_excluded_count: 1 },
      chart_data: { actual_vs_prediction: [], error_series: [] },
      error_data: [],
    })

    expect(result.candidate.metrics).toMatchObject({ mae: 1, sampleCount: 3, mapeValidCount: 2, mapeExcludedCount: 1 })
    expect(result.baseline.metrics.mae).toBeNull()
    expect(result.chartData).toEqual([])
    expect(result.errorData).toEqual([])
  })

  it('keeps evaluation blocks empty and metrics counts null when absent', () => {
    const result = normalizeEvaluation({
      job_id: 'job-1',
      model_version_id: 'model-1',
      metrics: { candidate: { mae: 1, sample_count: 3 }, baseline: { mape: null } },
      chart_data: [],
      error_data: [],
      model_comparison: {},
    })

    expect(result.candidate.metrics.mae).toBe(1)
    expect(result.baseline.metrics.mape).toBeNull()
    expect(result.sampleCount).toBe(3)
    expect(result.mapeValidCount).toBeNull()
    expect(result.chartData).toEqual([])
    expect(result.errorData).toEqual([])
    expect(result.modelComparison.candidate.modelVersionId).toBe('model-1')
    expect(result.modelComparison.baseline.modelVersionId).toBeNull()
  })

  it('separates legacy abnormal lifecycle from health and does not default health', () => {
    expect(normalizeModel({ id: 'model-1', model_type: 'electric_load', version: 'v1', status: 'ABNORMAL' })).toMatchObject({
      lifecycleStatus: UNKNOWN,
      healthStatus: 'ABNORMAL',
      compatibilityStatus: 'ABNORMAL',
    })
    expect(normalizeModel({ id: 'model-2', model_type: 'electric_load', version: 'v2', status: 'READY' }).healthStatus).toBe(UNKNOWN)
  })

  it('keeps unknown model and alert enums explicit', () => {
    expect(normalizeModel({ id: 'model-1', model_type: 'electric_load', version: 'v1', status: 'future', health_status: 'future' })).toMatchObject({
      lifecycleStatus: UNKNOWN,
      healthStatus: UNKNOWN,
      compatibilityStatus: 'FUTURE',
      compatibilityHealthStatus: 'FUTURE',
    })
    expect(normalizeTrainingJob({ status: 'future', progress_stage: null }).stage).toBe(UNKNOWN)
  })

  it('does not turn missing or unknown alert status into ACTIVE or invent resource links', () => {
    expect(normalizeAlert({ id: 'alert-1', model_type: 'electric_load', model_version: 'v1' })).toMatchObject({
      status: UNKNOWN,
      modelVersionId: null,
      rollbackFrom: null,
      rollbackTo: null,
    })
    expect(normalizeAlert({ id: 'alert-2', model_type: 'electric_load', status: 'acknowledged', model_version_id: 'model-1' }).status).toBe('ACKNOWLEDGED')
  })

  it('maps legacy upload aliases and nested missing-value summaries without inventing state', () => {
    const result = normalizeDatasetUpload({
      id: 'dataset-legacy',
      fileName: 'legacy.csv',
      fileSize: 0,
      columnNames: ['feature'],
      missingValues: { feature: { missing_count: 2, missing_ratio: 0.5 } },
      validation_result: { valid: null, errors: [], warnings: [], checks: {} },
      state: 'future-state',
    })

    expect(result).toMatchObject({
      id: 'dataset-legacy',
      fileName: 'legacy.csv',
      fileSize: 0,
      status: UNKNOWN,
      validationStatus: UNKNOWN,
    })
    expect(result.columns).toEqual([expect.objectContaining({
      name: 'feature', missingCount: 2, missingRatio: 0.5,
    })])
  })

  it('keeps null input and empty evaluation arrays as explicit empty states', () => {
    const dataset = normalizeDatasetUpload(null)
    expect(dataset).toMatchObject({
      id: null,
      fileName: NOT_PROVIDED_TEXT,
      status: UNKNOWN,
      validationStatus: UNKNOWN,
      columns: [],
    })

    const evaluation = normalizeEvaluation({ metrics: null, chart_data: [], error_data: [] })
    expect(evaluation.chartData).toEqual([])
    expect(evaluation.errorData).toEqual([])
    expect(evaluation.candidate.metrics).toMatchObject({ mae: null, rmse: null, mape: null, r2: null })
  })
})
