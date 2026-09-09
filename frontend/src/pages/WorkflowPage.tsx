import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ApiError, apiClient, createIdempotencyKey } from '../api'
import { EvaluationDashboard } from '../components/evaluation/EvaluationCharts'
import { useAppStore } from '../store/useAppStore'
import {
  MODEL_TYPE_CODES,
  MODEL_TYPE_NAMES,
  type DatasetSplitResult,
  type DatasetUploadResult,
  type ModelEvaluation,
  type JsonRecord,
  type ModelTypeCode,
  type ModelVersionDetail,
  type PreprocessTask,
  type ScriptContract,
  type TrainingJob,
  type TrainingLog,
  type ModelVersionSummary,
} from '../types/contracts'
import { workflowSteps, type WorkflowDraft, type WorkflowStepId } from '../types/workflow'

const stepIndexes = Object.fromEntries(
  workflowSteps.map((step, index) => [step.id, index]),
) as Record<WorkflowStepId, number>

const terminalTrainingStatuses = new Set(['SUCCEEDED', 'FAILED', 'CANCELLED'])

const hasOwn = (value: object, key: PropertyKey): boolean =>
  Object.prototype.hasOwnProperty.call(value, key)

const upperStatus = (value: string | null | undefined): string | null =>
  typeof value === 'string' && value.trim() ? value.toUpperCase() : null

function errorMessage(error: unknown) {
  return error instanceof ApiError
    ? error.message
    : error instanceof Error
      ? error.message
      : '请求失败，请稍后重试'
}

function isNotFound(error: unknown): boolean {
  return (error instanceof ApiError && error.status === 404)
    || (typeof error === 'object' && error !== null && 'status' in error && error.status === 404)
}

function asLogs(logs: Array<{ message: string } | string> | undefined) {
  return (logs ?? []).map((item) => typeof item === 'string' ? item : item.message)
}

function mergeTrainingLogs(current: TrainingLog[], incoming: TrainingLog[]): TrainingLog[] {
  const result = [...current]
  const keyFor = (item: TrainingLog) => typeof item === 'string' ? item : `${item.timestamp}|${item.level}|${item.message}`
  const keys = new Set(current.map(keyFor))
  incoming.forEach((item) => {
    const key = keyFor(item)
    if (!keys.has(key)) {
      keys.add(key)
      result.push(item)
    }
  })
  return result
}

function ErrorBox({ message, onRetry }: { message: string | null; onRetry?: () => void }) {
  if (!message) return null
  return (
    <div className="alert-box error" role="alert">
      <span>{message}</span>
      {onRetry && <button type="button" onClick={onRetry}>重试</button>}
    </div>
  )
}

function InfoBox({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warning' | 'success' }) {
  return <div className={`alert-box ${tone}`} role="status">{children}</div>
}

function ContractNote({ title, children }: { title: string; children: ReactNode }) {
  return (
    <InfoBox>
      <div className="workflow-contract-note">
        <strong>{title}</strong>
        {children}
      </div>
    </InfoBox>
  )
}

function Button({
  children,
  disabled,
  onClick,
  kind = 'primary',
  type = 'button',
}: {
  children: ReactNode
  disabled?: boolean
  onClick?: () => void
  kind?: 'primary' | 'secondary' | 'danger'
  type?: 'button' | 'submit'
}) {
  return <button className={`${kind}-button action-button`} disabled={disabled} onClick={onClick} type={type}>{children}</button>
}

function StepActions({
  back,
  next,
  nextDisabled,
  nextLabel = '继续',
}: {
  back?: () => void
  next?: () => void
  nextDisabled?: boolean
  nextLabel?: string
}) {
  return (
    <div className="step-actions">
      {back && <Button kind="secondary" onClick={back}>返回上一步</Button>}
      {next && <Button disabled={nextDisabled} onClick={next}>{nextLabel} <span aria-hidden="true">→</span></Button>}
    </div>
  )
}

function ModelTypeStep({ select }: { select: (type: ModelTypeCode) => void }) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const navigate = useNavigate()
  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 01 / MODEL TYPE</p>
          <h2>选择要训练的模型类型</h2>
        </div>
        <span className="panel-count">{modelType ? '已选择' : '必选'}</span>
      </div>
      <div className="model-select-grid">
        {MODEL_TYPE_CODES.map((code) => (
          <button
            className={`model-option${modelType === code ? ' selected' : ''}`}
            key={code}
            onClick={() => select(code)}
            type="button"
          >
            <span className="model-option-icon">{code === 'electric_load' ? '⚡' : code === 'heating_cooling_load' ? '◒' : '⌁'}</span>
            <span>
              <strong>{MODEL_TYPE_NAMES[code]}</strong>
              <em>{code}</em>
            </span>
            <b aria-hidden="true">{modelType === code ? '✓' : '○'}</b>
          </button>
        ))}
      </div>
      <StepActions next={() => navigate('/workflow/upload')} nextDisabled={!modelType} nextLabel="进入数据上传" />
    </section>
  )
}

function displayValue(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === '' ? '未知' : String(value)
}

function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '未知'
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

function validationLabel(dataset: DatasetUploadResult): string {
  if (dataset.validation?.valid === true) return '通过'
  if (dataset.validation?.valid === false) return '失败'
  const wireStatus = (dataset.validation as unknown as { status?: unknown } | null)?.status
  if (typeof wireStatus === 'string' && wireStatus.trim()) return wireStatus.toUpperCase()
  return '未知'
}

function DatasetSummary({ dataset }: { dataset: DatasetUploadResult }) {
  const range = dataset.time_range
  return (
    <div className="dataset-summary">
      <div className="file-facts" aria-label="上传数据状态">
        <span>文件名：<b>{displayValue(dataset.file_name)}</b></span>
        <span>数据集 ID：<b>{displayValue(dataset.id ?? dataset.dataset_id)}</b></span>
        <span>文件大小：<b>{formatBytes(dataset.file_size_bytes)}</b></span>
        <span>校验状态：<b>{validationLabel(dataset)}</b></span>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>字段名</th><th>角色</th><th>类型-缺失</th></tr></thead>
          <tbody>
            {(dataset.columns ?? []).length ? dataset.columns.map((column) => (
              <tr key={column.name}>
                <td><b>{displayValue(column.name)}</b></td>
                <td><span className={`role-pill ${column.role ?? 'unknown'}`}>{column.role === 'time' ? '时间列' : column.role === 'target' ? '目标 y' : column.role === 'feature' ? '特征 X' : '未识别'}</span></td>
                <td>{displayValue(column.data_type)} - 缺失 {displayValue(column.missing_count)}</td>
              </tr>
            )) : <tr><td colSpan={3}>暂无字段摘要</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="metadata-grid" aria-label="数据集元数据">
        <div><b>{displayValue(dataset.id ?? dataset.dataset_id)}</b><span>数据集 ID</span></div>
        <div><b>{displayValue(dataset.row_count)}</b><span>行数</span></div>
        <div><b>{displayValue(dataset.column_count)}</b><span>列数</span></div>
        <div><b>{range ? `${displayValue(range.start)} ～ ${displayValue(range.end)}` : '未知'}</b><span>时间范围</span></div>
      </div>
    </div>
  )
}

function UploadStep({
  setDataset,
  back,
  restoring,
}: {
  setDataset: (dataset: DatasetUploadResult) => void
  back: () => void
  restoring: boolean
}) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const dataset = useAppStore((state) => state.workflow.dataset)
  const navigate = useNavigate()
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const alive = useRef(true)

  useEffect(() => () => { alive.current = false }, [])

  const upload = async (file?: File) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('仅支持 .csv 文件，请重新选择')
      return
    }
    setUploading(true)
    setError(null)
    try {
      const result = await apiClient.uploadDataset(file, { model_type: modelType ?? undefined })
      // The upload may finish after a model switch. Do not let a response from
      // the unmounted/old workflow repopulate the newly selected chain.
      if (!alive.current || useAppStore.getState().workflow.modelType !== modelType) return
      setDataset(result)
    } catch (reason) {
      if (alive.current) setError(errorMessage(reason))
    } finally {
      if (alive.current) setUploading(false)
    }
  }

  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 02 / DATASET</p>
          <h2>上传并检查 CSV 数据</h2>
        </div>
      </div>
      <ErrorBox message={error} />
      <ContractNote title="平台输入校验（执行契约，不是业务逻辑说明）">
        <ul>
          <li>文件格式：仅接受 <code>.csv</code>；内容编码为 UTF-8 或 GB18030；文件大小不超过 50 MB。</li>
          <li>字段位置：首列必须是可解析且不重复的时间列；末列必须是有限数值目标列；中间列作为特征。</li>
          <li>数据量与特征：至少 2 行；特征允许缺失，但不能整列为空。</li>
        </ul>
        <p>以上是平台输入校验；平台只读取并保存，原始文件不改写。</p>
      </ContractNote>
      {!dataset && (
        <label
          className={`drop-zone${dragging ? ' dragging' : ''}`}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragging(false)
            void upload(event.dataTransfer.files[0])
          }}
        >
          <input accept=".csv,text/csv" onChange={(event) => void upload(event.target.files?.[0])} type="file" />
          <span className="upload-icon">↑</span>
          <strong>{uploading ? '正在上传并解析…' : '拖拽 CSV 文件到这里'}</strong>
          <small>或点击选择文件</small>
        </label>
      )}
      {uploading && <div className="loading-line"><span className="spinner" />上传、解析与校验进行中</div>}
      {dataset && (
        <>
          <div className="file-chip"><span>CSV</span><b>{displayValue(dataset.file_name)}</b><small>{validationLabel(dataset)}</small></div>
          <DatasetSummary dataset={dataset} />
        </>
      )}
      <StepActions back={back} next={() => navigate('/workflow/preprocess')} nextDisabled={!dataset || uploading || restoring} />
    </section>
  )
}

function LogList({ logs }: { logs: Array<{ message: string } | string> | undefined }) {
  const entries = asLogs(logs)
  return (
    <div className="log-box">
      {entries.length
        ? entries.map((log, index) => <div key={`${log}-${index}`}><span>•</span>{log}</div>)
        : <span className="muted-caption">暂无日志</span>}
    </div>
  )
}

function PreprocessStep({
  back,
  complete,
  restoring,
}: {
  back: () => void
  complete: (task: PreprocessTask, scriptId: string | null) => void
  restoring: boolean
}) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const datasetId = useAppStore((state) => state.workflow.datasetId)
  const task = useAppStore((state) => state.workflow.preprocessTask)
  const selected = useAppStore((state) => state.workflow.preprocessScriptId)
  const splitId = useAppStore((state) => state.workflow.splitId)
  const trainingJobId = useAppStore((state) => state.workflow.trainingJobId)
  const modelVersionId = useAppStore((state) => state.workflow.modelVersionId)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const navigate = useNavigate()
  const [scripts, setScripts] = useState<ScriptContract[]>([])
  const [skip, setSkip] = useState(!selected)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [uploadingScript, setUploadingScript] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setSkip(upperStatus(task?.status) === 'SKIPPED' || !selected)
  }, [task?.status, selected])

  useEffect(() => {
    if (!modelType) {
      setScripts([])
      setLoading(false)
      return
    }
    let alive = true
    setLoading(true)
    apiClient.listScripts({ model_type: modelType, script_type: 'preprocessor', status: 'ENABLED' })
      .then((response) => { if (alive) setScripts(response.items.filter((script) => upperStatus(script.status) === 'ENABLED')) })
      .catch((reason) => { if (alive) setError(errorMessage(reason)) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [modelType])

  const downstreamLocked = Boolean(splitId || trainingJobId || modelVersionId)

  const changeSelection = (nextSkip: boolean, scriptId: string | null = null) => {
    if (downstreamLocked) {
      setError('固定 80/20 划分已经完成，不能更换预处理；请从首页开始新的训练链路。')
      return
    }
    setSkip(nextSkip)
    setContext({
      preprocessScriptId: scriptId,
      preprocessTaskId: null,
      preprocessTask: null,
      splitId: null,
      split: null,
      trainScriptId: null,
      trainingJobId: null,
      trainingJob: null,
      modelVersionId: null,
      evaluation: null,
      modelVersion: null,
    })
  }

  const uploadScript = async (file?: File) => {
    if (downstreamLocked) {
      setError('固定 80/20 划分已经完成，不能更换预处理；请从首页开始新的训练链路。')
      return
    }
    if (!file || !modelType) return
    if (!file.name.toLowerCase().endsWith('.py')) {
      setError('仅支持 .py 文件')
      return
    }
    setUploadingScript(true)
    setError(null)
    try {
      const script = await apiClient.uploadScript({ file, name: file.name, script_type: 'preprocessor', supported_model_types: [modelType] })
      if (upperStatus(script.status) !== 'ENABLED') {
        setError('上传脚本未处于 ENABLED 状态，不能用于预处理')
        return
      }
      setScripts((current) => [...current.filter((item) => item.id !== script.id), script])
      changeSelection(false, script.id)
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setUploadingScript(false)
    }
  }

  const run = async () => {
    if (downstreamLocked) {
      setError('固定 80/20 划分已经完成，不能重新执行预处理；请从首页开始新的训练链路。')
      return
    }
    if (!datasetId || !modelType || (!skip && !selected)) return
    setRunning(true)
    setError(null)
    try {
      const result = await apiClient.createPreprocessingTask({
        model_type: modelType,
        dataset_id: datasetId,
        preprocess_script_id: skip ? null : selected,
        mode: skip ? 'skip' : 'use',
        skip,
      })
      complete(result, skip ? null : selected)
      navigate('/workflow/split')
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setRunning(false)
    }
  }

  const taskStatus = upperStatus(task?.status)
  const qualityChecks = task?.quality_checks?.checks ?? []
  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 03 / PREPROCESS</p>
          <h2>数据预处理</h2>
        </div>
      </div>
      <ErrorBox message={error} />
      <ContractNote title="平台执行契约（不限制具体业务算法）">
        <ul>
          <li>脚本文件：仅接受 <code>.py</code>；使用 UTF-8；大小不超过 5 MiB；Python 语法必须有效。</li>
          <li>执行接口：必须且只能定义一个 <code>Preprocessor</code> 类；<code>fit(df, config)</code> 与 <code>transform(df, config)</code> 必须是可调用的两参数方法；<code>fit</code> 返回 <code>self</code>，<code>transform</code> 返回 <code>pandas.DataFrame</code>。</li>
          <li>输出契约：输出行数须与输入一致；保留时间列、目标列和至少一个特征；训练集与测试集输出字段一致，字段名非空且不重复；非时间字段的非空值须可转换为有限数值，缺失值可保留。</li>
          <li>安全约束：仅允许导入平台白名单模块，禁止危险调用（如 <code>open</code>、<code>eval</code>、<code>exec</code>、<code>compile</code>、<code>input</code>）。</li>
        </ul>
        <p><b>不满足时会报错；代表性错误码：</b> <code>INVALID_SCRIPT</code>、<code>INVALID_PREPROCESSOR</code>、<code>UNSAFE_SCRIPT</code>、<code>PREPROCESS_FIT_FAILED</code>、<code>PREPROCESS_FIT_RETURN_INVALID</code>、<code>PREPROCESS_TRANSFORM_FAILED</code>、<code>PREPROCESS_RESULT_NOT_DATAFRAME</code>、<code>PREPROCESS_ROW_COUNT_INVALID</code>、<code>PREPROCESS_FIELDS_INVALID</code>、<code>PREPROCESS_FEATURES_EMPTY</code>、<code>PREPROCESS_VALUES_INVALID</code>。</p>
        <p><code>fit</code>/<code>transform</code> 是平台执行接口与输出契约，不是业务规则；具体业务算法可自行选择。</p>
      </ContractNote>
      <div className="choice-row">
        <label className={`skip-option${skip ? ' checked' : ''}`}>
          <input
            checked={skip}
            disabled={downstreamLocked}
            onChange={(event) => changeSelection(event.target.checked, event.target.checked ? null : selected)}
            type="checkbox"
          />
          <span><b>跳过预处理</b></span>
        </label>
        <label className="secondary-button action-button">
          上传预处理脚本
          <input aria-label="上传预处理脚本" accept=".py,text/x-python" disabled={downstreamLocked} hidden onChange={(event) => void uploadScript(event.target.files?.[0])} type="file" />
        </label>
      </div>
      {loading && <div className="loading-line"><span className="spinner" />读取 ENABLED 脚本…</div>}
      {!skip && !loading && (
        <div className="script-list">
          {scripts.map((script) => (
            <button
              className={`script-option${selected === script.id ? ' selected' : ''}`}
              disabled={downstreamLocked}
              key={script.id}
              onClick={() => changeSelection(false, script.id)}
              type="button"
            >
              <span><b>{script.name}</b><small>状态：{upperStatus(script.status) ?? '未知'} · ID：{script.id}</small></span>
              <i>{selected === script.id ? '已选择' : '选择'}</i>
            </button>
          ))}
          {!scripts.length && <div className="empty-state compact">暂无 ENABLED 预处理脚本</div>}
        </div>
      )}
      {downstreamLocked && <InfoBox tone="warning">固定 80/20 划分已绑定当前预处理任务，不能在本链路中更换预处理。</InfoBox>}
      {uploadingScript && <div className="loading-line"><span className="spinner" />上传脚本中…</div>}
      {task && (
        <div className="stage-result">
          <div className="stage-title"><span className={`status-dot ${taskStatus === 'FAILED' ? 'danger' : 'success'}`} />预处理{taskStatus === 'SKIPPED' ? '已跳过' : '任务结果'}</div>
          {task.preprocess_message && <InfoBox>{task.preprocess_message}</InfoBox>}
          <div className="table-wrap">
            <table>
              <thead><tr><th>项目</th><th>结果</th><th>状态</th></tr></thead>
              <tbody>
                <tr><td>任务状态</td><td>{displayValue(task.status)}</td><td>{taskStatus === 'FAILED' ? '失败' : taskStatus === 'SKIPPED' ? '已跳过' : displayValue(taskStatus)}</td></tr>
                <tr><td>当前阶段</td><td>{displayValue(task.current_stage ?? task.stage)}</td><td>{displayValue(task.progress_stage)}</td></tr>
                <tr><td>行数</td><td>{displayValue(task.input_row_count)} → {displayValue(task.output_row_count)}</td><td>{displayValue(task.data_source)}</td></tr>
                <tr><td>字段映射</td><td>{task.input_columns?.length ? `${task.input_columns.join('、')} → ${task.output_columns?.join('、') || '未知'}` : '未知'}</td><td>{task.preprocess_used ? '已使用' : '未使用'}</td></tr>
                <tr><td>质量检查</td><td>{qualityChecks.length ? qualityChecks.map((check) => `${check.code}：${check.message ?? '未提供'}`).join('；') : '未提供'}</td><td>{task.quality_checks ? task.quality_checks.valid ? '通过' : '失败' : '未提供'}</td></tr>
              </tbody>
            </table>
          </div>
          {task.error_message && <p className="error-text">{task.error_message}</p>}
        </div>
      )}
      <StepActions
        back={back}
        next={run}
        nextDisabled={restoring || running || uploadingScript || downstreamLocked || (!skip && !selected) || !datasetId}
        nextLabel={downstreamLocked ? '固定划分不可更改' : running ? '执行中…' : taskStatus === 'SUCCEEDED' || taskStatus === 'SKIPPED' ? '重新执行' : '执行并继续'}
      />
    </section>
  )
}

function SplitStep({
  back,
  complete,
  restoring,
}: {
  back: () => void
  complete: (split: DatasetSplitResult) => void
  restoring: boolean
}) {
  const datasetId = useAppStore((state) => state.workflow.datasetId)
  const taskId = useAppStore((state) => state.workflow.preprocessTaskId)
  const split = useAppStore((state) => state.workflow.split)
  const navigate = useNavigate()
  const [loading, setLoading] = useState(!split)
  const [error, setError] = useState<string | null>(null)

  const acceptSplit = useCallback((result: DatasetSplitResult) => {
    if (result.dataset_id !== datasetId || result.preprocessing_task_id !== taskId) {
      throw new Error('数据集划分与当前 datasetId/preprocessTaskId 不匹配，已阻断后续训练。')
    }
    complete(result)
  }, [complete, datasetId, taskId])

  useEffect(() => {
    if (split) {
      setLoading(false)
      return
    }
    if (!datasetId || !taskId || restoring) return
    let alive = true
    setLoading(true)
    apiClient.getDatasetSplit(datasetId)
      .then((result) => { if (alive) acceptSplit(result) })
      .catch((reason: unknown) => {
        if (alive && !isNotFound(reason)) setError(errorMessage(reason))
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [acceptSplit, datasetId, restoring, split, taskId])

  const create = async () => {
    if (!datasetId || !taskId) return
    setLoading(true)
    setError(null)
    try {
      acceptSplit(await apiClient.splitDataset(datasetId, taskId))
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 04 / FIXED SPLIT</p>
          <h2>数据集划分</h2>
        </div>
      </div>
      <ErrorBox message={error} />
      {loading && !split && <div className="loading-line"><span className="spinner" />读取划分结果…</div>}
      {split ? (
        <div className="split-result">
          <InfoBox tone="success">划分已完成</InfoBox>
          <div className="split-bars">
            <div><span style={{ width: '80%' }} /><b>训练集 80%</b><strong>{displayValue(split.train_row_count)} 行</strong></div>
            <div><span style={{ width: '20%' }} /><b>测试集 20%</b><strong>{displayValue(split.test_row_count)} 行</strong></div>
          </div>
          <div className="split-detail">
            <div><b>训练集时间范围</b><span>{displayValue(split.train_time_start)} ～ {displayValue(split.train_time_end)}</span></div>
            <div><b>测试集时间范围</b><span>{displayValue(split.test_time_start)} ～ {displayValue(split.test_time_end)}</span></div>
            <div><b>split_id</b><span>{displayValue(split.id)}</span></div>
            <div><b>规则</b><span>按时间顺序 80/20，不随机打散</span></div>
          </div>
        </div>
      ) : (
        <div className="empty-state">尚未创建数据集划分。点击下方按钮生成固定 80/20 结果。</div>
      )}
      <StepActions
        back={back}
        next={split ? () => navigate('/workflow/train') : create}
        nextDisabled={restoring || loading || !datasetId || !taskId}
        nextLabel={split ? '继续选择训练脚本' : '生成 80/20 划分'}
      />
    </section>
  )
}

function TrainingStep({
  back,
  complete,
  clearMissingJob,
  pollingEnabled,
  restoring,
}: {
  back: () => void
  complete: (job: TrainingJob) => void
  clearMissingJob: () => void
  pollingEnabled: boolean
  restoring: boolean
}) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const datasetId = useAppStore((state) => state.workflow.datasetId)
  const splitId = useAppStore((state) => state.workflow.splitId)
  const preprocessScriptId = useAppStore((state) => state.workflow.preprocessScriptId)
  const preprocessTaskId = useAppStore((state) => state.workflow.preprocessTaskId)
  const modelVersionId = useAppStore((state) => state.workflow.modelVersionId)
  const jobId = useAppStore((state) => state.workflow.trainingJobId)
  const job = useAppStore((state) => state.workflow.trainingJob)
  const selected = useAppStore((state) => state.workflow.trainScriptId)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const navigate = useNavigate()
  const [scripts, setScripts] = useState<ScriptContract[]>([])
  const [loading, setLoading] = useState(false)
  const [uploadingScript, setUploadingScript] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRun = useRef(0)
  const logCursor = useRef<string | undefined>(undefined)
  const logSince = useRef<string | undefined>(undefined)
  const knownLogs = useRef<TrainingLog[]>([])

  useEffect(() => {
    if (!modelType) {
      setScripts([])
      return
    }
    let alive = true
    apiClient.listScripts({ model_type: modelType, script_type: 'trainer', status: 'ENABLED' })
      .then((response) => { if (alive) setScripts(response.items.filter((script) => upperStatus(script.status) === 'ENABLED')) })
      .catch((reason) => { if (alive) setError(errorMessage(reason)) })
    return () => { alive = false }
  }, [modelType])

  const uploadScript = async (file?: File) => {
    if (!file || !modelType) return
    if (!file.name.toLowerCase().endsWith('.py')) {
      setError('仅支持 .py 文件')
      return
    }
    setUploadingScript(true)
    setError(null)
    try {
      const script = await apiClient.uploadScript({ file, name: file.name, script_type: 'trainer', supported_model_types: [modelType] })
      if (upperStatus(script.status) !== 'ENABLED') {
        setError('上传脚本未处于 ENABLED 状态，不能用于训练')
        return
      }
      setScripts((current) => [...current.filter((item) => item.id !== script.id), script])
      selectScript(script.id)
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setUploadingScript(false)
    }
  }

  const jobStatus = upperStatus(job?.status)
  useEffect(() => {
    if (!pollingEnabled || restoring || !jobId || !job || terminalTrainingStatuses.has(jobStatus ?? '')) return

    const runId = ++pollRun.current
    let cancelled = false
    let inFlight = false
    const poll = async () => {
      if (cancelled || inFlight) return
      inFlight = true
      try {
        const next = await apiClient.getTrainingJob(jobId)
        if (cancelled || pollRun.current !== runId) return
        if (next.id !== jobId || next.model_type !== modelType || next.dataset_id !== datasetId
          || next.preprocessing_task_id !== preprocessTaskId) {
          setError('训练任务与当前资源 ID 链不匹配，已停止轮询。')
          clearMissingJob()
          return
        }
        setError(null)
        knownLogs.current = mergeTrainingLogs(knownLogs.current, next.logs ?? [])
        complete({ ...next, logs: knownLogs.current })
      } catch (reason) {
        if (cancelled || pollRun.current !== runId) return
        setError(errorMessage(reason))
        if (isNotFound(reason)) clearMissingJob()
      } finally {
        inFlight = false
      }
    }

    void poll()
    const timer = window.setInterval(() => void poll(), 1500)
    return () => {
      cancelled = true
      pollRun.current += 1
      window.clearInterval(timer)
    }
  }, [clearMissingJob, complete, datasetId, jobId, jobStatus, modelType, pollingEnabled, preprocessTaskId, restoring])

  useEffect(() => {
    knownLogs.current = job?.logs ? [...job.logs] : []
    logCursor.current = undefined
    logSince.current = undefined
  }, [jobId])

  useEffect(() => {
    if (!pollingEnabled || restoring || !jobId || !job || terminalTrainingStatuses.has(jobStatus ?? '')) return
    let alive = true
    let inFlight = false
    const readLogs = async () => {
      if (!alive || inFlight) return
      inFlight = true
      try {
        const response = await apiClient.getTrainingJobLogs(jobId, {
          since: logSince.current,
          cursor: logCursor.current,
          limit: 100,
        })
        if (!alive || response.job_id !== jobId) return
        const incoming = response.items ?? []
        knownLogs.current = mergeTrainingLogs(knownLogs.current, incoming)
        logCursor.current = response.next_cursor ?? undefined
        const latest = incoming[incoming.length - 1]
        if (latest) logSince.current = latest.timestamp
        if (incoming.length) {
          const current = useAppStore.getState().workflow.trainingJob
          if (current?.id === jobId) complete({ ...current, logs: knownLogs.current })
        }
      } catch (reason) {
        if (alive) setError(errorMessage(reason))
      } finally {
        inFlight = false
      }
    }
    void readLogs()
    const timer = window.setInterval(() => void readLogs(), 1500)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [complete, jobId, jobStatus, pollingEnabled, restoring])

  const start = async () => {
    if (!modelType || !datasetId || !splitId || !preprocessTaskId || !selected) return
    setLoading(true)
    setError(null)
    try {
      const result = await apiClient.createTrainingJob({
        model_type: modelType,
        dataset_id: datasetId,
        preprocess_script_id: preprocessScriptId,
        preprocessing_task_id: preprocessTaskId,
        train_script_id: selected,
      })
      complete(result)
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setLoading(false)
    }
  }

  const retry = async () => {
    if (!jobId) return
    setLoading(true)
    setError(null)
    try {
      const next = await apiClient.retryTrainingJob(jobId)
      knownLogs.current = []
      logCursor.current = undefined
      logSince.current = undefined
      complete({ ...next, logs: [] })
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setLoading(false)
    }
  }

  const cancel = async () => {
    if (!jobId) return
    setLoading(true)
    setError(null)
    try {
      const next = await apiClient.cancelTrainingJob(jobId)
      knownLogs.current = mergeTrainingLogs(knownLogs.current, next.logs ?? [])
      complete({ ...next, logs: knownLogs.current })
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setLoading(false)
    }
  }

  const selectScript = (scriptId: string) => {
    setContext({
      trainScriptId: scriptId,
      trainingJobId: null,
      trainingJob: null,
      modelVersionId: null,
      evaluation: null,
      modelVersion: null,
    })
  }
  const terminal = terminalTrainingStatuses.has(jobStatus ?? '')
  const jobModelMatches = Boolean(job && hasOwn(job, 'model_version_id') && job.model_version_id === modelVersionId)
  const canEvaluate = jobStatus === 'SUCCEEDED' && Boolean(modelVersionId) && jobModelMatches
  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 05 / TRAINING</p>
          <h2>选择训练脚本并启动</h2>
        </div>
      </div>
      <ErrorBox message={error} />
      <ContractNote title="平台执行契约（不限制具体业务算法）">
        <ul>
          <li>脚本文件：仅接受 <code>.py</code>；使用 UTF-8；大小不超过 5 MiB；Python 语法必须有效。</li>
          <li>执行入口：必须定义 <code>train(X_train, y_train, X_test, y_test, config)</code>；返回对象必须有可调用的 <code>predict(X)</code>。</li>
          <li>输出契约：训练流程调用 <code>predict(X_test)</code> 时，结果须为一维、长度与测试集行数一致且可转换为有限数值。</li>
          <li>安全约束：仅允许导入平台白名单模块，禁止危险调用（如 <code>open</code>、<code>eval</code>、<code>exec</code>、<code>compile</code>、<code>input</code>）。</li>
        </ul>
        <p><b>不满足时会报错；代表性错误码：</b> <code>INVALID_SCRIPT</code>、<code>TRAIN_FUNCTION_INVALID</code>、<code>TRAIN_SIGNATURE_INVALID</code>、<code>MODEL_PREDICT_INVALID</code>、<code>UNSAFE_SCRIPT</code>、<code>TRAIN_EXECUTION_FAILED</code>、<code>PREDICTION_FAILED</code>、<code>PREDICTION_LENGTH_INVALID</code>、<code>PREDICTION_VALUES_INVALID</code>。</p>
        <p>算法实现可自行选择，只要满足上述执行接口和输出契约；本说明不是业务逻辑说明。</p>
      </ContractNote>
      {!job && (
        <>
        <label className="secondary-button action-button">
          上传训练脚本
          <input aria-label="上传训练脚本" accept=".py,text/x-python" hidden onChange={(event) => void uploadScript(event.target.files?.[0])} type="file" />
        </label>
        {uploadingScript && <div className="loading-line"><span className="spinner" />上传脚本中…</div>}
        <div className="script-list">
          {scripts.map((script) => (
            <button
              className={`script-option${selected === script.id ? ' selected' : ''}`}
              key={script.id}
              onClick={() => selectScript(script.id)}
              type="button"
            >
              <span><b>{script.name}</b><small>状态：{upperStatus(script.status) ?? '未知'} · ID：{script.id}</small></span>
              <i>{selected === script.id ? '✓' : '选择'}</i>
            </button>
          ))}
          {!scripts.length && <div className="empty-state compact">暂无 ENABLED 训练脚本</div>}
        </div>
        </>
      )}
      {job && (
        <div className="training-status">
          <div className="status-header">
            <span className={`status-badge ${jobStatus === 'FAILED' ? 'danger' : jobStatus === 'SUCCEEDED' ? 'success' : 'info'}`}>{job.status}</span>
            <b>{job.current_stage ?? job.progress_stage ?? '任务处理中'}</b>
            {!terminal && <span className="spinner" />}
          </div>
          <div className="stage-summary"><span>training_job_id：{displayValue(job.id)}</span><span>model_version_id：{displayValue(job.model_version_id)}</span><span>开始：{displayValue(job.started_at)}</span><span>完成：{displayValue(job.finished_at)}</span></div>
          {job.progress && <div className="stage-summary"><span>进度阶段：{displayValue(job.progress.stage)}</span><span>进度状态：{displayValue(job.progress.status)}</span><span>{displayValue(job.progress.message)}</span></div>}
          {job.stages?.length ? <div className="training-stages">{job.stages.map((stage, index) => <span className={stage.status === 'succeeded' ? 'done' : ''} key={`${stage.stage}-${index}`}>{displayValue(stage.stage)} · {displayValue(stage.status)}</span>)}</div> : null}
          <LogList logs={job.logs} />
          {job.error_message && <p className="error-text">{job.error_message}</p>}
          {jobStatus === 'FAILED' && <InfoBox tone="warning">本次训练失败，不会影响生产模型。修复脚本或配置后可以重试。</InfoBox>}
          {!terminal && <div className="retry-row"><Button kind="danger" disabled={loading || restoring} onClick={cancel}>取消训练</Button></div>}
          {jobStatus === 'SUCCEEDED' && !canEvaluate && (
            <InfoBox tone="warning">训练任务已 SUCCEEDED，但响应缺少 model_version_id；评估与发布已阻断，请重新检查训练结果。</InfoBox>
          )}
        </div>
      )}
      <StepActions
        back={back}
        next={canEvaluate ? () => navigate('/workflow/evaluate') : start}
        nextDisabled={restoring || loading || uploadingScript || !selected || !datasetId || !splitId || !preprocessTaskId
          || (job !== null && !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(jobStatus ?? ''))
          || (jobStatus === 'SUCCEEDED' && !canEvaluate)}
        nextLabel={loading ? '提交中…' : jobStatus === 'SUCCEEDED' ? '查看评估结果' : jobStatus === 'FAILED' || jobStatus === 'CANCELLED' ? '重新启动' : job ? '训练进行中…' : '启动训练'}
      />
      {jobStatus === 'FAILED' && (
        <div className="retry-row"><Button kind="secondary" disabled={loading || restoring} onClick={retry}>失败重试</Button></div>
      )}
    </section>
  )
}

function EvaluationStep({
  back,
  publish,
  restoring,
}: {
  back: () => void
  publish: () => void
  restoring: boolean
}) {
  const trainingJobId = useAppStore((state) => state.workflow.trainingJobId)
  const modelVersionId = useAppStore((state) => state.workflow.modelVersionId)
  const evaluation = useAppStore((state) => state.workflow.evaluation)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const [attempt, setAttempt] = useState(0)
  const [loading, setLoading] = useState(!evaluation)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (restoring || !trainingJobId || !modelVersionId || evaluation) return
    let alive = true
    setLoading(true)
    apiClient.getTrainingJobEvaluation(trainingJobId)
      .then((result) => {
        if (!alive) return
        if (result.job_id !== trainingJobId || result.model_version_id !== modelVersionId) {
          throw new Error('评估结果与当前 trainingJobId/modelVersionId 不匹配，已阻断发布。')
        }
        setContext({ evaluation: result })
      })
      .catch((reason) => { if (alive) setError(errorMessage(reason)) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [attempt, evaluation, modelVersionId, restoring, setContext, trainingJobId])

  const retry = () => {
    setError(null)
    setLoading(true)
    setContext({ evaluation: null })
    setAttempt((value) => value + 1)
  }

  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 06 / EVALUATION</p>
          <h2>评估结果与模型对比</h2>
        </div>
      </div>
      <ErrorBox message={error} onRetry={retry} />
      {(loading || restoring) && <div className="loading-line"><span className="spinner" />读取评估结果…</div>}
      {evaluation && (
        <>
          <div className="evaluation-meta" aria-label="评估结果摘要">
            <span>模型版本：{displayValue(evaluation.model_version_id)}</span>
            <span>样本数：{displayValue(evaluation.metrics.sample_count)}</span>
            <span>MAPE 有效：{displayValue(evaluation.metrics.mape_valid_count)}</span>
            <span>MAPE 排除：{displayValue(evaluation.metrics.mape_excluded_count)}</span>
          </div>
          <EvaluationDashboard evaluation={evaluation} />
        </>
      )}
      {!loading && !restoring && !evaluation && <div className="empty-state">暂无评估结果，请确认训练任务已成功完成。</div>}
      <StepActions back={back} next={publish} nextDisabled={restoring || !evaluation} nextLabel="保存与发布" />
    </section>
  )
}

function PublishStep({ back, restoring }: { back: () => void; restoring: boolean }) {
  const workflow = useAppStore((state) => state.workflow)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const [currentModel, setCurrentModel] = useState<ModelVersionSummary | null>(null)
  const [currentLoading, setCurrentLoading] = useState(false)
  const [currentError, setCurrentError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [savedLocally, setSavedLocally] = useState(false)
  const [publishedLocally, setPublishedLocally] = useState(false)
  const [publishOpen, setPublishOpen] = useState(false)
  const [reason, setReason] = useState('')
  const [error, setError] = useState<string | null>(null)
  const lifecycleStatus = upperStatus(workflow.modelVersion?.status)
  const saved = savedLocally || lifecycleStatus === 'READY' || lifecycleStatus === 'PUBLISHED'
  const published = publishedLocally || lifecycleStatus === 'PUBLISHED'
  const candidate = workflow.modelVersion
  const candidateHealth = upperStatus(candidate?.health_status) === 'HEALTHY' && candidate?.is_abnormal !== true
    ? 'HEALTHY'
    : upperStatus(candidate?.health_status) === 'ABNORMAL' || candidate?.is_abnormal === true ? 'ABNORMAL' : 'UNKNOWN'

  useEffect(() => {
    if (!workflow.modelType) return
    let alive = true
    setCurrentLoading(true)
    setCurrentError(null)
    apiClient.listModels({ model_type: workflow.modelType })
      .then((models) => {
        if (!alive) return
        setCurrentModel(models.find((model) => model.is_current === true) ?? null)
      })
      .catch((reason) => { if (alive) setCurrentError(errorMessage(reason)) })
      .finally(() => { if (alive) setCurrentLoading(false) })
    return () => { alive = false }
  }, [workflow.modelType])

  const save = async () => {
    if (!workflow.modelVersionId || !workflow.modelType) return
    setSaving(true)
    setError(null)
    try {
      const model = await apiClient.saveModel(workflow.modelVersionId, {
        model_type: workflow.modelType,
        status: 'READY',
        training_job_id: workflow.trainingJobId ?? undefined,
        train_script_id: workflow.trainScriptId ?? undefined,
        preprocess_script_id: workflow.preprocessScriptId,
        preprocess_used: Boolean(workflow.preprocessScriptId),
        time_column: workflow.dataset?.time_column,
        feature_columns: workflow.dataset?.feature_columns,
        target_column: workflow.dataset?.target_column,
        // The backend stores the complete evaluation envelope in metrics;
        // sending only the flat metric set would erase chart/comparison data.
        metrics: (workflow.evaluation ?? {}) as unknown as JsonRecord,
      })
      setContext({ modelVersion: { ...model, status: 'READY' }, modelVersionId: model.id })
      setSavedLocally(true)
      window.dispatchEvent(new CustomEvent('model-registry-updated'))
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setSaving(false)
    }
  }

  const openPublish = () => {
    setError(null)
    setReason('')
    setPublishOpen(true)
  }

  const publish = async () => {
    if (!workflow.modelVersionId) return
    const trimmedReason = reason.trim()
    if (!trimmedReason) {
      setError('发布原因不能为空')
      return
    }
    if (candidateHealth !== 'HEALTHY') {
      setError('发布需要 READY 且 HEALTHY 候选版本')
      return
    }
    setPublishing(true)
    setError(null)
    try {
      const result = await apiClient.publishModel(workflow.modelVersionId, {
        confirmed: true,
        reason: trimmedReason,
        idempotency_key: createIdempotencyKey('publish'),
      })
      setContext({ modelVersion: { ...result.model, status: 'PUBLISHED' }, modelVersionId: workflow.modelVersionId })
      setPublishedLocally(true)
      setPublishOpen(false)
      window.dispatchEvent(new CustomEvent('model-registry-updated'))
    } catch (reasonValue) {
      setError(errorMessage(reasonValue))
    } finally {
      setPublishing(false)
    }
  }

  const featureColumns = candidate?.feature_columns ?? workflow.dataset?.feature_columns ?? []
  return (
    <section className="workflow-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">STEP 07 / PUBLISH</p>
          <h2>保存与发布</h2>
        </div>
      </div>
      <ErrorBox message={error} />
      {restoring && <div className="loading-line"><span className="spinner" />读取候选模型版本…</div>}
      {currentLoading && <div className="loading-line"><span className="spinner" />读取当前生产版本…</div>}
      {currentError && <ErrorBox message={currentError} />}
      {candidate && workflow.modelVersionId ? (
        <div className="publish-card">
          <div className="publish-model">
            <span className="model-symbol">◆</span>
            <div><b>候选版本</b><small>{displayValue(candidate.version)} · {displayValue(workflow.modelVersionId)}</small></div>
            <span className={`status-badge ${saved ? 'success' : 'info'}`}>{saved ? 'READY' : displayValue(candidate.status)}</span>
            <span className="status-badge">{candidateHealth}</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>项目</th><th>当前值</th><th>状态</th></tr></thead>
              <tbody>
                <tr><td>模型版本</td><td>{displayValue(candidate.version)} / {displayValue(workflow.modelVersionId)}</td><td>{saved ? 'READY' : displayValue(candidate.status)}</td></tr>
                <tr><td>训练任务</td><td>{displayValue(workflow.trainingJobId)}</td><td>{displayValue(workflow.trainingJob?.status)}</td></tr>
                <tr><td>当前生产版本</td><td>{displayValue(currentModel?.version)}</td><td>{currentModel ? displayValue(currentModel.status) : '未知'}</td></tr>
                <tr><td>时间列 / 目标列</td><td>{displayValue(candidate.time_column ?? workflow.dataset?.time_column)} / {displayValue(candidate.target_column ?? workflow.dataset?.target_column)}</td><td>—</td></tr>
                <tr><td>特征列</td><td>{featureColumns.length ? featureColumns.join('、') : '未知'}</td><td>{featureColumns.length ? `${featureColumns.length} 列` : '未知'}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="publish-actions">
            <Button disabled={restoring || saving || saved} kind="secondary" onClick={save}>{saving ? '保存中…' : saved ? 'READY' : '保存候选版本'}</Button>
            <Button disabled={restoring || !saved || candidateHealth !== 'HEALTHY' || publishing || published} onClick={openPublish}>{publishing ? '发布中…' : published ? 'PUBLISHED' : '发布'}</Button>
          </div>
        </div>
      ) : !restoring ? <div className="empty-state">暂无候选模型版本</div> : null}
      {published && <InfoBox tone="success">发布成功</InfoBox>}
      <StepActions back={back} />
      {publishOpen && candidate && (
        <div className="confirm-overlay" role="presentation">
          <div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="publish-dialog-title">
            <h2 id="publish-dialog-title">确认发布</h2>
            <p>候选版本：<b>{displayValue(candidate.version)}</b></p>
            <p>当前生产版本：<b>{displayValue(currentModel?.version)}</b></p>
            <label htmlFor="publish-reason">发布原因（必填）</label>
            <textarea id="publish-reason" aria-label="发布原因" value={reason} onChange={(event) => setReason(event.target.value)} />
            <div className="publish-actions"><Button kind="secondary" onClick={() => setPublishOpen(false)}>取消</Button><Button disabled={publishing || !reason.trim()} onClick={() => void publish()}>{publishing ? '提交中…' : '确认发布'}</Button></div>
          </div>
        </div>
      )}
    </section>
  )
}

type WorkflowGate = { allowed: boolean; step: WorkflowStepId; reason: string }

type RestoreState = {
  key: string | null
  loading: boolean
  error: string | null
  blockedStep: WorkflowStepId | null
}

type WorkflowSnapshot = Pick<WorkflowDraft,
  | 'modelType'
  | 'datasetId'
  | 'preprocessTaskId'
  | 'splitId'
  | 'trainingJobId'
  | 'modelVersionId'
>

type ResourceResult<T> = { value: T | null; error: unknown }

const emptyResult = <T,>(): ResourceResult<T> => ({ value: null, error: null })

async function readResource<T>(load: () => Promise<T>): Promise<ResourceResult<T>> {
  try {
    return { value: await load(), error: null }
  } catch (error) {
    return { value: null, error }
  }
}

function workflowSnapshot(workflow: WorkflowDraft): WorkflowSnapshot {
  return {
    modelType: workflow.modelType,
    datasetId: workflow.datasetId,
    preprocessTaskId: workflow.preprocessTaskId,
    splitId: workflow.splitId,
    trainingJobId: workflow.trainingJobId,
    modelVersionId: workflow.modelVersionId,
  }
}

function snapshotKey(snapshot: WorkflowSnapshot): string {
  return JSON.stringify([
    snapshot.modelType,
    snapshot.datasetId,
    snapshot.preprocessTaskId,
    snapshot.splitId,
    snapshot.trainingJobId,
    snapshot.modelVersionId,
  ])
}

function sameSnapshot(left: WorkflowSnapshot, right: WorkflowSnapshot): boolean {
  return snapshotKey(left) === snapshotKey(right)
}

function clearFromDataset(): Partial<WorkflowDraft> {
  return {
    datasetId: null,
    dataset: null,
    preprocessScriptId: null,
    preprocessTaskId: null,
    preprocessTask: null,
    splitId: null,
    split: null,
    trainScriptId: null,
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }
}

function clearFromPreprocessTask(): Partial<WorkflowDraft> {
  return {
    preprocessTaskId: null,
    preprocessTask: null,
    splitId: null,
    split: null,
    trainScriptId: null,
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }
}

function clearFromSplit(): Partial<WorkflowDraft> {
  return {
    splitId: null,
    split: null,
    trainScriptId: null,
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }
}

function clearFromTrainingJob(): Partial<WorkflowDraft> {
  return {
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }
}

function workflowGate(
  workflow: WorkflowDraft,
  step: WorkflowStepId,
  restore: Pick<RestoreState, 'error' | 'blockedStep'>,
): WorkflowGate {
  const requestedIndex = stepIndexes[step]
  const blockedStep = restore.blockedStep
  if (blockedStep && requestedIndex > stepIndexes[blockedStep]) {
    return {
      allowed: false,
      step: blockedStep,
      reason: restore.error ?? '资源恢复失败，无法确认后续步骤的前置条件。',
    }
  }
  if (step === 'model-type') return { allowed: true, step, reason: '' }
  if (!workflow.modelType) {
    return { allowed: false, step: 'model-type', reason: '缺少模型类型，请先选择模型类型。' }
  }
  if (requestedIndex >= stepIndexes.preprocess && !workflow.datasetId) {
    return { allowed: false, step: 'upload', reason: '缺少 datasetId，请先上传并确认数据集。' }
  }
  if (requestedIndex >= stepIndexes.split) {
    if (!workflow.preprocessTaskId) {
      return { allowed: false, step: 'preprocess', reason: '缺少 preprocessTaskId，请先执行预处理；跳过预处理也必须创建 SKIPPED 任务。' }
    }
    const task = workflow.preprocessTask
    if (!task || task.id !== workflow.preprocessTaskId || task.dataset_id !== workflow.datasetId
      || task.model_type !== workflow.modelType) {
      return { allowed: false, step: 'preprocess', reason: '无法确认 preprocessTaskId 对应当前模型和数据集，请重新执行预处理。' }
    }
    if (!['SUCCEEDED', 'SKIPPED'].includes(upperStatus(task.status) ?? '')) {
      return { allowed: false, step: 'preprocess', reason: '预处理任务尚未 SUCCEEDED 或 SKIPPED，不能继续划分数据集。' }
    }
  }
  if (requestedIndex >= stepIndexes.train && !workflow.splitId) {
    return { allowed: false, step: 'split', reason: '缺少 splitId，请先完成固定 80/20 数据集划分。' }
  }
  if (requestedIndex >= stepIndexes.evaluate) {
    const job = workflow.trainingJob
    if (!workflow.trainingJobId || !job || job.id !== workflow.trainingJobId) {
      return { allowed: false, step: 'train', reason: '缺少 trainingJobId，请先提交训练任务。' }
    }
    if (job.model_type !== workflow.modelType || job.dataset_id !== workflow.datasetId
      || job.preprocessing_task_id !== workflow.preprocessTaskId) {
      return { allowed: false, step: 'train', reason: '训练任务与当前资源 ID 链不匹配，请重新提交训练任务。' }
    }
    if (upperStatus(job.status) !== 'SUCCEEDED') {
      return { allowed: false, step: 'train', reason: '训练任务必须为 SUCCEEDED 后才能进入评估或发布。' }
    }
    const jobModelId = hasOwn(job, 'model_version_id') ? job.model_version_id : null
    if (!hasOwn(job, 'model_version_id') || !workflow.modelVersionId || !jobModelId || jobModelId !== workflow.modelVersionId) {
      return { allowed: false, step: 'train', reason: '训练已成功但缺少 model_version_id，已阻断评估/发布；请重新检查训练结果。' }
    }
  }
  if (step === 'publish') {
    const evaluation = workflow.evaluation
    if (!evaluation || evaluation.job_id !== workflow.trainingJobId
      || evaluation.model_version_id !== workflow.modelVersionId) {
      return { allowed: false, step: 'evaluate', reason: '缺少与当前训练任务匹配的评估结果，请先完成模型评估后才能保存与发布。' }
    }
  }
  return { allowed: true, step, reason: '' }
}

export function WorkflowPage() {
  const navigate = useNavigate()
  const { stepId } = useParams<{ stepId?: string }>()
  const [searchParams] = useSearchParams()
  const workflow = useAppStore((state) => state.workflow)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const setModelType = useAppStore((state) => state.setModelType)
  const startNewWorkflow = useAppStore((state) => state.startNewWorkflow)
  const resetWorkflow = useAppStore((state) => state.resetWorkflow)
  const [guardNotice, setGuardNotice] = useState<{ message: string; target: WorkflowStepId } | null>(null)
  const [restoreAttempt, setRestoreAttempt] = useState(0)
  const [restoreState, setRestoreState] = useState<RestoreState>({
    key: null,
    loading: false,
    error: null,
    blockedStep: null,
  })
  const restoreRun = useRef(0)

  const queryModel = useMemo(() => {
    const candidate = searchParams.get('model_type') ?? searchParams.get('model') ?? searchParams.get('modelType')
    return candidate && MODEL_TYPE_CODES.includes(candidate as ModelTypeCode)
      ? candidate as ModelTypeCode
      : null
  }, [searchParams])
  const queryDatasetId = searchParams.get('dataset_id')
  const queryTaskId = searchParams.get('preprocessing_task_id')
  const querySplitId = searchParams.get('split_id')
  const queryJobId = searchParams.get('training_job_id')
  const queryStartsNew = searchParams.get('new') === '1' || Boolean(queryModel)
  const queryContextActive = Boolean(queryStartsNew || queryDatasetId || queryTaskId || querySplitId || queryJobId)
  const queryContextKey = useMemo(() => JSON.stringify([
    queryModel, queryDatasetId, queryTaskId, querySplitId, queryJobId, queryStartsNew,
  ]), [queryDatasetId, queryJobId, queryModel, querySplitId, queryStartsNew, queryTaskId])
  const [queryRestore, setQueryRestore] = useState<{ key: string | null; loading: boolean; error: string | null }>({
    key: null,
    loading: false,
    error: null,
  })
  const queryHydrating = queryContextActive
    && (queryRestore.key !== queryContextKey || queryRestore.loading)

  useEffect(() => {
    if (!queryContextActive) {
      if (queryRestore.key !== queryContextKey || queryRestore.error) {
        setQueryRestore({ key: queryContextKey, loading: false, error: null })
      }
      return
    }
    if (queryRestore.key === queryContextKey) return

    const runId = ++restoreRun.current
    let alive = true
    setGuardNotice(null)
    setQueryRestore({ key: queryContextKey, loading: true, error: null })

    void (async () => {
      try {
        if (queryStartsNew && !queryDatasetId && !queryTaskId && !querySplitId && !queryJobId) {
          startNewWorkflow(queryModel)
        } else {
          resetWorkflow()
          let patch: Partial<WorkflowDraft>
          if (queryJobId) {
            const job = await apiClient.getTrainingJob(queryJobId)
            const dataset = await apiClient.getDataset(job.dataset_id)
            const task = job.preprocessing_task_id
              ? await apiClient.getPreprocessingTask(job.preprocessing_task_id)
              : null
            const split = await apiClient.getDatasetSplit(job.dataset_id)
            if (queryModel && queryModel !== job.model_type) throw new Error('训练任务与 model_type 不匹配。')
            if (task && (task.dataset_id !== dataset.id || task.model_type !== job.model_type)) {
              throw new Error('训练任务与预处理任务的资源 ID 链不匹配。')
            }
            if (split.preprocessing_task_id !== (task?.id ?? null)) {
              throw new Error('训练任务与数据集划分的 preprocessing_task_id 不匹配。')
            }
            patch = {
              modelType: job.model_type,
              datasetId: dataset.id,
              dataset,
              preprocessScriptId: task?.preprocess_script_id ?? null,
              preprocessTaskId: task?.id ?? null,
              preprocessTask: task,
              splitId: split.id,
              split,
              trainScriptId: job.train_script_id,
              trainingJobId: job.id,
              trainingJob: job,
              modelVersionId: job.model_version_id,
              currentStep: 'train',
            }
          } else if (queryTaskId) {
            const task = await apiClient.getPreprocessingTask(queryTaskId)
            const dataset = await apiClient.getDataset(task.dataset_id)
            if (queryModel && queryModel !== task.model_type) throw new Error('预处理任务与 model_type 不匹配。')
            patch = {
              modelType: task.model_type,
              datasetId: dataset.id,
              dataset,
              preprocessScriptId: task.preprocess_script_id,
              preprocessTaskId: task.id,
              preprocessTask: task,
              currentStep: 'preprocess',
            }
          } else if (querySplitId) {
            if (!queryDatasetId) throw new Error('split 深链缺少 dataset_id，无法按数据集读取固定划分。')
            const dataset = await apiClient.getDataset(queryDatasetId)
            const split = await apiClient.getDatasetSplit(queryDatasetId)
            if (split.id !== querySplitId || split.dataset_id !== dataset.id) throw new Error('数据集划分与 split_id 不匹配。')
            const task = split.preprocessing_task_id
              ? await apiClient.getPreprocessingTask(split.preprocessing_task_id)
              : null
            const modelType = queryModel ?? task?.model_type ?? null
            if (!modelType) throw new Error('split 深链缺少可恢复的 model_type。')
            if (task && task.dataset_id !== dataset.id) throw new Error('数据集划分与预处理任务不匹配。')
            patch = {
              modelType,
              datasetId: dataset.id,
              dataset,
              preprocessScriptId: task?.preprocess_script_id ?? null,
              preprocessTaskId: task?.id ?? null,
              preprocessTask: task,
              splitId: split.id,
              split,
              currentStep: 'split',
            }
          } else if (queryDatasetId) {
            const dataset = await apiClient.getDataset(queryDatasetId)
            patch = {
              modelType: queryModel,
              datasetId: dataset.id,
              dataset,
              currentStep: queryModel ? 'upload' : 'model-type',
            }
          } else {
            throw new Error('缺少可恢复的资源 ID。')
          }
          if (!alive || restoreRun.current !== runId) return
          setContext(patch)
        }
        if (alive && restoreRun.current === runId) {
          setQueryRestore({ key: queryContextKey, loading: false, error: null })
        }
      } catch (reason) {
        if (alive && restoreRun.current === runId) {
          setQueryRestore({ key: queryContextKey, loading: false, error: errorMessage(reason) })
        }
      }
    })()

    return () => { alive = false }
  }, [queryContextActive, queryContextKey, queryDatasetId, queryJobId, queryModel,
    querySplitId, queryStartsNew, queryTaskId, resetWorkflow, restoreAttempt, setContext,
    startNewWorkflow])

  const snapshot = workflowSnapshot(workflow)
  const hydrationKey = snapshotKey(snapshot)

  useEffect(() => {
    if (queryHydrating) return
    if (!snapshot.modelType) {
      restoreRun.current += 1
      setRestoreState({ key: hydrationKey, loading: false, error: null, blockedStep: null })
      return
    }

    const runId = ++restoreRun.current
    let alive = true
    const isCurrent = () => alive
      && restoreRun.current === runId
      && sameSnapshot(workflowSnapshot(useAppStore.getState().workflow), snapshot)

    setRestoreState({ key: hydrationKey, loading: true, error: null, blockedStep: null })

    void (async () => {
      const [datasetResult, taskResult, splitResult, jobResult] = await Promise.all([
        snapshot.datasetId
          ? readResource(() => apiClient.getDataset(snapshot.datasetId!))
          : Promise.resolve(emptyResult<DatasetUploadResult>()),
        snapshot.preprocessTaskId
          ? readResource(() => apiClient.getPreprocessingTask(snapshot.preprocessTaskId!))
          : Promise.resolve(emptyResult<PreprocessTask>()),
        snapshot.datasetId && snapshot.splitId
          ? readResource(() => apiClient.getDatasetSplit(snapshot.datasetId!))
          : Promise.resolve(emptyResult<DatasetSplitResult>()),
        snapshot.trainingJobId
          ? readResource(() => apiClient.getTrainingJob(snapshot.trainingJobId!))
          : Promise.resolve(emptyResult<TrainingJob>()),
      ])
      if (!isCurrent()) return

      const patch: Partial<WorkflowDraft> = {}
      let firstError: string | null = null
      let blockedStep: WorkflowStepId | null = null
      const fail = (step: WorkflowStepId, label: string, reason: unknown) => {
        if (!firstError) firstError = `${label}：${errorMessage(reason)}`
        if (!blockedStep || stepIndexes[step] < stepIndexes[blockedStep]) blockedStep = step
      }
      const missing = (step: WorkflowStepId, label: string) => {
        if (!firstError) firstError = `${label}已不存在（404），已清理该资源及全部下游上下文。`
        if (!blockedStep || stepIndexes[step] < stepIndexes[blockedStep]) blockedStep = step
      }

      let datasetValid = !snapshot.datasetId
      if (snapshot.datasetId) {
        const dataset = datasetResult.value
        const responseId = dataset?.id ?? dataset?.dataset_id
        if (datasetResult.error) {
          if (isNotFound(datasetResult.error)) {
            Object.assign(patch, clearFromDataset())
            missing('upload', '数据集')
          } else {
            Object.assign(patch, {
              datasetId: snapshot.datasetId,
              dataset: null,
              preprocessTaskId: snapshot.preprocessTaskId,
              preprocessTask: null,
              splitId: snapshot.splitId,
              split: null,
              trainingJobId: snapshot.trainingJobId,
              trainingJob: null,
              modelVersionId: snapshot.modelVersionId,
              modelVersion: null,
              evaluation: null,
            })
            fail('upload', '恢复数据集失败', datasetResult.error)
          }
        } else if (!dataset || responseId !== snapshot.datasetId) {
          Object.assign(patch, clearFromDataset())
          fail('upload', '数据集响应与 datasetId 不匹配', new Error('资源响应无效'))
        } else {
          datasetValid = true
          Object.assign(patch, { datasetId: snapshot.datasetId, dataset })
        }
      } else {
        Object.assign(patch, { dataset: null })
        if (snapshot.preprocessTaskId || snapshot.splitId || snapshot.trainingJobId || snapshot.modelVersionId) {
          Object.assign(patch, clearFromDataset())
        }
      }

      const hasResourcesAfterTask = Boolean(snapshot.splitId || snapshot.trainingJobId || snapshot.modelVersionId)
      let taskValid = datasetValid && !snapshot.preprocessTaskId && !hasResourcesAfterTask
      if (datasetValid && snapshot.preprocessTaskId) {
        const task = taskResult.value
        if (taskResult.error) {
          if (isNotFound(taskResult.error)) {
            Object.assign(patch, clearFromPreprocessTask())
            missing('preprocess', '预处理任务')
          } else {
            Object.assign(patch, {
              preprocessTaskId: snapshot.preprocessTaskId,
              preprocessTask: null,
              splitId: snapshot.splitId,
              split: null,
              trainingJobId: snapshot.trainingJobId,
              trainingJob: null,
              modelVersionId: snapshot.modelVersionId,
              modelVersion: null,
              evaluation: null,
            })
            fail('preprocess', '恢复预处理任务失败', taskResult.error)
          }
        } else if (!task || task.id !== snapshot.preprocessTaskId
          || task.dataset_id !== snapshot.datasetId || task.model_type !== snapshot.modelType) {
          Object.assign(patch, clearFromPreprocessTask())
          fail('preprocess', '预处理任务与当前资源 ID 链不匹配', new Error('资源响应无效'))
        } else {
          taskValid = true
          Object.assign(patch, {
            preprocessTaskId: snapshot.preprocessTaskId,
            preprocessTask: task,
            preprocessScriptId: task.preprocess_script_id,
          })
        }
      } else if (!snapshot.preprocessTaskId) {
        Object.assign(patch, { preprocessTask: null })
        if (snapshot.splitId || snapshot.trainingJobId || snapshot.modelVersionId) {
          Object.assign(patch, clearFromPreprocessTask())
        }
      }

      const hasResourcesAfterSplit = Boolean(snapshot.trainingJobId || snapshot.modelVersionId)
      let splitValid = taskValid && !snapshot.splitId && !hasResourcesAfterSplit
      if (taskValid && snapshot.splitId) {
        const split = splitResult.value
        if (splitResult.error) {
          if (isNotFound(splitResult.error)) {
            Object.assign(patch, clearFromSplit())
            missing('split', '数据集划分')
          } else {
            Object.assign(patch, {
              splitId: snapshot.splitId,
              split: null,
              trainingJobId: snapshot.trainingJobId,
              trainingJob: null,
              modelVersionId: snapshot.modelVersionId,
              modelVersion: null,
              evaluation: null,
            })
            fail('split', '恢复数据集划分失败', splitResult.error)
          }
        } else if (!split || split.id !== snapshot.splitId || split.dataset_id !== snapshot.datasetId
          || split.preprocessing_task_id !== snapshot.preprocessTaskId) {
          Object.assign(patch, clearFromSplit())
          fail('split', '数据集划分与当前资源 ID 链不匹配', new Error('资源响应无效'))
        } else {
          splitValid = true
          Object.assign(patch, { splitId: snapshot.splitId, split })
        }
      } else if (!snapshot.splitId) {
        Object.assign(patch, { split: null })
        if (snapshot.trainingJobId || snapshot.modelVersionId) Object.assign(patch, clearFromSplit())
      }

      let jobValid = splitValid && !snapshot.trainingJobId
      let restoredJob: TrainingJob | null = null
      let effectiveModelId: string | null = null
      if (splitValid && snapshot.trainingJobId) {
        const job = jobResult.value
        if (jobResult.error) {
          if (isNotFound(jobResult.error)) {
            Object.assign(patch, clearFromTrainingJob())
            missing('train', '训练任务')
          } else {
            Object.assign(patch, {
              trainingJobId: snapshot.trainingJobId,
              trainingJob: null,
              modelVersionId: snapshot.modelVersionId,
              modelVersion: null,
              evaluation: null,
            })
            fail('train', '恢复训练任务失败', jobResult.error)
          }
        } else if (!job || job.id !== snapshot.trainingJobId || job.model_type !== snapshot.modelType
          || job.dataset_id !== snapshot.datasetId || job.preprocessing_task_id !== snapshot.preprocessTaskId) {
          Object.assign(patch, clearFromTrainingJob())
          fail('train', '训练任务与当前资源 ID 链不匹配', new Error('资源响应无效'))
        } else {
          jobValid = true
          restoredJob = job
          effectiveModelId = hasOwn(job, 'model_version_id') ? job.model_version_id : null
          Object.assign(patch, {
            trainingJobId: snapshot.trainingJobId,
            trainingJob: job,
            trainScriptId: job.train_script_id,
            modelVersionId: effectiveModelId,
          })
          if (effectiveModelId !== snapshot.modelVersionId) {
            Object.assign(patch, { modelVersion: null, evaluation: null })
          }
        }
      } else if (!snapshot.trainingJobId) {
        effectiveModelId = null
        Object.assign(patch, { trainingJob: null, modelVersionId: null, modelVersion: null, evaluation: null })
      }

      let modelResult: ResourceResult<ModelVersionDetail> = emptyResult()
      let evaluationResult: ResourceResult<ModelEvaluation> = emptyResult()
      if (jobValid && effectiveModelId) {
        ;[modelResult, evaluationResult] = await Promise.all([
          readResource(() => apiClient.getModel(effectiveModelId!)),
          restoredJob && upperStatus(restoredJob.status) === 'SUCCEEDED'
            ? readResource(() => apiClient.getTrainingJobEvaluation(restoredJob.id))
            : Promise.resolve(emptyResult<ModelEvaluation>()),
        ])
      }
      if (!isCurrent()) return

      if (jobValid && effectiveModelId) {
        const model = modelResult.value
        let modelValid = false
        if (modelResult.error) {
          if (isNotFound(modelResult.error)) {
            Object.assign(patch, { modelVersionId: null, modelVersion: null, evaluation: null })
            missing('train', '模型版本')
          } else {
            Object.assign(patch, { modelVersionId: effectiveModelId, modelVersion: null, evaluation: null })
            fail('train', '恢复模型版本失败', modelResult.error)
          }
        } else if (!model || model.id !== effectiveModelId || model.model_type !== snapshot.modelType
          || (model.training_job_id && model.training_job_id !== snapshot.trainingJobId)) {
          Object.assign(patch, { modelVersionId: null, modelVersion: null, evaluation: null })
          fail('train', '模型版本与当前资源 ID 链不匹配', new Error('资源响应无效'))
        } else {
          modelValid = true
          Object.assign(patch, { modelVersionId: effectiveModelId, modelVersion: model })
        }

        if (modelValid && restoredJob && upperStatus(restoredJob.status) === 'SUCCEEDED') {
          const evaluation = evaluationResult.value
          if (evaluationResult.error) {
            Object.assign(patch, { evaluation: null })
            if (isNotFound(evaluationResult.error)) {
              missing('evaluate', '评估结果')
            } else {
              fail('evaluate', '恢复评估结果失败', evaluationResult.error)
            }
          } else if (!evaluation || evaluation.job_id !== restoredJob.id
            || evaluation.model_version_id !== effectiveModelId) {
            Object.assign(patch, { evaluation: null })
            fail('evaluate', '评估结果与当前资源 ID 链不匹配', new Error('资源响应无效'))
          } else {
            Object.assign(patch, { evaluation })
          }
        } else {
          Object.assign(patch, { evaluation: null })
        }
      } else if (restoredJob && upperStatus(restoredJob.status) === 'SUCCEEDED' && !effectiveModelId) {
        Object.assign(patch, { modelVersionId: null, modelVersion: null, evaluation: null })
      }

      if (!isCurrent()) return
      if (Object.keys(patch).length) setContext(patch)
      if (!isCurrent()) return
      setRestoreState({ key: hydrationKey, loading: false, error: firstError, blockedStep })
    })()

    return () => { alive = false }
  }, [hydrationKey, queryHydrating, restoreAttempt, setContext])

  const hydrating = queryHydrating
    || Boolean(workflow.modelType && (restoreState.key !== hydrationKey || restoreState.loading))
  const gate = useCallback(
    (step: WorkflowStepId) => workflowGate(workflow, step, restoreState),
    [restoreState, workflow],
  )

  const persistedStep = workflowSteps.some((step) => step.id === workflow.currentStep)
    ? workflow.currentStep
    : 'model-type'
  const requested = (workflowSteps.some((step) => step.id === stepId)
    ? stepId
    : stepId
      ? 'model-type'
      : persistedStep) as WorkflowStepId

  useEffect(() => {
    if (hydrating) return
    const result = gate(requested)
    if (!result.allowed) {
      setGuardNotice({ message: result.reason, target: result.step })
      if (requested !== result.step) navigate(`/workflow/${result.step}`, { replace: true })
      return
    }
    if (guardNotice && guardNotice.target !== requested) setGuardNotice(null)
    if (workflow.currentStep !== requested) setContext({ currentStep: requested })
  }, [gate, guardNotice, hydrating, navigate, requested, setContext, workflow.currentStep])

  const go = (step: WorkflowStepId) => {
    if (hydrating) {
      setGuardNotice({ message: '正在按资源 ID 恢复训练上下文，请稍候。', target: requested })
      return
    }
    const result = gate(step)
    if (result.allowed) {
      setGuardNotice(null)
      navigate(`/workflow/${step}`)
    } else {
      setGuardNotice({ message: result.reason, target: result.step })
      navigate(`/workflow/${result.step}`)
    }
  }

  const setDataset = useCallback((dataset: DatasetUploadResult) => setContext({
    dataset,
    datasetId: dataset.id,
    preprocessScriptId: null,
    preprocessTaskId: null,
    preprocessTask: null,
    splitId: null,
    split: null,
    trainScriptId: null,
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }), [setContext])

  const completePreprocess = useCallback((task: PreprocessTask, scriptId: string | null) => setContext({
    preprocessTask: task,
    preprocessTaskId: task.id,
    preprocessScriptId: scriptId,
    splitId: null,
    split: null,
    trainScriptId: null,
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }), [setContext])

  const completeSplit = useCallback((split: DatasetSplitResult) => setContext({
    splitId: split.id,
    split,
    trainScriptId: null,
    trainingJobId: null,
    trainingJob: null,
    modelVersionId: null,
    evaluation: null,
    modelVersion: null,
  }), [setContext])

  const completeTraining = useCallback((trainingJob: TrainingJob) => setContext({
    trainingJob,
    trainingJobId: trainingJob.id,
    modelVersionId: trainingJob.model_version_id ?? null,
    modelVersion: null,
    evaluation: null,
  }), [setContext])

  const clearMissingJob = useCallback(() => {
    setContext(clearFromTrainingJob())
  }, [setContext])

  const activeIndex = workflowSteps.findIndex((step) => step.id === requested)
  const content = requested === 'model-type'
    ? <ModelTypeStep select={(modelType) => {
      setGuardNotice(null)
      if (queryDatasetId && !queryModel) setContext({ modelType, currentStep: 'upload' })
      else setModelType(modelType)
      navigate('/workflow/upload')
    }} />
    : requested === 'upload'
      ? <UploadStep back={() => go('model-type')} restoring={hydrating} setDataset={setDataset} />
      : requested === 'preprocess'
        ? <PreprocessStep back={() => go('upload')} complete={completePreprocess} restoring={hydrating} />
        : requested === 'split'
          ? <SplitStep back={() => go('preprocess')} complete={completeSplit} restoring={hydrating} />
          : requested === 'train'
            ? (
              <TrainingStep
                back={() => go('split')}
                clearMissingJob={clearMissingJob}
                complete={completeTraining}
                pollingEnabled={restoreState.blockedStep !== 'train'
                  || Boolean(workflow.trainingJob && !terminalTrainingStatuses.has(upperStatus(workflow.trainingJob.status) ?? ''))}
                restoring={hydrating}
              />
            )
            : requested === 'evaluate'
              ? <EvaluationStep back={() => go('train')} publish={() => go('publish')} restoring={hydrating} />
              : <PublishStep back={() => go('evaluate')} restoring={hydrating} />

  const retryRestore = () => {
    if (queryContextActive) setQueryRestore({ key: null, loading: false, error: null })
    setRestoreState((state) => ({ ...state, loading: true, error: null, blockedStep: null }))
    setRestoreAttempt((value) => value + 1)
  }

  return (
    <div className="page-content workflow-page">
      <div className="workflow-header">
        <div>
          <p className="eyebrow">TRAINING WORKFLOW / CONTROL ROOM</p>
          <h1 className="page-title">训练工作流</h1>
          <p className="page-description">按顺序完成模型选择、数据检查、训练和发布。刷新页面后已完成的上下文会保留。</p>
        </div>
      </div>
      {hydrating && <div className="loading-line" role="status"><span className="spinner" />正在按资源 ID 恢复训练上下文…</div>}
      {queryRestore.error && !queryRestore.loading && <ErrorBox message={queryRestore.error} onRetry={retryRestore} />}
      {restoreState.error && !restoreState.loading && <ErrorBox message={restoreState.error} onRetry={retryRestore} />}
      {guardNotice && <ErrorBox message={guardNotice.message} />}
      <section className="step-shell" aria-label="训练流程步骤">
        <div className="step-list">
          {workflowSteps.map((step, index) => {
            const stepGate = gate(step.id)
            const stepNumber = String(index + 1).padStart(2, '0')
            const locked = !stepGate.allowed || hydrating
            return (
              <button
                aria-label={locked ? `${stepNumber} × ${step.label} ${step.caption}` : undefined}
                className={`step-item${index === activeIndex ? ' active' : ''}${locked ? ' locked' : ''}`}
                key={step.id}
                onClick={() => go(step.id)}
                type="button"
              >
                <div className="step-number">{stepNumber}</div>
                <span className="step-label">{step.label}</span>
                <span className="step-caption">{step.caption}</span>
              </button>
            )
          })}
        </div>
        {content}
      </section>
    </div>
  )
}
