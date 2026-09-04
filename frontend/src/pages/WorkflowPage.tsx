import { useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ApiError, apiClient } from '../api'
import { useAppStore } from '../store/useAppStore'
import { EvaluationDashboard } from '../components/evaluation/EvaluationCharts'
import { MODEL_TYPE_CODES, MODEL_TYPE_NAMES, type DatasetSplitResult, type DatasetUploadResult, type ModelTypeCode, type PreprocessTask, type ScriptContract, type TrainingJob } from '../types/contracts'
import { workflowSteps, type WorkflowStepId } from '../types/workflow'

const modelDescriptions: Record<ModelTypeCode, string> = {
  electric_load: '面向建筑与设备用电负荷的时序预测。',
  heating_cooling_load: '面向冷热源系统负荷的时序预测。',
  integrated_energy: '面向综合能源消耗趋势的时序预测。',
}

function errorMessage(error: unknown) {
  return error instanceof ApiError ? error.message : error instanceof Error ? error.message : '请求失败，请稍后重试'
}

function asLogs(logs: Array<{ message: string } | string> | undefined) {
  return (logs ?? []).map((item) => typeof item === 'string' ? item : item.message)
}

function ErrorBox({ message, onRetry }: { message: string | null; onRetry?: () => void }) {
  if (!message) return null
  return <div className="alert-box error" role="alert"><span>{message}</span>{onRetry && <button type="button" onClick={onRetry}>重试</button>}</div>
}

function InfoBox({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warning' | 'success' }) {
  return <div className={`alert-box ${tone}`} role="status">{children}</div>
}

function Button({ children, disabled, onClick, kind = 'primary', type = 'button' }: { children: ReactNode; disabled?: boolean; onClick?: () => void; kind?: 'primary' | 'secondary' | 'danger'; type?: 'button' | 'submit' }) {
  return <button className={`${kind}-button action-button`} disabled={disabled} onClick={onClick} type={type}>{children}</button>
}

function StepActions({ back, next, nextDisabled, nextLabel = '继续' }: { back?: () => void; next?: () => void; nextDisabled?: boolean; nextLabel?: string }) {
  return <div className="step-actions">{back && <Button kind="secondary" onClick={back}>返回上一步</Button>}{next && <Button disabled={nextDisabled} onClick={next}>{nextLabel} <span aria-hidden="true">→</span></Button>}</div>
}

function ModelTypeStep({ select }: { select: (type: ModelTypeCode) => void }) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const navigate = useNavigate()
  return <section className="workflow-panel">
    <div className="panel-heading"><div><p className="eyebrow">STEP 01 / MODEL TYPE</p><h2>选择要训练的模型类型</h2><p>先确定目标模型，后续数据与脚本会按类型校验。</p></div><span className="panel-count">{modelType ? '已选择' : '必选'}</span></div>
    <div className="model-select-grid">{MODEL_TYPE_CODES.map((code) => <button className={`model-option${modelType === code ? ' selected' : ''}`} key={code} onClick={() => select(code)} type="button"><span className="model-option-icon">{code === 'electric_load' ? '⚡' : code === 'heating_cooling_load' ? '◒' : '⌁'}</span><span><strong>{MODEL_TYPE_NAMES[code]}</strong><small>{modelDescriptions[code]}</small><em>{code}</em></span><b aria-hidden="true">{modelType === code ? '✓' : '○'}</b></button>)}</div>
    <StepActions next={() => navigate('/workflow/upload')} nextDisabled={!modelType} nextLabel="进入数据上传" />
  </section>
}

function DatasetSummary({ dataset }: { dataset: DatasetUploadResult }) {
  const missing = dataset.missing_values ?? {}
  return <div className="dataset-summary">
    <div className="summary-stat-grid"><div><small>数据行数</small><strong>{dataset.row_count.toLocaleString()}</strong></div><div><small>字段数</small><strong>{dataset.column_count}</strong></div><div><small>数值列</small><strong>{dataset.numeric_columns?.length ?? 0}</strong></div><div><small>时间解析</small><strong className="success-text">{dataset.time_parse?.success ? '成功' : '失败'}</strong></div></div>
    <div className="summary-meta"><span>时间列：<b>{dataset.time_column}</b></span><span>目标列：<b>{dataset.target_column}</b></span><span>特征列：<b>{dataset.feature_columns?.join('、') || '—'}</b></span></div>
    {dataset.time_parse && <div className="inline-note">时间范围：{dataset.time_parse.min ?? '—'} ～ {dataset.time_parse.max ?? '—'} {dataset.time_parse.message && `· ${dataset.time_parse.message}`}</div>}
    <div className="table-wrap"><table><thead><tr><th>字段</th><th>角色</th><th>类型</th><th>缺失值</th></tr></thead><tbody>{(dataset.columns ?? []).map((column) => <tr key={column.name}><td><b>{column.name}</b></td><td><span className={`role-pill ${column.role}`}>{column.role === 'time' ? '时间' : column.role === 'target' ? '目标' : '特征'}</span></td><td>{column.data_type}</td><td>{column.missing_count}（{(column.missing_ratio * 100).toFixed(1)}%）{missing[column.name]?.missing_count ? ' · 可在预处理中处理' : ''}</td></tr>)}</tbody></table></div>
    <h3 className="subheading">样例数据（前 5 行）</h3><div className="table-wrap preview-table"><table><thead><tr>{dataset.columns.map((column) => <th key={column.name}>{column.name}</th>)}</tr></thead><tbody>{(dataset.preview_rows ?? []).map((row, index) => <tr key={index}>{dataset.columns.map((column) => <td key={column.name}>{String(row[column.name] ?? '—')}</td>)}</tr>)}</tbody></table></div>
  </div>
}

function UploadStep({ setDataset, back }: { setDataset: (dataset: DatasetUploadResult) => void; back: () => void }) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const dataset = useAppStore((state) => state.workflow.dataset)
  const navigate = useNavigate()
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const upload = async (file?: File) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) { setError('仅支持 .csv 文件，请重新选择'); return }
    setUploading(true); setError(null)
    try { setDataset(await apiClient.uploadDataset(file, { model_type: modelType ?? undefined })) } catch (reason) { setError(errorMessage(reason)) } finally { setUploading(false) }
  }
  return <section className="workflow-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 02 / DATASET</p><h2>上传并检查 CSV 数据</h2><p>上传后自动解析完整表头、字段角色、时间列和缺失值信息。</p></div></div>
    <ErrorBox message={error} />
    {!dataset && <label className={`drop-zone${dragging ? ' dragging' : ''}`} onDragEnter={(event) => { event.preventDefault(); setDragging(true) }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void upload(event.dataTransfer.files[0]) }}><input accept=".csv,text/csv" onChange={(event) => void upload(event.target.files?.[0])} type="file" /><span className="upload-icon">↑</span><strong>{uploading ? '正在上传并解析…' : '拖拽 CSV 文件到这里'}</strong><small>或点击选择文件 · 支持 UTF-8 / GB18030 · 最大 50 MB</small></label>}
    {uploading && <div className="loading-line"><span className="spinner" />上传、解析与校验进行中</div>}
    {dataset && <><div className="file-chip"><span>CSV</span><b>{dataset.file_name}</b><small>{dataset.status === 'parsed' ? '解析完成' : dataset.status}</small></div><DatasetSummary dataset={dataset} /></>}
    <StepActions back={back} next={() => navigate('/workflow/preprocess')} nextDisabled={!dataset || uploading} />
  </section>
}

function LogList({ logs }: { logs: Array<{ message: string } | string> | undefined }) {
  const entries = asLogs(logs)
  return <div className="log-box">{entries.length ? entries.map((log, index) => <div key={`${log}-${index}`}><span>•</span>{log}</div>) : <span className="muted-caption">暂无日志</span>}</div>
}

function PreprocessStep({ back, complete }: { back: () => void; complete: (task: PreprocessTask, scriptId: string | null) => void }) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const datasetId = useAppStore((state) => state.workflow.datasetId)
  const task = useAppStore((state) => state.workflow.preprocessTask)
  const selected = useAppStore((state) => state.workflow.preprocessScriptId)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const navigate = useNavigate()
  const [scripts, setScripts] = useState<ScriptContract[]>([])
  const [skip, setSkip] = useState(!selected)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (!modelType) return; apiClient.listScripts({ model_type: modelType, script_type: 'preprocessor' }).then((response) => setScripts(response.items)).catch((reason) => setError(errorMessage(reason))).finally(() => setLoading(false)) }, [modelType])
  const run = async () => {
    if (!datasetId || !modelType || (!skip && !selected)) return
    setRunning(true); setError(null)
    try { const result = await apiClient.createPreprocessingTask({ model_type: modelType, dataset_id: datasetId, preprocess_script_id: skip ? null : selected, mode: skip ? 'skip' : 'use', skip }); complete(result, skip ? null : selected); navigate('/workflow/split') } catch (reason) { setError(errorMessage(reason)) } finally { setRunning(false) }
  }
  return <section className="workflow-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 03 / PREPROCESS</p><h2>选择数据预处理方式</h2><p>预处理在后端执行并保存阶段日志；也可以明确跳过。</p></div></div><ErrorBox message={error} />
    <div className="choice-row"><label className={`skip-option${skip ? ' checked' : ''}`}><input checked={skip} onChange={(event) => { setSkip(event.target.checked); if (event.target.checked) setContext({ preprocessScriptId: null }) }} type="checkbox" /><span><b>跳过预处理</b><small>直接使用原始特征进入数据集划分</small></span></label><span className="muted-caption">{loading ? '加载脚本中…' : `${scripts.length} 个可用脚本`}</span></div>
    {!skip && <div className="script-list">{scripts.map((script) => <button className={`script-option${selected === script.id ? ' selected' : ''}`} key={script.id} onClick={() => setContext({ preprocessScriptId: script.id })} type="button"><span><b>{script.name}</b><small>版本 {script.version} · {script.status.toLowerCase() === 'enabled' ? '已启用' : '已停用'}</small></span><i>{selected === script.id ? '✓' : '选择'}</i></button>)}{!loading && !scripts.length && <div className="empty-state compact">没有适用于该模型的预处理脚本，可选择跳过。</div>}</div>}
    {task && <div className="stage-result"><div className="stage-title"><span className={`status-dot ${task.status.toLowerCase() === 'failed' ? 'danger' : 'success'}`} />预处理{task.status.toLowerCase() === 'skipped' ? '已跳过' : task.status.toLowerCase() === 'succeeded' ? '已完成' : task.status}</div><div className="stage-track">{['数据读取', '预处理', '结果校验', '完成'].map((stage, index) => <span className={task.stage === 'completed' || index < 3 ? 'done' : ''} key={stage}>{index + 1}. {stage}</span>)}</div><div className="stage-summary"><span>输入：{task.input_row_count ?? '—'} 行</span><span>输出：{task.output_row_count ?? '—'} 行</span><span>字段：{task.output_columns?.join('、') || task.input_columns?.join('、') || '—'}</span></div><LogList logs={task.logs} />{task.error_message && <p className="error-text">{task.error_message}</p>}</div>}
    <StepActions back={back} next={run} nextDisabled={running || (!skip && !selected) || !datasetId} nextLabel={running ? '执行中…' : task?.status.toLowerCase() === 'succeeded' || task?.status.toLowerCase() === 'skipped' ? '重新执行' : '执行并继续'} />
  </section>
}

function SplitStep({ back, complete }: { back: () => void; complete: (split: DatasetSplitResult) => void }) {
  const datasetId = useAppStore((state) => state.workflow.datasetId)
  const navigate = useNavigate()
  const taskId = useAppStore((state) => state.workflow.preprocessTaskId)
  const split = useAppStore((state) => state.workflow.split)
  const [loading, setLoading] = useState(!split)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (!datasetId || split) return; apiClient.getDatasetSplit(datasetId).then(complete).catch((reason: unknown) => { if (!(reason instanceof ApiError && reason.status === 404)) setError(errorMessage(reason)) }).finally(() => setLoading(false)) }, [datasetId, split, complete])
  const create = async () => { if (!datasetId) return; setLoading(true); setError(null); try { complete(await apiClient.splitDataset(datasetId, taskId)) } catch (reason) { setError(errorMessage(reason)) } finally { setLoading(false) } }
  return <section className="workflow-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 04 / FIXED SPLIT</p><h2>数据集划分</h2><p>平台按时间升序固定划分 80% 训练集与 20% 测试集，不提供比例编辑。</p></div></div><ErrorBox message={error} />{loading && !split && <div className="loading-line"><span className="spinner" />读取划分结果…</div>}{split ? <div className="split-result"><InfoBox tone="success">✓ 划分已完成 · 时间升序排序后固定切分</InfoBox><div className="split-bars"><div><span style={{ width: '80%' }} /><b>训练集 80%</b><strong>{split.train_row_count.toLocaleString()} 行</strong></div><div><span style={{ width: '20%' }} /><b>测试集 20%</b><strong>{split.test_row_count.toLocaleString()} 行</strong></div></div><div className="split-detail"><div><b>训练集时间范围</b><span>{split.train_time_start} ～ {split.train_time_end}</span></div><div><b>测试集时间范围</b><span>{split.test_time_start} ～ {split.test_time_end}</span></div></div></div> : <div className="empty-state">尚未创建数据集划分。点击下方按钮生成固定 80/20 结果。</div>}<StepActions back={back} next={split ? () => navigate('/workflow/train') : create} nextDisabled={loading || !datasetId} nextLabel={split ? '继续选择训练脚本' : '生成 80/20 划分'} /></section>
}

function TrainingStep({ back, complete }: { back: () => void; complete: (job: TrainingJob) => void }) {
  const modelType = useAppStore((state) => state.workflow.modelType)
  const datasetId = useAppStore((state) => state.workflow.datasetId)
  const navigate = useNavigate()
  const preprocessScriptId = useAppStore((state) => state.workflow.preprocessScriptId)
  const preprocessTaskId = useAppStore((state) => state.workflow.preprocessTaskId)
  const job = useAppStore((state) => state.workflow.trainingJob)
  const selected = useAppStore((state) => state.workflow.trainScriptId)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const [scripts, setScripts] = useState<ScriptContract[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (!modelType) return; apiClient.listScripts({ model_type: modelType, script_type: 'trainer' }).then((response) => setScripts(response.items)).catch((reason) => setError(errorMessage(reason))) }, [modelType])
  useEffect(() => { if (!job || ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.status)) return; let cancelled = false; const poll = async () => { try { const next = await apiClient.getTrainingJob(job.id); if (!cancelled) complete(next) } catch (reason) { if (!cancelled) setError(errorMessage(reason)) } }; void poll(); const timer = window.setInterval(() => void poll(), 1500); return () => { cancelled = true; window.clearInterval(timer) } }, [job?.id, job?.status, complete])
  const start = async () => { if (!modelType || !datasetId || !selected) return; setLoading(true); setError(null); try { complete(await apiClient.createTrainingJob({ model_type: modelType, dataset_id: datasetId, preprocess_script_id: preprocessScriptId, preprocessing_task_id: preprocessTaskId, train_script_id: selected })) } catch (reason) { setError(errorMessage(reason)) } finally { setLoading(false) } }
  const retry = async () => { if (!job) return; setLoading(true); setError(null); try { complete(await apiClient.retryTrainingJob(job.id)) } catch (reason) { setError(errorMessage(reason)) } finally { setLoading(false) } }
  const terminal = job ? ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.status) : false
  return <section className="workflow-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 05 / TRAINING</p><h2>选择训练脚本并启动</h2><p>训练会生成候选模型，不会改变当前生产模型；完成后再决定是否发布。</p></div></div><ErrorBox message={error} />
    {!job && <div className="script-list">{scripts.map((script) => <button className={`script-option${selected === script.id ? ' selected' : ''}`} key={script.id} onClick={() => setContext({ trainScriptId: script.id })} type="button"><span><b>{script.name}</b><small>版本 {script.version} · 适用 {MODEL_TYPE_NAMES[modelType as ModelTypeCode]}</small></span><i>{selected === script.id ? '✓' : '选择'}</i></button>)}{!scripts.length && <div className="empty-state compact">暂无可用训练脚本，请先在脚本库启用兼容脚本。</div>}</div>}
    {job && <div className="training-status"><div className="status-header"><span className={`status-badge ${job.status === 'FAILED' ? 'danger' : job.status === 'SUCCEEDED' ? 'success' : 'info'}`}>{job.status}</span><b>{job.current_stage ?? job.progress_stage ?? '任务处理中'}</b>{!terminal && <span className="spinner" />}</div><div className="training-stages">{['准备数据', '加载训练脚本', '执行训练', '保存模型', '进入评估'].map((stage) => <span className={(job.logs ?? []).some((log) => (typeof log === 'string' ? log : log.message).includes(stage)) ? 'done' : ''} key={stage}>{stage}</span>)}</div><LogList logs={job.logs} />{job.error_message && <p className="error-text">{job.error_message}</p>}{job.status === 'FAILED' && <InfoBox tone="warning">本次训练失败，不会影响生产模型。修复脚本或配置后可以重试。</InfoBox>}</div>}
    <StepActions back={back} next={job?.status === 'SUCCEEDED' ? () => navigate('/workflow/evaluate') : start} nextDisabled={loading || !selected || !datasetId || (job !== null && job.status !== 'FAILED')} nextLabel={loading ? '提交中…' : job?.status === 'SUCCEEDED' ? '查看评估结果' : job?.status === 'FAILED' ? '重新启动' : job ? '训练进行中…' : '启动训练'} />{job?.status === 'FAILED' && <div className="retry-row"><Button kind="secondary" disabled={loading} onClick={retry}>失败重试</Button></div>}</section>
}

function EvaluationStep({ back, publish }: { back: () => void; publish: () => void }) {
  const job = useAppStore((state) => state.workflow.trainingJob)
  const evaluation = useAppStore((state) => state.workflow.evaluation)
  const loadEvaluation = useAppStore((state) => state.setWorkflowContext)
  const [loading, setLoading] = useState(!evaluation)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (!job?.id || evaluation) return; let alive = true; apiClient.getTrainingJobEvaluation(job.id).then((result) => alive && loadEvaluation({ evaluation: result })).catch((reason) => alive && setError(errorMessage(reason))).finally(() => alive && setLoading(false)); return () => { alive = false } }, [job?.id, evaluation, loadEvaluation])
  const retry = () => { setError(null); setLoading(true); loadEvaluation({ evaluation: null }) }
  return <section className="workflow-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 06 / EVALUATION</p><h2>评估结果与模型对比</h2><p>指标来自完整测试集；图表按测试集时间展示，数据量过大时仅抽样图表点位。</p></div></div><ErrorBox message={error} onRetry={retry} />{loading && <div className="loading-line"><span className="spinner" />读取评估结果…</div>}{evaluation && <><EvaluationDashboard evaluation={evaluation} /><InfoBox>评估已完成，可进入保存与发布。图表悬停可查看具体时间与数值。</InfoBox></>} {!loading && !evaluation && <div className="empty-state">暂无评估结果，请确认训练任务已成功完成。</div>}<StepActions back={back} next={publish} nextDisabled={!evaluation} nextLabel="保存与发布" /></section>
}

function PublishStep({ back }: { back: () => void }) {
  const workflow = useAppStore((state) => state.workflow)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const [saving, setSaving] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [saved, setSaved] = useState(Boolean(workflow.modelVersion))
  const [published, setPublished] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const save = async () => { if (!workflow.modelVersion || !workflow.modelType) return; setSaving(true); setError(null); try { const model = await apiClient.saveModel(workflow.modelVersion.id, { model_type: workflow.modelType, status: 'READY', training_job_id: workflow.trainingJobId ?? undefined, train_script_id: workflow.trainScriptId ?? undefined, preprocess_script_id: workflow.preprocessScriptId, preprocess_used: Boolean(workflow.preprocessScriptId), time_column: workflow.dataset?.time_column, feature_columns: workflow.dataset?.feature_columns, target_column: workflow.dataset?.target_column, metrics: (workflow.evaluation?.metrics ?? {}) as Record<string, import('../types/contracts').JsonValue> }); setContext({ modelVersion: model }); setSaved(true) } catch (reason) { setError(errorMessage(reason)) } finally { setSaving(false) } }
  const publish = async () => { if (!workflow.modelVersion) return; if (!window.confirm('发布后将成为当前生产模型，是否确认发布？')) return; setPublishing(true); setError(null); try { const result = await apiClient.publishModel(workflow.modelVersion.id, { confirmed: true, message: '通过训练工作流发布' }); setContext({ modelVersion: result.model as typeof workflow.modelVersion }); setPublished(true) } catch (reason) { setError(errorMessage(reason)) } finally { setPublishing(false) } }
  return <section className="workflow-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 07 / PUBLISH</p><h2>保存与发布模型</h2><p>保存候选版本后，发布前会再次确认；未发布的候选不会影响生产模型。</p></div></div><ErrorBox message={error} />{workflow.modelVersion ? <div className="publish-card"><div className="publish-model"><span className="model-symbol">◆</span><div><b>{MODEL_TYPE_NAMES[workflow.modelType!]}</b><small>版本 {workflow.modelVersion.version} · {published ? '已发布' : saved ? '已保存，待发布' : '训练候选'}</small></div><span className={`status-badge ${published ? 'success' : 'info'}`}>{published ? 'PUBLISHED' : workflow.modelVersion.status}</span></div><dl className="publish-facts"><div><dt>训练任务</dt><dd>{workflow.trainingJobId ?? '—'}</dd></div><div><dt>生产影响</dt><dd>发布前不会改变</dd></div></dl><div className="publish-actions"><Button disabled={saving || saved} kind="secondary" onClick={save}>{saving ? '保存中…' : saved ? '已保存' : '保存候选模型'}</Button><Button disabled={!saved || publishing || published} onClick={publish}>{publishing ? '发布中…' : published ? '发布成功' : '发布模型'}</Button></div></div> : <div className="empty-state">训练尚未生成可保存的模型版本。</div>}{published && <InfoBox tone="success">✓ 发布成功，模型已成为当前版本。</InfoBox>}<StepActions back={back} /></section>
}

export function WorkflowPage() {
  const navigate = useNavigate()
  const { stepId } = useParams<{ stepId?: string }>()
  const [searchParams] = useSearchParams()
  const workflow = useAppStore((state) => state.workflow)
  const setContext = useAppStore((state) => state.setWorkflowContext)
  const [guardError, setGuardError] = useState<string | null>(null)
  const requested = (workflowSteps.some((step) => step.id === stepId) ? stepId : stepId ? 'model-type' : workflow.currentStep) as WorkflowStepId
  const canEnter = useCallback((step: WorkflowStepId) => {
    if (step !== 'model-type' && !workflow.modelType) return false
    if (['preprocess', 'split', 'train', 'evaluate', 'publish'].includes(step) && !workflow.datasetId) return false
    if (['split', 'train', 'evaluate', 'publish'].includes(step) && !workflow.preprocessTaskId) return false
    if (['train', 'evaluate', 'publish'].includes(step) && !workflow.split) return false
    if (['evaluate', 'publish'].includes(step) && workflow.trainingJob?.status !== 'SUCCEEDED') return false
    return true
  }, [workflow])
  useEffect(() => {
    const queryModel = searchParams.get('model') as ModelTypeCode | null
    if (queryModel && MODEL_TYPE_CODES.includes(queryModel)) setContext({ modelType: queryModel })
  }, [searchParams, setContext])
  useEffect(() => {
    if (!canEnter(requested)) {
      const firstAvailable = !workflow.modelType ? 'model-type' : !workflow.datasetId ? 'upload' : !workflow.preprocessTaskId ? 'preprocess' : !workflow.split ? 'split' : !workflow.trainingJobId || !workflow.trainingJob || workflow.trainingJob.status !== 'SUCCEEDED' ? 'train' : 'evaluate'
      setGuardError('请先完成前置步骤，当前页面已为你返回到可继续的位置。')
      navigate(`/workflow/${firstAvailable}`, { replace: true })
      return
    }
    setGuardError(null)
    setContext({ currentStep: requested })
  }, [requested, canEnter, navigate, setContext, workflow.modelType, workflow.datasetId, workflow.preprocessTaskId, workflow.split, workflow.trainingJobId])
  const go = (step: WorkflowStepId) => { if (canEnter(step)) navigate(`/workflow/${step}`); else setGuardError('该步骤尚未解锁，请按顺序完成前置步骤。') }
  const activeIndex = workflowSteps.findIndex((step) => step.id === requested)
  const content = requested === 'model-type' ? <ModelTypeStep select={(modelType) => { setContext({ modelType, datasetId: null, dataset: null, preprocessScriptId: null, preprocessTaskId: null, preprocessTask: null, split: null, trainScriptId: null, trainingJobId: null, trainingJob: null, evaluation: null, modelVersion: null, currentStep: 'model-type' }); navigate('/workflow/upload') }} /> : requested === 'upload' ? <UploadStep back={() => go('model-type')} setDataset={(dataset) => setContext({ dataset, datasetId: dataset.id, preprocessTaskId: null, preprocessTask: null, split: null, trainingJobId: null, trainingJob: null, evaluation: null, modelVersion: null })} /> : requested === 'preprocess' ? <PreprocessStep back={() => go('upload')} complete={(task, scriptId) => setContext({ preprocessTask: task, preprocessTaskId: task.id, preprocessScriptId: scriptId, split: null, trainingJobId: null, trainingJob: null, evaluation: null, modelVersion: null })} /> : requested === 'split' ? <SplitStep back={() => go('preprocess')} complete={(split) => setContext({ split })} /> : requested === 'train' ? <TrainingStep back={() => go('split')} complete={(trainingJob) => setContext({ trainingJob, trainingJobId: trainingJob.id, modelVersion: trainingJob.model_version_id ? { id: trainingJob.model_version_id, model_type: trainingJob.model_type, version: '候选版本', status: 'READY', is_baseline: false, is_current: false, is_abnormal: false, is_rollback_available: false, metrics: null, created_at: trainingJob.created_at, published_at: null } : workflow.modelVersion })} /> : requested === 'evaluate' ? <EvaluationStep back={() => go('train')} publish={() => go('publish')} /> : <PublishStep back={() => go('evaluate')} />
  return <div className="page-content"><div className="workflow-header"><div><p className="eyebrow">TRAINING WORKFLOW / CONTROL ROOM</p><h1 className="page-title">训练工作流</h1><p className="page-description">按顺序完成模型选择、数据检查、训练和发布。刷新页面后已完成的上下文会保留。</p></div></div>{guardError && <ErrorBox message={guardError} />}<section className="step-shell" aria-label="训练流程步骤"><div className="step-list">{workflowSteps.map((step, index) => <button className={`step-item${index === activeIndex ? ' active' : ''}${canEnter(step.id) ? '' : ' locked'}`} key={step.id} onClick={() => go(step.id)} type="button"><div className="step-number">{canEnter(step.id) ? String(index + 1).padStart(2, '0') : '×'}</div><span className="step-label">{step.label}</span><span className="step-caption">{step.caption}</span></button>)}</div>{content}</section></div>
}
