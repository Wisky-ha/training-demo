import { useMemo, useState } from 'react'
import type { MouseEvent, ReactNode } from 'react'
import type { MetricName, ModelEvaluation } from '../../types/contracts'
import {
  evaluationMetrics,
  evaluationModels,
  metricDelta,
  metricLabel,
  normalizeEvaluationData,
  type EvaluationModelResult,
  type NormalizedEvaluationPoint,
} from '../../utils/evaluation'

const WIDTH = 760
const HEIGHT = 280
const PLOT = { left: 52, right: 18, top: 20, bottom: 42 }
const COLORS = { actual: '#246bfe', candidate: '#18a673', baseline: '#e18a25', error: '#e04f5f' }
const METRICS: MetricName[] = ['mae', 'rmse', 'mape', 'r2']

type LineSeries = { key: string; label: string; color: string; values: Array<number | null> }

function finite(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value)
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (!finite(value)) return '暂无'
  return Math.abs(value) >= 10000 ? value.toExponential(3) : value.toFixed(digits)
}

function formatTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'short', timeStyle: 'short' }).format(date)
}

function formatAxisTime(value: string): string {
  const formatted = formatTime(value)
  return formatted.length > 18 ? `${formatted.slice(0, 17)}…` : formatted
}

function extent(series: LineSeries[]): [number, number] | null {
  const values = series.flatMap((item) => item.values).filter(finite)
  if (!values.length) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (min === max) {
    const padding = Math.abs(min) * 0.1 || 1
    return [min - padding, max + padding]
  }
  const padding = (max - min) * 0.08
  return [min - padding, max + padding]
}

function pathFor(values: Array<number | null>, y: (value: number) => number, x: (index: number) => number): string {
  let path = ''
  values.forEach((value, index) => {
    if (!finite(value)) return
    const command = index === 0 || !finite(values[index - 1]) ? 'M' : 'L'
    path += `${command} ${x(index).toFixed(2)} ${y(value).toFixed(2)} `
  })
  return path.trim()
}

function ChartEmpty({ message }: { message: string }) {
  return <div className="chart-empty" role="status"><span aria-hidden="true">⌁</span><p>{message}</p></div>
}

function ChartLegend({ series }: { series: LineSeries[] }) {
  return <div className="chart-legend">{series.map((item) => <span key={item.key}><i style={{ backgroundColor: item.color }} />{item.label}</span>)}</div>
}

function LineChart({
  title,
  description,
  points,
  series,
  error = false,
}: {
  title: string
  description: string
  points: NormalizedEvaluationPoint[]
  series: LineSeries[]
  error?: boolean
}) {
  const [hover, setHover] = useState<number | null>(null)
  const innerWidth = WIDTH - PLOT.left - PLOT.right
  const innerHeight = HEIGHT - PLOT.top - PLOT.bottom
  const range = extent(series)
  if (!points.length || !range) return <article className="chart-card"><div className="chart-heading"><div><h3>{title}</h3><p>{description}</p></div></div><ChartEmpty message="暂无可绘制的有效数据" /></article>
  const x = (index: number) => PLOT.left + (points.length === 1 ? innerWidth / 2 : index * innerWidth / (points.length - 1))
  const y = (value: number) => PLOT.top + (range[1] - value) * innerHeight / (range[1] - range[0])
  const active = hover == null ? null : points[hover]
  const activeX = hover == null ? 0 : x(hover)
  const ticks = [range[1], range[1] - (range[1] - range[0]) / 2, range[0]]
  const labels = [points[0].timestamp, points[Math.floor((points.length - 1) / 2)].timestamp, points[points.length - 1].timestamp]
  const setHoverFromEvent = (event: MouseEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.ownerSVGElement?.getBoundingClientRect()
    if (!bounds) return
    const viewX = (event.clientX - bounds.left) / bounds.width * WIDTH
    const index = Math.round((Math.min(Math.max(viewX, PLOT.left), WIDTH - PLOT.right) - PLOT.left) / innerWidth * (points.length - 1))
    setHover(index)
  }
  return <article className="chart-card"><div className="chart-heading"><div><h3>{title}</h3><p>{description}</p></div><ChartLegend series={series} /></div><div className="chart-visual"><svg aria-label={title} role="img" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} onMouseLeave={() => setHover(null)}>
    {ticks.map((tick) => <g key={tick}><line className="chart-grid-line" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={y(tick)} y2={y(tick)} /><text className="chart-axis-label" x={PLOT.left - 8} y={y(tick) + 3} textAnchor="end">{formatNumber(tick, 2)}</text></g>)}
    <line className="chart-axis-line" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={HEIGHT - PLOT.bottom} y2={HEIGHT - PLOT.bottom} />
    {series.map((item) => <path className="chart-line" d={pathFor(item.values, y, x)} key={item.key} style={{ stroke: item.color }} />)}
    {hover != null && <line className="chart-hover-line" x1={activeX} x2={activeX} y1={PLOT.top} y2={HEIGHT - PLOT.bottom} />}
    {series.map((item) => item.values.map((value, index) => finite(value) && (hover === index || points.length < 80) ? <circle className="chart-point" cx={x(index)} cy={y(value)} fill={item.color} key={`${item.key}-${index}`} onFocus={() => setHover(index)} onMouseEnter={() => setHover(index)} r={hover === index ? 4 : 2.2} tabIndex={0}><title>{`${item.label} · ${formatTime(points[index].timestamp)} · ${formatNumber(value)}`}</title></circle> : null))}
    {labels.map((label, index) => <text className="chart-time-label" key={`${label}-${index}`} x={[PLOT.left, WIDTH / 2, WIDTH - PLOT.right][index]} y={HEIGHT - 14} textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{formatAxisTime(label)}</text>)}
    <rect aria-label="在图表上查看时间点详情" className="chart-hover-target" x={PLOT.left} y={PLOT.top} width={innerWidth} height={innerHeight} onMouseMove={setHoverFromEvent} onMouseLeave={() => setHover(null)} />
  </svg>{active && <div className="chart-tooltip" style={{ left: `${Math.min(88, Math.max(12, activeX / WIDTH * 100))}%` }}><b>{formatTime(active.timestamp)}</b>{series.map((item) => <span key={item.key}><i style={{ backgroundColor: item.color }} />{item.label}：{formatNumber(item.values[hover ?? 0])}</span>)}{error && <small>误差 = 实际值 − 预测值</small>}</div>}</div></article>
}

function MetricComparisonChart({ candidate, production }: { candidate: EvaluationModelResult | null; production: EvaluationModelResult | null }) {
  const [hover, setHover] = useState<number | null>(null)
  if (!candidate || !production) return <article className="chart-card"><div className="chart-heading"><div><h3>模型指标对比</h3><p>当前生产模型数据由后端提供时展示</p></div></div><ChartEmpty message="暂无当前生产模型对比数据" /></article>
  const values = METRICS.flatMap((metric) => [candidate.metrics[metric], production.metrics[metric]]).filter(finite)
  if (!values.length) return <article className="chart-card"><div className="chart-heading"><div><h3>模型指标对比</h3><p>当前生产模型与本次新模型</p></div></div><ChartEmpty message="暂无可绘制的指标数据" /></article>
  const rawMin = Math.min(0, ...values)
  const rawMax = Math.max(0, ...values)
  const padding = (rawMax - rawMin || 1) * 0.12
  const min = rawMin - padding
  const max = rawMax + padding
  const chartWidth = WIDTH - PLOT.left - PLOT.right
  const chartHeight = HEIGHT - PLOT.top - PLOT.bottom
  const x = (index: number) => PLOT.left + (index + 0.5) * chartWidth / METRICS.length
  const y = (value: number) => PLOT.top + (max - value) * chartHeight / (max - min)
  const zero = y(0)
  const groupWidth = chartWidth / METRICS.length
  const active = hover == null ? null : METRICS[hover]
  return <article className="chart-card"><div className="chart-heading"><div><h3>模型指标对比</h3><p>指标差值 = 新模型 − 当前生产模型</p></div><div className="chart-legend"><span><i style={{ backgroundColor: COLORS.baseline }} />{production.version}</span><span><i style={{ backgroundColor: COLORS.candidate }} />{candidate.version}</span></div></div><div className="chart-visual"><svg aria-label="当前生产模型与本次新模型指标对比" role="img" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} onMouseLeave={() => setHover(null)}>
    {[max, max - (max - min) / 2, min].map((tick) => <g key={tick}><line className="chart-grid-line" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={y(tick)} y2={y(tick)} /><text className="chart-axis-label" x={PLOT.left - 8} y={y(tick) + 3} textAnchor="end">{formatNumber(tick, 2)}</text></g>)}
    <line className="chart-axis-line" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={zero} y2={zero} />
    {METRICS.map((metric, index) => <g key={metric} onMouseEnter={() => setHover(index)} onFocus={() => setHover(index)} tabIndex={0}>
      {finite(production.metrics[metric]) && <rect className="metric-bar" fill={COLORS.baseline} x={x(index) - groupWidth * 0.27} y={Math.min(zero, y(production.metrics[metric] as number))} width={groupWidth * 0.22} height={Math.abs(zero - y(production.metrics[metric] as number))} rx={3}><title>{`${production.version} · ${metricLabel(metric)} · ${formatNumber(production.metrics[metric])}`}</title></rect>}
      {finite(candidate.metrics[metric]) && <rect className="metric-bar" fill={COLORS.candidate} x={x(index) + groupWidth * 0.05} y={Math.min(zero, y(candidate.metrics[metric] as number))} width={groupWidth * 0.22} height={Math.abs(zero - y(candidate.metrics[metric] as number))} rx={3}><title>{`${candidate.version} · ${metricLabel(metric)} · ${formatNumber(candidate.metrics[metric])}`}</title></rect>}
      <text className="chart-time-label" x={x(index)} y={HEIGHT - 14} textAnchor="middle">{metricLabel(metric)}</text>
    </g>)}
    {active && <line className="chart-hover-line" x1={x(METRICS.indexOf(active))} x2={x(METRICS.indexOf(active))} y1={PLOT.top} y2={HEIGHT - PLOT.bottom} />}
  </svg>{active && <div className="chart-tooltip metric-tooltip" style={{ left: `${Math.min(85, Math.max(15, x(METRICS.indexOf(active)) / WIDTH * 100))}%` }}><b>{metricLabel(active)}</b><span><i style={{ backgroundColor: COLORS.baseline }} />{production.version}：{formatNumber(production.metrics[active])}</span><span><i style={{ backgroundColor: COLORS.candidate }} />{candidate.version}：{formatNumber(candidate.metrics[active])}</span><small>变化：{formatDelta(metricDelta(candidate.metrics[active], production.metrics[active]), active)}</small></div>}</div></article>
}

function formatDelta(delta: number | null, metric: MetricName): string {
  if (delta === null) return '暂无'
  return `${delta > 0 ? '+' : ''}${formatNumber(delta)}${metric === 'r2' ? '' : ''}`
}

function metricCardValue(value: number | null | undefined): string {
  return finite(value) ? formatNumber(value) : '暂无'
}

function comparisonRows(candidate: EvaluationModelResult | null, production: EvaluationModelResult | null) {
  return METRICS.map((metric) => ({
    metric,
    candidate: candidate?.metrics[metric] ?? null,
    production: production?.metrics[metric] ?? null,
    delta: metricDelta(candidate?.metrics[metric], production?.metrics[metric]),
  }))
}

export function EvaluationDashboard({ evaluation }: { evaluation: ModelEvaluation }) {
  const data = useMemo(() => normalizeEvaluationData(evaluation), [evaluation])
  const metrics = evaluationMetrics(evaluation)
  const { candidate, production } = useMemo(() => evaluationModels(evaluation), [evaluation])
  const rows = comparisonRows(candidate, production)
  const mapeValidCount = metrics.mape_valid_count
  const mapeNote = metrics.mape_note
  const lineSeries: LineSeries[] = [
    { key: 'actual', label: '实际值', color: COLORS.actual, values: data.points.map((point) => point.actual) },
    { key: 'candidate', label: '新模型预测', color: COLORS.candidate, values: data.points.map((point) => point.candidate) },
  ]
  if (data.points.some((point) => point.baseline !== null)) lineSeries.push({ key: 'baseline', label: '生产模型预测', color: COLORS.baseline, values: data.points.map((point) => point.baseline) })
  const errorSeries: LineSeries[] = [{ key: 'error', label: '预测误差', color: COLORS.error, values: data.points.map((point) => point.error) }]
  const chartNote = data.points.length === 0
    ? data.sourceCount > 0 ? `后端返回 ${data.sourceCount.toLocaleString()} 条测试数据，但暂无可用图表点位` : '暂无图表数据'
    : data.serverSampled || data.clientSampled
      ? `图表显示 ${data.points.length.toLocaleString()} / ${data.sourceCount.toLocaleString()} 个时间点${data.serverSampled ? '（后端已抽样' : '（前端已抽样'}${data.clientSampled ? '，前端再次抽样）' : '）'}`
      : `图表显示全部 ${data.points.length.toLocaleString()} 个时间点`
  return <div className="evaluation-dashboard">
    <div className="metric-grid evaluation-metrics" aria-label="测试集评估指标">{METRICS.map((metric) => <div className="metric-card" key={metric}><small>{metricLabel(metric)}</small><strong>{metricCardValue(metrics[metric])}</strong><span>{metric === 'r2' ? '越高越好' : '越低越好'}</span></div>)}</div>
    <div className="evaluation-meta"><span>指标样本：{finite(metrics.sample_count) ? metrics.sample_count.toLocaleString() : '—'}（后端完整测试集）</span>{finite(mapeValidCount) && <span>MAPE 有效样本：{mapeValidCount.toLocaleString()}</span>}{mapeNote && <span>{mapeNote}</span>}</div>
    <section className="comparison-section"><div className="section-heading compact-heading"><div><h3>生产模型与本次新模型</h3><p>变化值按“新模型 − 生产模型”计算</p></div></div>{!production && <InfoBox tone="warning">暂无当前生产模型对比数据，本次新模型指标仍可查看；发布后可用于后续对比。</InfoBox>}<div className="table-wrap comparison-table-wrap"><table className="comparison-table"><thead><tr><th>指标</th><th>{production?.version ?? '当前生产模型'}<small>当前生产</small></th><th>{candidate?.version ?? '本次新模型'}<small>本次候选</small></th><th>变化 / 方向</th></tr></thead><tbody>{rows.map((row) => <tr key={row.metric}><th>{metricLabel(row.metric)}</th><td>{metricCardValue(row.production)}</td><td>{metricCardValue(row.candidate)}</td><td className={row.delta == null ? '' : row.delta === 0 ? 'delta-neutral' : ((row.metric === 'r2' ? row.delta > 0 : row.delta < 0) ? 'delta-good' : 'delta-bad')}>{formatDelta(row.delta, row.metric)}{row.delta != null && <small>{row.metric === 'r2' ? row.delta > 0 ? ' ↑' : row.delta < 0 ? ' ↓' : ' →' : row.delta < 0 ? ' ↓' : row.delta > 0 ? ' ↑' : ' →'}</small>}</td></tr>)}</tbody></table></div>{(production?.metrics.mape_note || mapeNote) && <p className="comparison-note">MAPE 说明：{production?.metrics.mape_note ?? mapeNote}</p>}</section>
    <div className="charts-grid"><LineChart title="测试集实际值 / 预测值" description="横轴为测试集时间，悬停或聚焦数据点查看详情" points={data.points} series={lineSeries} /><LineChart title="预测误差" description="误差 = 实际值 − 新模型预测值" points={data.points} series={errorSeries} error /><MetricComparisonChart candidate={candidate} production={production} /></div>
    <div className="chart-data-note"><span>{chartNote}</span>{data.invalidCount > 0 && <span> · 已忽略 {data.invalidCount} 条无效图表记录；指标未受影响</span>}</div>
  </div>
}

function InfoBox({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warning' }) {
  return <div className={`alert-box ${tone}`} role="status">{children}</div>
}
