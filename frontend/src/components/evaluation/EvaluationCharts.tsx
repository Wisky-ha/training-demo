import { useMemo, useState } from 'react'
import type { MouseEvent, ReactNode } from 'react'
import type { MetricName, ModelEvaluation } from '../../types/contracts'
import {
  evaluationMetrics,
  evaluationModels,
  metricDelta,
  metricLabel,
  normalizeEvaluationData,
  normalizeEvaluationErrorData,
  type EvaluationModelResult,
  type NormalizedEvaluationPoint,
} from '../../utils/evaluation'

const WIDTH = 760
const HEIGHT = 280
const PLOT = { left: 52, right: 18, top: 20, bottom: 42 }
const COLORS = { actual: '#252b2f', candidate: '#1267bd', baseline: '#7f8a91', error: '#1267bd' }
const METRICS: MetricName[] = ['mae', 'rmse', 'mape', 'r2']
const SUPERSCRIPT_DIGITS: Record<string, string> = {
  '-': '⁻',
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
}

type LineSeries = { key: string; label: string; color: string; values: Array<number | null> }

function finite(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value)
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (!finite(value)) return '暂无'
  if (Math.abs(value) >= 10000) {
    const [mantissa, rawExponent] = value.toExponential(Math.min(3, Math.max(0, digits))).split('e')
    const exponent = String(Number(rawExponent)).split('').map((character) => SUPERSCRIPT_DIGITS[character] ?? character).join('')
    return `${mantissa} ×10${exponent}`
  }
  return value.toFixed(digits)
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
  const tooltipPosition = activeX / WIDTH < 0.34 ? 'start' : activeX / WIDTH > 0.66 ? 'end' : 'center'
  return <article className="chart-card"><div className="chart-heading"><div><h3>{title}</h3><p>{description}</p></div><ChartLegend series={series} /></div><div className="chart-visual"><svg aria-label={title} role="img" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} onMouseLeave={() => setHover(null)}>
    <desc>{description}</desc>
    {ticks.map((tick) => <g key={tick}><line className="chart-grid-line" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={y(tick)} y2={y(tick)} /><text className="chart-axis-label" x={PLOT.left - 8} y={y(tick) + 3} textAnchor="end">{formatNumber(tick, 2)}</text></g>)}
    <line className="chart-axis-line" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={HEIGHT - PLOT.bottom} y2={HEIGHT - PLOT.bottom} />
    {series.map((item) => <path className="chart-line" d={pathFor(item.values, y, x)} key={item.key} style={{ stroke: item.color }} />)}
    {hover != null && <line className="chart-hover-line" x1={activeX} x2={activeX} y1={PLOT.top} y2={HEIGHT - PLOT.bottom} />}
    {series.map((item) => item.values.map((value, index) => finite(value) && (hover === index || points.length < 80) ? <circle className="chart-point" cx={x(index)} cy={y(value)} fill={item.color} key={`${item.key}-${index}`} onFocus={() => setHover(index)} onMouseEnter={() => setHover(index)} r={hover === index ? 4 : 2.2} tabIndex={0}><title>{`${item.label} · ${formatTime(points[index].timestamp)} · ${formatNumber(value)}`}</title></circle> : null))}
    {labels.map((label, index) => <text className="chart-time-label" key={`${label}-${index}`} x={[PLOT.left, WIDTH / 2, WIDTH - PLOT.right][index]} y={HEIGHT - 14} textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{formatAxisTime(label)}</text>)}
    <rect aria-label="在图表上查看时间点详情" className="chart-hover-target" x={PLOT.left} y={PLOT.top} width={innerWidth} height={innerHeight} onMouseMove={setHoverFromEvent} onMouseLeave={() => setHover(null)} />
  </svg>{active && <div className={`chart-tooltip chart-tooltip-${tooltipPosition}`}><b>{formatTime(active.timestamp)}</b>{series.map((item) => <span key={item.key}><i style={{ backgroundColor: item.color }} />{item.label}：{formatNumber(item.values[hover ?? 0])}</span>)}{error && <small>误差 = 实际值 − 预测值</small>}</div>}</div></article>
}

function MetricComparisonChart({ candidate, production }: { candidate: EvaluationModelResult | null; production: EvaluationModelResult | null }) {
  const [hover, setHover] = useState<MetricName | null>(null)
  if (!candidate || !production) return <article className="chart-card"><div className="chart-heading"><div><h3>模型指标对比</h3><p>当前生产模型数据由后端提供时展示</p></div></div><ChartEmpty message="暂无当前生产模型对比数据" /></article>
  const values = METRICS.flatMap((metric) => [candidate.metrics[metric], production.metrics[metric]]).filter(finite)
  if (!values.length) return <article className="chart-card"><div className="chart-heading"><div><h3>模型指标对比</h3><p>当前生产模型与本次新模型</p></div></div><ChartEmpty message="暂无可绘制的指标数据" /></article>

  const activeCandidate = hover ? candidate.metrics[hover] : null
  const activeProduction = hover ? production.metrics[hover] : null
  return <article className="chart-card"><div className="chart-heading"><div><h3>模型指标对比</h3><p>横向条长只在同一指标内归一比较</p></div></div><div className="metric-comparison-bars" role="img" aria-label="当前生产模型与本次新模型横向指标对比" onMouseLeave={() => setHover(null)}>
    <div className="metric-bar-axis" aria-hidden="true"><span /><span><i>0</i><i>同指标较大绝对值</i></span><span /></div>
    {METRICS.flatMap((metric) => {
      const candidateValue = candidate.metrics[metric]
      const productionValue = production.metrics[metric]
      const maxMagnitude = Math.max(
        finite(candidateValue) ? Math.abs(candidateValue) : 0,
        finite(productionValue) ? Math.abs(productionValue) : 0,
      )
      const barWidth = (value: number | null | undefined) => finite(value) && maxMagnitude > 0
        ? Math.max(1.5, Math.abs(value) / maxMagnitude * 100)
        : 0
      const candidateLabel = `${candidate.version} · ${metricLabel(metric)} · ${metricCardValue(candidateValue, metric)}`
      const productionLabel = `${production.version} · ${metricLabel(metric)} · ${metricCardValue(productionValue, metric)}`
      return [
        <div
          aria-label={`${candidateLabel}（新模型）`}
          className="metric-bar-row"
          key={`${metric}-candidate`}
          onFocus={() => setHover(metric)}
          onMouseEnter={() => setHover(metric)}
          tabIndex={0}
          title={candidateLabel}
        >
          <span className="metric-bar-label">{metricLabel(metric)} / 新</span><span className="metric-bar-track"><i style={{ backgroundColor: COLORS.candidate, width: `${barWidth(candidateValue)}%` }} /></span><span className="metric-bar-value">{metricCardValue(candidateValue, metric)}</span>
        </div>,
        <div
          aria-label={`${productionLabel}（生产模型）`}
          className="metric-bar-row"
          key={`${metric}-production`}
          onFocus={() => setHover(metric)}
          onMouseEnter={() => setHover(metric)}
          tabIndex={0}
          title={productionLabel}
        >
          <span className="metric-bar-label">{metricLabel(metric)} / 产</span><span className="metric-bar-track"><i className="baseline" style={{ backgroundColor: COLORS.baseline, width: `${barWidth(productionValue)}%` }} /></span><span className="metric-bar-value">{metricCardValue(productionValue, metric)}</span>
        </div>,
      ]
    })}
    {hover && <div className="metric-chart-tooltip" role="status"><b>{metricLabel(hover)}</b><span><i style={{ backgroundColor: COLORS.candidate }} />{candidate.version}：{metricCardValue(activeCandidate, hover)}</span><span><i style={{ backgroundColor: COLORS.baseline }} />{production.version}：{metricCardValue(activeProduction, hover)}</span><small>变化：{formatDelta(metricDelta(activeCandidate, activeProduction), hover)}</small></div>}
  </div></article>
}

function formatDelta(delta: number | null, metric: MetricName): string {
  if (delta === null) return '暂无'
  return `${delta > 0 ? '+' : ''}${metricCardValue(delta, metric)}`
}

function metricCardValue(value: number | null | undefined, metric?: MetricName): string {
  if (!finite(value)) return '暂无'
  return `${formatNumber(value)}${metric === 'mape' ? '%' : ''}`
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
  const errorData = useMemo(() => normalizeEvaluationErrorData(evaluation), [evaluation])
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
  const errorSeries: LineSeries[] = [{ key: 'error', label: '预测误差', color: COLORS.error, values: errorData.points.map((point) => point.error) }]
  const chartNote = data.points.length === 0
    ? data.sourceCount > 0 ? `后端返回 ${data.sourceCount.toLocaleString()} 条测试数据，但暂无可用图表点位` : '暂无图表数据'
    : data.serverSampled || data.clientSampled
      ? `图表显示 ${data.points.length.toLocaleString()} / ${data.sourceCount.toLocaleString()} 个时间点${data.serverSampled ? '（后端已抽样' : '（前端已抽样'}${data.clientSampled ? '，前端再次抽样）' : '）'}`
      : `图表显示全部 ${data.points.length.toLocaleString()} 个时间点`
  return <div className="evaluation-dashboard">
    <div className="metric-grid evaluation-metrics" aria-label="测试集评估指标">{METRICS.map((metric) => <div className="metric-card" key={metric}><small>{metricLabel(metric)}</small><strong>{metricCardValue(metrics[metric], metric)}</strong><span>{metric === 'r2' ? '越高越好' : '越低越好'}</span></div>)}</div>
    <div className="evaluation-meta"><span>指标样本：{finite(metrics.sample_count) ? metrics.sample_count.toLocaleString() : '未知'}</span>{finite(mapeValidCount) && <span>MAPE 有效样本：{mapeValidCount.toLocaleString()}</span>}{finite(metrics.mape_excluded_count) && <span>MAPE 排除样本：{metrics.mape_excluded_count.toLocaleString()}</span>}{mapeNote && <span>{mapeNote}</span>}</div>
    <section className="comparison-section"><div className="section-heading compact-heading"><div><h3>生产模型与本次新模型</h3><p>变化值按“新模型 − 生产模型”计算</p></div></div>{!production && <InfoBox tone="warning">暂无当前生产模型对比数据，本次新模型指标仍可查看；发布后可用于后续对比。</InfoBox>}<div className="table-wrap comparison-table-wrap"><table className="comparison-table"><thead><tr><th>指标</th><th>{production?.version ?? '当前生产模型'}<small>当前生产</small></th><th>{candidate?.version ?? '本次新模型'}<small>本次候选</small></th><th>变化 / 方向</th></tr></thead><tbody>{rows.map((row) => <tr key={row.metric}><th>{metricLabel(row.metric)}</th><td>{metricCardValue(row.production, row.metric)}</td><td>{metricCardValue(row.candidate, row.metric)}</td><td className={row.delta == null ? '' : row.delta === 0 ? 'delta-neutral' : ((row.metric === 'r2' ? row.delta > 0 : row.delta < 0) ? 'delta-good' : 'delta-bad')}>{formatDelta(row.delta, row.metric)}{row.delta != null && <small>{row.metric === 'r2' ? row.delta > 0 ? ' ↑' : row.delta < 0 ? ' ↓' : ' →' : row.delta < 0 ? ' ↓' : row.delta > 0 ? ' ↑' : ' →'}</small>}</td></tr>)}</tbody></table></div>{(production?.metrics.mape_note || mapeNote) && <p className="comparison-note">MAPE 说明：{production?.metrics.mape_note ?? mapeNote}</p>}</section>
    <div className="charts-grid"><LineChart title="测试集实际值 / 预测值" description="横轴为测试集时间，悬停或聚焦数据点查看详情" points={data.points} series={lineSeries} /><LineChart title="预测误差" description="误差 = 实际值 − 新模型预测值" points={errorData.points} series={errorSeries} error /><MetricComparisonChart candidate={candidate} production={production} /></div>
    <div className="chart-data-note"><span>{chartNote}</span>{data.invalidCount > 0 && <span> · 已忽略 {data.invalidCount} 条无效图表记录；指标未受影响</span>}</div>
  </div>
}

function InfoBox({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warning' }) {
  return <div className={`alert-box ${tone}`} role="status">{children}</div>
}
