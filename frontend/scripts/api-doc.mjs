/**
 * 前端 API 接口面自动生成器（零依赖，静态解析源码）。
 *
 * 用法：仓库根目录执行 `npm --prefix frontend run api:doc`，
 *       或进入 frontend 后执行 `npm run api:doc`。
 *
 * 原理：前端没有 FastAPI 那样的运行时反射，因此从源码静态提取——
 *   - src/api/client.ts        ：方法名、HTTP 动词、URL 路径模板、入参/返回类型
 *   - src/types/contracts.ts   ：类型字段展开（入参/出参明细）
 *   - src/pages + src/components：每个接口方法是否被页面实际调用
 *
 * 输出：docs/前端API接口面.md（覆盖写入，永不手动维护）。
 * 修改 client.ts / contracts.ts / 页面后重跑一次即可刷新。
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '../..')
const srcDir = path.join(repoRoot, 'frontend/src')
const clientFile = path.join(srcDir, 'api/client.ts')
const contractsFile = path.join(srcDir, 'types/contracts.ts')
const outFile = path.join(repoRoot, 'docs/前端API接口面.md')

const GENERIC_METHODS = new Set(['constructor', 'buildUrl', 'request', 'get', 'postJson', 'postForm'])

// ---------- 通用工具 ----------

function linesOf(text) {
  return text.split('\n')
}

/** 从第 from 个字符开始，返回配平括号后第一个闭合 ')' 的下标。 */
function findClosingParen(code, from) {
  let depth = 0
  for (let i = from; i < code.length; i += 1) {
    const ch = code[i]
    if (ch === '(') depth += 1
    else if (ch === ')') {
      depth -= 1
      if (depth === 0) return i
    }
  }
  return -1
}

/** 把 `datasets/${encodeURIComponent(datasetId)}/split` 还原为可读的 datasets/{datasetId}/split。 */
function humanizeUrl(raw) {
  return raw
    .replace(/\$\{encodeURIComponent\(([^)]*)\)\}/g, '{$1}')
    .replace(/\$\{([^}]*)\}/g, '{$1}')
}

function collapse(text, max = 120) {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max)}…` : flat
}

// ---------- contracts.ts 类型库 ----------

function extractInterfaces(code) {
  const types = new Map()
  const re = /export interface\s+(\w+)\s*\{/g
  let match
  while ((match = re.exec(code))) {
    const name = match[1]
    const bodyStart = re.lastIndex
    let depth = 1
    let i = bodyStart
    while (i < code.length && depth > 0) {
      if (code[i] === '{') depth += 1
      else if (code[i] === '}') depth -= 1
      i += 1
    }
    types.set(name, { kind: 'interface', body: code.slice(bodyStart, i - 1) })
    re.lastIndex = i
  }
  return types
}

function extractTypeAliases(code) {
  const types = new Map()
  const re = /export type\s+(\w+)\s*=\s*/g
  let match
  while ((match = re.exec(code))) {
    const name = match[1]
    const start = re.lastIndex
    let i = start
    while (i < code.length) {
      const lineEnd = code.indexOf('\n', i)
      const line = code.slice(i, lineEnd === -1 ? code.length : lineEnd)
      if (/^\s*export\s/.test(line)) break
      i = lineEnd === -1 ? code.length : lineEnd + 1
      if (i > start && /;\s*$/.test(line)) break
    }
    const text = code.slice(start, i).trim()
    types.set(name, { kind: 'alias', body: text })
  }
  return types
}

/** 接口顶层字段（缩进 2 空格的行；值跨行时按括号吞并）。 */
function topFields(body) {
  const out = []
  const lines = linesOf(body)
  for (let idx = 0; idx < lines.length; idx += 1) {
    const raw = lines[idx]
    if (!raw.trim() || raw.trim().startsWith('/**') || raw.trim().startsWith('*')) continue
    const lead = (raw.match(/^\s*/) || [''])[0].length
    if (lead !== 2) continue
    const trimmed = raw.trim()
    if (trimmed.startsWith('[') || trimmed.startsWith('}') || trimmed === '') continue
    const fm = trimmed.match(/^([\w$]+)(\??):\s*(.*)$/)
    if (!fm) continue
    let value = fm[3]
    // 值跨行时合并后续行直到括号配平
    let depth = countUnclosed(value)
    let cursor = idx + 1
    while (depth > 0 && cursor < lines.length) {
      const more = lines[cursor].trim()
      value += ` ${more}`
      depth = countUnclosed(value)
      cursor += 1
    }
    out.push({ name: fm[1], optional: fm[2] === '?', type: value.replace(/,\s*$/, '').trim() })
    idx = cursor - 1
  }
  return out
}

function countUnclosed(text) {
  let depth = 0
  for (const ch of text) {
    if (ch === '{' || ch === '[') depth += 1
    else if (ch === '}' || ch === ']') depth -= 1
  }
  return Math.max(0, depth)
}

// ---------- client.ts 方法提取 ----------

function extractMethods(clientCode) {
  const clsStart = clientCode.indexOf('export class ApiClient {')
  const clsEndAnchor = clientCode.indexOf('export const apiClient')
  if (clsStart === -1) throw new Error('client.ts 中找不到 ApiClient 类')
  const clsEnd = clsEndAnchor === -1 ? clientCode.length : clsEndAnchor
  const scope = clientCode.slice(clsStart, clsEnd)

  const reMethod = /^  ([a-zA-Z]\w*)\s*\(/gm
  const found = []
  let match
  while ((match = reMethod.exec(scope))) {
    const name = match[1]
    if (GENERIC_METHODS.has(name)) continue
    const parenStart = scope.indexOf('(', match.index)
    const parenEnd = findClosingParen(scope, parenStart)
    if (parenEnd === -1) continue
    // 返回类型：闭括号之后同一行的 Promise<...>
    const afterLineEnd = scope.indexOf('\n', parenEnd)
    const tail = scope.slice(parenEnd + 1, afterLineEnd === -1 ? scope.length : afterLineEnd)
    const rm = tail.match(/:\s*(.+?)\s*\{?\s*$/)
    const retType = rm ? rm[1].trim() : ''
    // 方法体：到下一个方法定义行 / 类结束
    const bodyStart = afterLineEnd === -1 ? scope.length : afterLineEnd + 1
    const nextMethod = reMethod.lastIndex < scope.length ? scope.indexOf('\n  ', afterLineEnd) : -1
    const nextIdx = reMethod.exec ? null : null // 已由循环推进 lastIndex，不能重复 exec
    // 用循环匹配下一个方法更简单：记录当前匹配后预读
    // —— 此处直接扫描到下一个 "  name(" 行
    const bodyEndMatch = /\n  [a-zA-Z]\w*\s*\(/g
    bodyEndMatch.lastIndex = bodyStart
    const next = bodyEndMatch.exec(scope)
    const body = scope.slice(bodyStart, next ? next.index : scope.length)
    found.push({
      name,
      line: linesOf(scope.slice(0, match.index)).length + 1,
      paramsText: scope.slice(parenStart + 1, parenEnd),
      retType,
      body,
      http: '',
      url: '',
    })
  }
  return found
}

/** 从方法体识别 HTTP 动词与首个 URL 字符串（不解析泛型，定位动词后直接找引号）。 */
function detectCall(body) {
  const verbMap = { get: 'GET', postJson: 'POST', postForm: 'POST (multipart)' }
  const re = /this\.(get|postJson|postForm)\s*(?:<[^()]*>)?\s*\(/g
  let match
  while ((match = re.exec(body))) {
    const after = body.slice(re.lastIndex)
    const quoted = after.match(/^\s*([`'\"])([\s\S]*?)\1/)
    if (quoted) {
      return { http: verbMap[match[1]], url: humanizeUrl(quoted[2]) }
    }
  }
  return { http: '', url: '' }
}

/** 显示为带 /api 基址前缀的完整路径。 */
function formatUrl(url) {
  return url ? `/api/${url}` : '?'
}

// ---------- 页面调用统计 ----------

function pageUsage(methodName) {
  let hits = 0
  const where = []
  for (const dir of ['pages', 'components']) {
    const base = path.join(srcDir, dir)
    if (!fs.existsSync(base)) continue
    for (const file of walk(base)) {
      const text = fs.readFileSync(file, 'utf8')
      if (new RegExp(`apiClient\\.${methodName}\\s*\\(`).test(text)) {
        hits += 1
        where.push(`${dir}/${path.basename(file)}`)
      }
    }
  }
  return { hits, where }
}

function walk(dir) {
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...walk(full))
    else if (/\.(ts|tsx)$/.test(entry.name)) out.push(full)
  }
  return out
}

// ---------- 类型明细展示 ----------

function inlineFields(paramsText) {
  // 仅当形如 `input: { ... }` 的类型内联对象才提取字段；`= {}` 默认值不算。
  const typeStart = paramsText.match(/:\s*\{/)
  if (!typeStart) return []
  const braceIdx = paramsText.indexOf('{', typeStart.index)
  const seg = paramsText.slice(braceIdx)
  const out = []
  for (const line of linesOf(seg)) {
    const fm = line.trim().match(/^([\w$]+)(\??):\s*([^,]*?),?\s*$/)
    if (fm) out.push({ name: fm[1], optional: fm[2] === '?', type: fm[3].trim() })
  }
  return out
}

function namedParamType(paramsText) {
  // 形如 id: EntityId 的具名参数（不含内联对象）取类型名
  const names = []
  const re = /:\s*([\w.]+(?:<[^>]*>)?)(?:\s*=\s*[^,]*)?(?:,|$)/g
  let m
  while ((m = re.exec(paramsText))) names.push(m[1])
  return [...new Set(names.filter((n) => n !== 'undefined'))]
}

function fieldsMarkdown(rows) {
  if (!rows.length) return ''
  const lines = ['| 字段 | 类型 | 说明/可选 |', '|---|---|---|']
  for (const f of rows) {
    lines.push(`| \`${f.name}\`${f.optional ? '?' : ''} | \`${collapse(f.type, 160)}\` | ${f.optional ? '可选' : '必填'} |`)
  }
  return lines.join('\n')
}

// ---------- 主流程 ----------

const clientCode = fs.readFileSync(clientFile, 'utf8')
const contractsCode = fs.readFileSync(contractsFile, 'utf8')
const types = new Map([...extractInterfaces(contractsCode), ...extractTypeAliases(contractsCode)])

const methods = extractMethods(clientCode).map((m) => {
  const call = detectCall(m.body)
  const usage = pageUsage(m.name)
  return { ...m, ...call, ...usage }
})

const usedMethods = methods.filter((m) => m.hits > 0)
const unusedMethods = methods.filter((m) => m.hits === 0)

const md = []
md.push('# 前端 API 接口面（自动生成）')
md.push('')
md.push(`> 生成时间：${new Date().toISOString().slice(0, 19).replace('T', ' ')}`)
md.push('>')
md.push(`> 来源：\`frontend/src/api/client.ts\`（方法/路径）、\`frontend/src/types/contracts.ts\`（类型字段）、\`frontend/src/pages|components\`（页面使用统计）`)
md.push('>')
md.push('> 此文档由脚本覆盖生成，请勿手改。修改源码后重新执行：`npm run api:doc`（在 frontend 目录）')
md.push('')
md.push(`## 总览`)
md.push('')
md.push(`| 指标 | 数值 |`)
md.push('|---|---|')
md.push(`| 业务接口方法数 | ${methods.length} |`)
md.push(`| 被页面调用 | ${usedMethods.length} |`)
md.push(`| 零调用（仅封装未使用） | ${unusedMethods.length} |`)
md.push('')
md.push('## 一览表')
md.push('')
md.push('| # | 方法 | HTTP | 路径 | 入参（类型） | 返回类型 | 页面使用 |')
md.push('|---|---|---|---|---|---|---|')
methods.forEach((m, i) => {
  const paramsSummary = m.paramsText.includes('{')
    ? collapse(m.paramsText, 90)
    : namedParamType(m.paramsText).join(', ') || '无'
  const ret = m.retType.replace(/^Promise<(.+)>$/, '$1')
  md.push(`| ${i + 1} | \`${m.name}\` | ${m.http || '?'} | \`${formatUrl(m.url)}\` | \`${collapse(paramsSummary, 100)}\` | \`${collapse(ret, 100)}\` | ${m.hits ? `${m.hits}（${m.where.join('、')}）` : '未使用'} |`)
})
md.push('')

md.push('## 明细')
md.push('')
methods.forEach((m, i) => {
  const ret = m.retType.replace(/^Promise<(.+)>$/, '$1')
  md.push(`### ${i + 1}. ${m.name} \`${m.http || '?'} ${formatUrl(m.url)}\``)
  md.push('')
  md.push(`- 方法签名行号：client.ts L${m.line}`)
  md.push(`- 返回类型：\`${ret}\``)
  md.push(`- 页面使用：${m.hits ? `${m.hits} 处（${m.where.join('、')}）` : '未使用（页面未调用，可能为预留接口）'}`)
  md.push('')
  // 入参明细
  if (/\{\s*\n/.test(m.paramsText) || /:\s*\{/.test(m.paramsText)) {
    const fields = inlineFields(m.paramsText)
    if (fields.length) {
      md.push('**入参（内联对象字段）**')
      md.push('')
      md.push(fieldsMarkdown(fields))
      md.push('')
    } else {
      md.push(`**入参：** \`${collapse(m.paramsText, 300)}\``)
      md.push('')
    }
  } else {
    const named = namedParamType(m.paramsText)
    const expandable = named.filter((n) => types.has(n) && types.get(n).kind === 'interface')
    if (expandable.length) {
      md.push('**入参结构**')
      md.push('')
      for (const n of expandable) {
        md.push(`<details><summary><code>${n}</code>（${types.get(n).body.split('\n').filter((l) => l.trim()).length} 行）</summary>`)
        md.push('')
        md.push(fieldsMarkdown(topFields(types.get(n).body)))
        md.push('')
        md.push('</details>')
        md.push('')
      }
    } else if (named.length) {
      md.push(`**入参：** \`${named.join(', ')}\``)
      md.push('')
    }
  }
  // 出参明细
  if (types.has(ret) && types.get(ret).kind === 'interface') {
    md.push('**出参结构**')
    md.push('')
    md.push(`<details><summary><code>${ret}</code></summary>`)
    md.push('')
    md.push(fieldsMarkdown(topFields(types.get(ret).body)))
    md.push('')
    md.push('</details>')
    md.push('')
  }
})

md.push('## 零调用接口（页面未使用）')
md.push('')
if (unusedMethods.length) {
  md.push('| 方法 | HTTP | 路径 | 返回类型 |')
  md.push('|---|---|---|---|')
  for (const m of unusedMethods) {
    const ret = m.retType.replace(/^Promise<(.+)>$/, '$1')
    md.push(`| \`${m.name}\` | ${m.http || '?'} | \`${formatUrl(m.url)}\` | \`${ret}\` |`)
  }
  md.push('')
} else {
  md.push('无。')
  md.push('')
}

fs.writeFileSync(outFile, `\uFEFF${md.join('\n')}`, 'utf8')
console.log(`✅ 已生成 ${path.relative(repoRoot, outFile)}`)
console.log(`   业务方法 ${methods.length} 个：页面调用 ${usedMethods.length}，零调用 ${unusedMethods.length}`)
console.log('   重新生成：npm run api:doc（frontend 目录）')
