/**
 * Web Search Tool Executor
 * Web 搜索工具执行器
 * 
 * 参考 DuckDuckGo MCP Server 实现
 * - 支持 safesearch 设置 (on/moderate/off)
 * - 支持 output_format (json/text)
 * - 支持后端回退机制 (duckduckgo -> brave)
 */

import { createTool } from '../registry'
import type { ToolExecutionContext, WebSearchArgs, WebSearchResult } from '../types'

// Re-export types for external use
export type { WebSearchArgs, WebSearchResult }

// ========== Helper Functions ==========

/**
 * 转换原始搜索结果为标准格式
 * 对应 Python: _format_search_result
 */
function formatSearchResult(result: Record<string, unknown>): WebSearchResult {
  return {
    title: String(result.title || ''),
    url: String(result.href || result.url || ''),
    snippet: String(result.body || result.snippet || ''),
  }
}

/**
 * 将搜索结果格式化为 LLM 友好的文本
 * 对应 Python: _format_results_as_text
 */
function formatResultsAsText(results: WebSearchResult[], query: string): string {
  if (!results || results.length === 0) {
    return (
      `No results found for '${query}'. ` +
      'This could be due to DuckDuckGo rate limiting, the query returning no matches, ' +
      'or network issues. Try rephrasing your search or try again in a few minutes.'
    )
  }

  const lines: string[] = [`Found ${results.length} search results:\n`]

  results.forEach((result, index) => {
    const position = index + 1
    lines.push(`${position}. ${result.title || 'No title'}`)
    lines.push(`   URL: ${result.url || 'No URL'}`)
    lines.push(`   Summary: ${result.snippet || 'No summary available'}`)
    lines.push('')  // Empty line between results
  })

  return lines.join('\n')
}

/**
 * 验证搜索参数并返回规范化的 safesearch 值
 * 对应 Python: _validate_search_params
 */
function validateSearchParams(
  query: string,
  maxResults: number,
  safesearch: string
): string {
  if (!query || typeof query !== 'string') {
    throw new Error('Query must be a non-empty string')
  }

  if (typeof maxResults !== 'number' || maxResults <= 0) {
    throw new Error('max_results must be a positive integer')
  }

  const validSafesearch = ['on', 'moderate', 'off']
  if (!validSafesearch.includes(safesearch)) {
    console.warn(`[WebSearch] Invalid safesearch value: '${safesearch}'. Using 'moderate' instead.`)
    return 'moderate'
  }

  return safesearch
}

/**
 * 执行搜索
 * 对应 Python: _execute_search
 */
async function executeSearch(
  query: string,
  safesearch: string,
  maxResults: number,
  backend: string,
  context: ToolExecutionContext
): Promise<WebSearchResult[]> {
  console.log(`[WebSearch] 执行搜索: query="${query}", backend="${backend}", safesearch="${safesearch}"`)
  
  const response = await fetch('/api/tools/web-search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...context.headers,
    },
    body: JSON.stringify({
      query,
      num_results: maxResults,
      safesearch,
      backend,
    }),
    signal: context.signal,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText)
    throw new Error(`Search failed: ${response.status} - ${errorText}`)
  }

  const data = await response.json()
  const results: WebSearchResult[] = (data.results || []).map(formatSearchResult)
  
  console.log(`[WebSearch] 获取到 ${results.length} 条结果`)
  return results
}

/**
 * 尝试使用 brave 后端作为回退
 * 对应 Python: _try_fallback_search
 */
async function tryFallbackSearch(
  query: string,
  safesearch: string,
  maxResults: number,
  originalError: Error,
  context: ToolExecutionContext
): Promise<WebSearchResult[]> {
  // 如果错误已经是关于后端的，不重试
  if (originalError.message.toLowerCase().includes('backend')) {
    return []
  }

  console.log('[WebSearch] 使用 brave 后端作为回退重试...')
  try {
    return await executeSearch(query, safesearch, maxResults, 'brave', context)
  } catch (e) {
    console.error('[WebSearch] 回退搜索失败:', e)
    return []
  }
}

/**
 * 主搜索函数
 * 对应 Python: search_duckduckgo
 */
async function searchDuckDuckGo(
  query: string,
  maxResults: number,
  safesearch: string,
  context: ToolExecutionContext
): Promise<WebSearchResult[]> {
  safesearch = validateSearchParams(query, maxResults, safesearch)

  try {
    return await executeSearch(query, safesearch, maxResults, 'duckduckgo', context)
  } catch (e) {
    const error = e instanceof Error ? e : new Error(String(e))
    console.error('[WebSearch] DuckDuckGo 搜索错误:', error.message)
    return await tryFallbackSearch(query, safesearch, maxResults, error, context)
  }
}

// ========== Validation ==========

/**
 * Validate web search arguments
 */
function validateWebSearchArgs(args: unknown): WebSearchArgs {
  if (typeof args !== 'object' || args === null) {
    throw new Error('Arguments must be an object')
  }

  const { query, max_results, safesearch, output_format } = args as Record<string, unknown>

  if (typeof query !== 'string' || !query.trim()) {
    throw new Error('query is required and must be a non-empty string')
  }

  // 类型强制转换（MCP 客户端可能传递字符串）
  let parsedMaxResults = 5
  if (max_results !== undefined) {
    if (typeof max_results === 'number') {
      parsedMaxResults = max_results
    } else if (typeof max_results === 'string') {
      const parsed = parseInt(max_results, 10)
      if (isNaN(parsed)) {
        throw new Error('max_results must be a valid positive integer')
      }
      parsedMaxResults = parsed
    }
  }

  // 验证 output_format
  let parsedOutputFormat: 'json' | 'text' = 'json'
  if (output_format !== undefined) {
    const format = String(output_format).toLowerCase()
    if (format !== 'json' && format !== 'text') {
      console.warn(`[WebSearch] Invalid output_format: '${output_format}'. Using 'json' instead.`)
    } else {
      parsedOutputFormat = format as 'json' | 'text'
    }
  }

  // 验证 safesearch
  let parsedSafesearch: 'on' | 'moderate' | 'off' = 'moderate'
  if (safesearch !== undefined) {
    const safe = String(safesearch).toLowerCase()
    if (safe === 'on' || safe === 'moderate' || safe === 'off') {
      parsedSafesearch = safe
    }
  }

  return {
    query: query.trim(),
    max_results: parsedMaxResults,
    safesearch: parsedSafesearch,
    output_format: parsedOutputFormat,
  }
}

// ========== Executor ==========

/**
 * Execute web search
 * 对应 Python: duckduckgo_search
 */
async function executeWebSearch(
  args: WebSearchArgs,
  context: ToolExecutionContext
): Promise<WebSearchResult[] | string> {
  const { 
    query, 
    max_results = 5, 
    safesearch = 'moderate',
    output_format = 'json' 
  } = args

  console.log(`\n${'='.repeat(60)}`)
  console.log(`[WebSearch] 🔍 开始搜索`)
  console.log(`[WebSearch] Query: ${query}`)
  console.log(`[WebSearch] MaxResults: ${max_results}, SafeSearch: ${safesearch}, Format: ${output_format}`)
  console.log(`${'='.repeat(60)}`)

  context.onProgress?.(10)

  const results = await searchDuckDuckGo(query, max_results, safesearch, context)

  context.onProgress?.(90)

  if (results.length === 0) {
    console.warn(`[WebSearch] 未找到结果: '${query}'`)
  }

  context.onProgress?.(100)

  // 根据输出格式返回
  if (output_format === 'text') {
    console.log(`[WebSearch] ✅ 搜索完成，返回文本格式`)
    return formatResultsAsText(results, query)
  }

  console.log(`[WebSearch] ✅ 搜索完成，返回 ${results.length} 条结果`)
  return results
}

// ========== Result Formatting ==========

/**
 * Format web search results for display
 */
function formatWebSearchResult(results: WebSearchResult[] | string): string {
  // 如果已经是字符串（text 格式），直接返回
  if (typeof results === 'string') {
    return results
  }

  if (!results || results.length === 0) {
    return 'No results found.'
  }

  return results
    .map(
      (r, i) =>
        `[${i + 1}] ${r.title}\n    URL: ${r.url}\n    ${r.snippet}`
    )
    .join('\n\n')
}

// ========== Tool Definition ==========

/**
 * Web Search Tool Definition
 * 对应 Python: @mcp.tool() duckduckgo_search
 */
export const webSearchTool = createTool<WebSearchArgs, WebSearchResult[] | string>({
  metadata: {
    id: 'search_web',
    displayName: 'Web Search',
    category: 'search',
    icon: 'search',
    enabled: true,
    version: '2.0.0',
  },

  definition: {
    type: 'function',
    function: {
      name: 'search_web',
      description:
        'Search the web using DuckDuckGo. Supports safe search settings and multiple output formats. Has automatic fallback to brave backend if DuckDuckGo fails.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query',
          },
          max_results: {
            type: 'integer',
            description: 'Maximum number of search results to return (default: 5)',
            default: 5,
          },
          safesearch: {
            type: 'string',
            enum: ['on', 'moderate', 'off'],
            description: "Safe search setting ('on', 'moderate', 'off'; default: 'moderate')",
            default: 'moderate',
          },
          output_format: {
            type: 'string',
            enum: ['json', 'text'],
            description: "Output format - 'json' returns list of results, 'text' returns LLM-friendly formatted string (default: 'json')",
            default: 'json',
          },
        },
        required: ['query'],
      },
    },
  },

  validateArgs: validateWebSearchArgs,
  executor: executeWebSearch,
  formatResult: formatWebSearchResult,
})
