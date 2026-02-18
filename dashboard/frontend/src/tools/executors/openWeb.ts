/**
 * Open Web Tool Executor
 * 网页内容提取工具执行器
 * 
 * 参考 DuckDuckGo MCP Server 的 jina_fetch.py 实现
 * - 使用 Jina Reader API (r.jina.ai) 抓取网页
 * - 支持 markdown/json 输出格式
 * - 支持内容长度限制
 * - 支持图片 alt 文本生成
 */

import { createTool } from '../registry'
import type { ToolExecutionContext, OpenWebArgs, OpenWebResult } from '../types'

// Re-export types for external use
export type { OpenWebArgs, OpenWebResult }

// ========== Constants ==========

/** Jina Reader API base URL */
const JINA_READER_BASE_URL = 'https://r.jina.ai/'

/** 默认最大内容长度（字符） */
const DEFAULT_MAX_LENGTH = 15000

/** 默认超时时间（秒） */
const DEFAULT_TIMEOUT = 30

// ========== Helper Functions ==========

/**
 * 验证 URL 格式
 * 对应 Python: _validate_url
 */
function validateUrl(url: string): void {
  if (!url || typeof url !== 'string') {
    throw new Error('URL must be a non-empty string')
  }

  try {
    const parsedUrl = new URL(url)
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      throw new Error('Only HTTP/HTTPS URLs are supported')
    }
  } catch (e) {
    if (e instanceof Error && e.message.includes('Only HTTP/HTTPS')) {
      throw e
    }
    throw new Error('Invalid URL format')
  }
}

/**
 * 构建请求头
 * 对应 Python: _build_headers
 */
function buildHeaders(outputFormat: string, withImages: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'x-no-cache': 'true',
  }

  if (outputFormat.toLowerCase() === 'json') {
    headers['Accept'] = 'application/json'
  } else if (outputFormat.toLowerCase() !== 'markdown') {
    console.warn(`[OpenWeb] Unsupported format: ${outputFormat}. Using markdown as default.`)
  }

  if (withImages) {
    headers['X-With-Generated-Alt'] = 'true'
  }

  return headers
}

/**
 * 截断内容
 * 对应 Python: _truncate_content
 */
function truncateContent(content: string, maxLength: number | null): { content: string; truncated: boolean } {
  if (maxLength && content.length > maxLength) {
    return {
      content: content.substring(0, maxLength) + '... (content truncated)',
      truncated: true,
    }
  }
  return { content, truncated: false }
}

/**
 * 处理响应数据
 * 对应 Python: _process_response
 */
function processResponse(
  data: unknown,
  outputFormat: string,
  maxLength: number | null
): { title: string; content: string; truncated: boolean } {
  if (outputFormat.toLowerCase() === 'json') {
    const jsonData = data as Record<string, unknown>
    let content = String(jsonData.content || '')
    let truncated = false

    if (maxLength && content.length > maxLength) {
      const result = truncateContent(content, maxLength)
      content = result.content
      truncated = result.truncated
    }

    return {
      title: String(jsonData.title || 'Untitled'),
      content,
      truncated,
    }
  }

  // markdown 格式
  const text = typeof data === 'string' ? data : String(data)
  const { content, truncated } = truncateContent(text, maxLength)
  
  // 尝试从 markdown 中提取标题
  const titleMatch = text.match(/^#\s+(.+)$/m)
  const title = titleMatch ? titleMatch[1] : 'Untitled'

  return { title, content, truncated }
}

/**
 * 使用 Jina Reader 抓取 URL
 * 对应 Python: fetch_url
 */
async function fetchUrl(
  url: string,
  outputFormat: string,
  maxLength: number | null,
  withImages: boolean,
  context: ToolExecutionContext
): Promise<OpenWebResult> {
  validateUrl(url)
  const headers = buildHeaders(outputFormat, withImages)
  // 注意：不使用 encodeURIComponent，因为 Jina Reader 接受完整的原始 URL
  // Python 版本使用 quote(url)，默认 safe='/'，也不会编码 URL 中的斜杠
  const jinaUrl = `${JINA_READER_BASE_URL}${url}`
  const startTime = Date.now()

  console.log(`[OpenWeb] 开始抓取: ${url}`)
  console.log(`[OpenWeb] Jina URL: ${jinaUrl}`)
  console.log(`[OpenWeb] 输出格式: ${outputFormat}, 包含图片: ${withImages}`)

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT * 1000)

  try {
    const response = await fetch(jinaUrl, {
      method: 'GET',
      headers: {
        ...headers,
        ...context.headers,
      },
      signal: context.signal || controller.signal,
    })

    clearTimeout(timeoutId)
    const responseTime = Date.now() - startTime

    console.log(`[OpenWeb] 响应状态: ${response.status} ${response.statusText}`)
    console.log(`[OpenWeb] 响应耗时: ${responseTime}ms`)

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    let data: unknown
    if (outputFormat.toLowerCase() === 'json') {
      data = await response.json()
    } else {
      data = await response.text()
    }

    const { title, content, truncated } = processResponse(data, outputFormat, maxLength)

    console.log(`[OpenWeb] 获取标题: ${title}`)
    console.log(`[OpenWeb] 内容长度: ${content.length} 字符`)
    if (truncated) {
      console.log(`[OpenWeb] 内容已截断`)
    }
    console.log(`[OpenWeb] ✅ 抓取成功，总耗时: ${Date.now() - startTime}ms`)

    return {
      url,
      title,
      content,
      length: content.length,
      truncated,
    }
  } catch (error) {
    const elapsed = Date.now() - startTime
    if (error instanceof Error && error.name === 'AbortError') {
      console.error(`[OpenWeb] ❌ 请求超时 (${DEFAULT_TIMEOUT}s)`)
      throw new Error(`Error fetching URL (${url}): Request timeout`)
    }
    console.error(`[OpenWeb] ❌ 抓取失败 (${elapsed}ms):`, error)
    throw new Error(`Error fetching URL (${url}): ${error instanceof Error ? error.message : String(error)}`)
  } finally {
    clearTimeout(timeoutId)
  }
}

// ========== Validation ==========

/**
 * Validate open web arguments
 * 对应 Python: jina_fetch 的参数验证
 */
function validateOpenWebArgs(args: unknown): OpenWebArgs {
  if (typeof args !== 'object' || args === null) {
    throw new Error('Arguments must be an object')
  }

  const { url, format, max_length, with_images } = args as Record<string, unknown>

  // 验证 url（必需）
  if (!url) {
    throw new Error('Missing required parameter: url')
  }
  if (typeof url !== 'string' || !url.trim()) {
    throw new Error('url must be a non-empty string')
  }

  // 验证 format
  let parsedFormat: 'markdown' | 'json' = 'markdown'
  if (format !== undefined) {
    const fmt = String(format).toLowerCase()
    if (fmt !== 'markdown' && fmt !== 'json') {
      throw new Error("Format must be either 'markdown' or 'json'")
    }
    parsedFormat = fmt as 'markdown' | 'json'
  }

  // 验证 max_length
  let parsedMaxLength: number | undefined = DEFAULT_MAX_LENGTH
  if (max_length !== undefined && max_length !== null) {
    const len = typeof max_length === 'number' ? max_length : parseInt(String(max_length), 10)
    if (isNaN(len) || len <= 0) {
      throw new Error('max_length must be a positive integer')
    }
    parsedMaxLength = len
  }

  // 验证 with_images
  const parsedWithImages = typeof with_images === 'boolean' ? with_images : false

  return {
    url: url.trim(),
    format: parsedFormat,
    max_length: parsedMaxLength,
    with_images: parsedWithImages,
  }
}

// ========== Executor ==========

/**
 * Execute open web
 * 对应 Python: jina_fetch
 */
async function executeOpenWeb(
  args: OpenWebArgs,
  context: ToolExecutionContext
): Promise<OpenWebResult> {
  const { 
    url, 
    format = 'markdown', 
    max_length = DEFAULT_MAX_LENGTH,
    with_images = false 
  } = args

  console.log(`\n${'='.repeat(60)}`)
  console.log(`[OpenWeb] 🚀 开始抓取网页`)
  console.log(`[OpenWeb] URL: ${url}`)
  console.log(`[OpenWeb] Format: ${format}, MaxLength: ${max_length}, WithImages: ${with_images}`)
  console.log(`${'='.repeat(60)}`)

  context.onProgress?.(10)

  const result = await fetchUrl(url, format, max_length, with_images, context)

  context.onProgress?.(100)

  console.log(`${'='.repeat(60)}\n`)

  return result
}

// ========== Result Formatting ==========

/**
 * Format open web result for display
 */
function formatOpenWebResult(result: OpenWebResult): string {
  const truncatedNote = result.truncated ? ' (truncated)' : ''
  return `# ${result.title}\n\nURL: ${result.url}\nLength: ${result.length} chars${truncatedNote}\n\n${result.content}`
}

// ========== Tool Definition ==========

/**
 * Open Web Tool Definition
 * 对应 Python: @mcp.tool() jina_fetch
 */
export const openWebTool = createTool<OpenWebArgs, OpenWebResult>({
  metadata: {
    id: 'open_web',
    displayName: 'Open Web Page',
    category: 'search',
    icon: 'globe',
    enabled: true,
    version: '2.0.0',
  },

  definition: {
    type: 'function',
    function: {
      name: 'open_web',
      description:
        'Fetch a URL and convert it to markdown or JSON using Jina Reader. Supports HTML and PDF content extraction.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'The URL to fetch and convert',
          },
          format: {
            type: 'string',
            enum: ['markdown', 'json'],
            description: "Output format - 'markdown' (default) or 'json'",
            default: 'markdown',
          },
          max_length: {
            type: 'integer',
            description: 'Maximum content length to return (default: 15000)',
            default: 15000,
          },
          with_images: {
            type: 'boolean',
            description: 'Whether to include image alt text generation (default: false)',
            default: false,
          },
        },
        required: ['url'],
      },
    },
  },

  validateArgs: validateOpenWebArgs,
  executor: executeOpenWeb,
  formatResult: formatOpenWebResult,
})
