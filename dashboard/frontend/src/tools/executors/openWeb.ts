/**
 * Open Web Tool Executor
 * Web content extraction tool executor
 * 
 * Fetches web content through the dashboard backend proxy
 * - Avoids TLS/CORS issues from browser-side requests to third-party services
 * - Preserves the existing open_web argument contract
 * - Keeps the response shape stable
 */

import { createTool } from '../registry'
import type { ToolExecutionContext, OpenWebArgs, OpenWebResult } from '../types'

// Re-export types for external use
export type { OpenWebArgs, OpenWebResult }

// ========== Constants ==========

/** Default maximum content length in characters */
const DEFAULT_MAX_LENGTH = 15000

/** Default timeout in seconds */
const DEFAULT_TIMEOUT = 30

// ========== Helper Functions ==========

/**
 * Validate URL format
 * Mirrors Python: _validate_url
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
 * Truncate content
 * Mirrors Python: _truncate_content
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

interface OpenWebProxyResponse {
  url?: string
  title?: string
  content?: string
  length?: number
  truncated?: boolean
  error?: string
}

function isPdfUrl(url: string): boolean {
  try {
    return new URL(url).pathname.toLowerCase().endsWith('.pdf')
  } catch {
    return false
  }
}

/**
 * Fetch a URL through the dashboard backend proxy
 */
async function fetchUrl(
  url: string,
  outputFormat: string,
  maxLength: number | null,
  withImages: boolean,
  context: ToolExecutionContext
): Promise<OpenWebResult> {
  validateUrl(url)
  const startTime = Date.now()

  console.log(`[OpenWeb] Starting fetch: ${url}`)
  console.log(`[OpenWeb] Output format: ${outputFormat}, include images: ${withImages}`)

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT * 1000)

  try {
    const response = await fetch('/api/tools/open-web', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...context.headers,
      },
      body: JSON.stringify({
        url,
        timeout: DEFAULT_TIMEOUT,
        // Prefer direct server-side fetches for ordinary pages, but preserve
        // Jina-only paths for PDFs and image-alt enrichment.
        force_jina: withImages || isPdfUrl(url),
        format: outputFormat,
        max_length: maxLength,
        with_images: withImages,
      }),
      signal: context.signal || controller.signal,
    })

    clearTimeout(timeoutId)
    const responseTime = Date.now() - startTime

    console.log(`[OpenWeb] Response status: ${response.status} ${response.statusText}`)
    console.log(`[OpenWeb] Response time: ${responseTime}ms`)

    let payload: OpenWebProxyResponse | null = null
    const contentType = response.headers.get('content-type') || ''

    if (contentType.includes('application/json')) {
      const parsedPayload = await response.json()
      payload = parsedPayload as OpenWebProxyResponse
    } else {
      const text = await response.text()
      payload = { error: text || response.statusText }
    }

    if (!response.ok) {
      throw new Error(payload?.error || `HTTP ${response.status}: ${response.statusText}`)
    }

    if (payload?.error) {
      throw new Error(payload.error)
    }

    const rawContent = typeof payload?.content === 'string' ? payload.content : ''
    const truncationResult = truncateContent(rawContent, maxLength)
    const content = truncationResult.content
    const truncated = Boolean(payload?.truncated) || truncationResult.truncated
    const title = payload?.title || 'Untitled'
    const resolvedUrl = payload?.url || url

    console.log(`[OpenWeb] Resolved title: ${title}`)
    console.log(`[OpenWeb] Content length: ${content.length} chars`)
    if (truncated) {
      console.log(`[OpenWeb] Content truncated`)
    }
    console.log(`[OpenWeb] Success, total time: ${Date.now() - startTime}ms`)

    return {
      url: resolvedUrl,
      title,
      content,
      length: content.length,
      truncated,
    }
  } catch (error) {
    const elapsed = Date.now() - startTime
    if (error instanceof Error && error.name === 'AbortError') {
      console.error(`[OpenWeb] Request timed out (${DEFAULT_TIMEOUT}s)`)
      throw new Error(`Error fetching URL (${url}): Request timeout`)
    }
    console.error(`[OpenWeb] Fetch failed (${elapsed}ms):`, error)
    throw new Error(`Error fetching URL (${url}): ${error instanceof Error ? error.message : String(error)}`)
  } finally {
    clearTimeout(timeoutId)
  }
}

// ========== Validation ==========

/**
 * Validate open web arguments
 * Mirrors Python: jina_fetch argument validation
 */
function validateOpenWebArgs(args: unknown): OpenWebArgs {
  if (typeof args !== 'object' || args === null) {
    throw new Error('Arguments must be an object')
  }

  const { url, format, max_length, with_images } = args as Record<string, unknown>

  // Validate required url
  if (!url) {
    throw new Error('Missing required parameter: url')
  }
  if (typeof url !== 'string' || !url.trim()) {
    throw new Error('url must be a non-empty string')
  }

  // Validate format
  let parsedFormat: 'markdown' | 'json' = 'markdown'
  if (format !== undefined) {
    const fmt = String(format).toLowerCase()
    if (fmt !== 'markdown' && fmt !== 'json') {
      throw new Error("Format must be either 'markdown' or 'json'")
    }
    parsedFormat = fmt as 'markdown' | 'json'
  }

  // Validate max_length
  let parsedMaxLength: number | undefined = DEFAULT_MAX_LENGTH
  if (max_length !== undefined && max_length !== null) {
    const len = typeof max_length === 'number' ? max_length : parseInt(String(max_length), 10)
    if (isNaN(len) || len <= 0) {
      throw new Error('max_length must be a positive integer')
    }
    parsedMaxLength = len
  }

  // Validate with_images
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
 * Mirrors Python: jina_fetch
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
  console.log(`[OpenWeb] Starting web fetch`)
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
 * Mirrors Python: @mcp.tool() jina_fetch
 */
export const openWebTool = createTool<OpenWebArgs, OpenWebResult>({
  metadata: {
    id: 'open_web',
    displayName: 'Open Web Page',
    category: 'search',
    icon: 'globe',
    enabled: true,
    version: '2.1.0',
  },

  definition: {
    type: 'function',
    function: {
      name: 'open_web',
      description:
        'Fetch a URL through the dashboard backend proxy and return extracted page content. Supports HTML and PDF extraction.',
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
