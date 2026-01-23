/**
 * Web Search Tool Executor
 * Web 搜索工具执行器
 */

import { createTool } from '../registry'
import type {
  WebSearchArgs,
  WebSearchResult,
  ToolExecutionContext,
} from '../types'

/**
 * Decode HTML entities in text
 */
function decodeHtmlEntities(text: string): string {
  if (!text) return text
  
  // Handle Unicode escapes like \u0026
  let decoded = text.replace(/\\u([0-9a-fA-F]{4})/g, (_, code) => 
    String.fromCharCode(parseInt(code, 16))
  )
  
  // Handle HTML entities like &#x27; &amp; &lt; etc.
  const entities: Record<string, string> = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#x27;': "'",
    '&#39;': "'",
    '&apos;': "'",
    '&#x2F;': '/',
    '&nbsp;': ' ',
  }
  
  for (const [entity, char] of Object.entries(entities)) {
    decoded = decoded.split(entity).join(char)
  }
  
  // Handle numeric entities like &#123;
  decoded = decoded.replace(/&#(\d+);/g, (_, code) => 
    String.fromCharCode(parseInt(code, 10))
  )
  
  // Handle hex entities like &#x7B;
  decoded = decoded.replace(/&#x([0-9a-fA-F]+);/g, (_, code) => 
    String.fromCharCode(parseInt(code, 16))
  )
  
  return decoded
}

/**
 * Validate web search arguments
 */
function validateWebSearchArgs(args: unknown): WebSearchArgs {
  if (typeof args !== 'object' || args === null) {
    throw new Error('Arguments must be an object')
  }

  const { query, num_results } = args as Record<string, unknown>

  if (typeof query !== 'string' || !query.trim()) {
    throw new Error('query is required and must be a non-empty string')
  }

  return {
    query: query.trim(),
    num_results: typeof num_results === 'number' ? num_results : 5,
  }
}

/**
 * Execute web search
 */
async function executeWebSearch(
  args: WebSearchArgs,
  context: ToolExecutionContext
): Promise<WebSearchResult[]> {
  const { query, num_results = 5 } = args

  context.onProgress?.(10)

  const response = await fetch('/api/tools/web-search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...context.headers,
    },
    body: JSON.stringify({ query, num_results }),
    signal: context.signal,
  })

  context.onProgress?.(80)

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`)
  }

  const data = await response.json()

  context.onProgress?.(100)

  const results: WebSearchResult[] = (data.results || []).map(
    (r: { title: string; url: string; snippet: string; domain: string }) => ({
      title: decodeHtmlEntities(r.title),
      url: r.url,
      snippet: decodeHtmlEntities(r.snippet),
      domain: r.domain,
    })
  )

  if (results.length === 0) {
    return [
      {
        title: 'No results found',
        url: '',
        snippet: `No search results found for "${query}". Please try a different query.`,
        domain: '',
      },
    ]
  }

  return results
}

/**
 * Format web search results for display
 */
function formatWebSearchResult(results: WebSearchResult[]): string {
  if (results.length === 0) {
    return 'No results found.'
  }

  return results
    .map(
      (r, i) =>
        `[${i + 1}] ${r.title}\n    URL: ${r.url}\n    ${r.snippet}`
    )
    .join('\n\n')
}

/**
 * Web Search Tool Definition
 */
export const webSearchTool = createTool<WebSearchArgs, WebSearchResult[]>({
  metadata: {
    id: 'search_web',
    displayName: 'Web Search',
    category: 'search',
    icon: 'search',
    enabled: true,
    version: '1.0.0',
  },

  definition: {
    type: 'function',
    function: {
      name: 'search_web',
      description:
        'Search the web for real-time information. Use this when you need current information, facts, news, or data that may have changed after your knowledge cutoff.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query',
          },
          num_results: {
            type: 'integer',
            description: 'Number of results to return (default: 5)',
            default: 5,
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
