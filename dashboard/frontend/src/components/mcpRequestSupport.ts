import { responseErrorMessage } from '../pages/configPageRequestErrors'
import type { BuiltInTool } from './mcpConfigPanelTypes'

export {
  createLatestMCPRequestRunner,
  getMCPRequestErrorMessage,
  isMCPAbortError,
  type LatestMCPRequestRunner,
} from '../tools/mcp/requestSupport'

async function toolsDbErrorMessage(response: Response): Promise<string> {
  const fallback = `Failed to load the tools database (${response.status} ${response.statusText})`
  try {
    const reason = await responseErrorMessage(response)
    if (typeof reason !== 'string') return fallback
    const trimmed = reason.trim()
    // responseErrorMessage reports an empty body with its own status line.
    if (!trimmed || trimmed === `HTTP ${response.status}: ${response.statusText}`) return fallback
    // http.Error bodies are plain text; refuse markup such as proxy error pages.
    if (trimmed.startsWith('<')) return fallback
    return trimmed
  } catch {
    return fallback
  }
}

export async function fetchMCPToolsDatabase(signal: AbortSignal): Promise<BuiltInTool[]> {
  const response = await fetch('/api/tools-db', { signal })
  if (!response.ok) {
    throw new Error(await toolsDbErrorMessage(response))
  }

  const data: unknown = await response.json()
  if (!Array.isArray(data)) {
    throw new Error('The tools database returned an invalid response.')
  }
  return data as BuiltInTool[]
}
