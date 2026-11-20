import { responseErrorMessage } from '../pages/configPageRequestErrors'
import type { BuiltInTool } from './mcpConfigPanelTypes'

export {
  createLatestMCPRequestRunner,
  getMCPRequestErrorMessage,
  isMCPAbortError,
  type LatestMCPRequestRunner,
} from '../tools/mcp/requestSupport'

async function toolsDbErrorMessage(response: Response): Promise<string> {
  const statusText = response.statusText.trim()
  const fallback = statusText
    ? `Failed to load the tools database (${response.status} ${statusText})`
    : `Failed to load the tools database (${response.status})`
  try {
    const reason = await responseErrorMessage(response)
    if (typeof reason !== 'string') return fallback
    const trimmed = reason.trim()
    // responseErrorMessage reports an empty body with its own status line.
    if (!trimmed || trimmed === `HTTP ${response.status}: ${statusText}`.trim()) return fallback
    // Plain text only: refuse markup from proxies and raw JSON that had no
    // displayable message or error fields.
    if (/^[<[{]/.test(trimmed)) return fallback
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
