import type { BuiltInTool } from './mcpConfigPanelTypes'

export {
  createLatestMCPRequestRunner,
  getMCPRequestErrorMessage,
  isMCPAbortError,
  type LatestMCPRequestRunner,
} from '../tools/mcp/requestSupport'

export async function fetchMCPToolsDatabase(signal: AbortSignal): Promise<BuiltInTool[]> {
  const response = await fetch('/api/tools-db', { signal })
  if (!response.ok) {
    throw new Error(`Failed to load the tools database (${response.status} ${response.statusText})`)
  }

  const data: unknown = await response.json()
  if (!Array.isArray(data)) {
    throw new Error('The tools database returned an invalid response.')
  }
  return data as BuiltInTool[]
}
