import type {
  MCPServerConfig,
  MCPServerState,
  MCPToolDefinition,
  MCPTransportType,
} from '../tools/mcp'
import type { RegisteredTool } from '../tools'
import type {
  BuiltInTool,
  ServerFilter,
  ServerSort,
  ToolSort,
  ToolSourceFilter,
  UnifiedTool,
  UnifiedToolParameter,
} from './mcpConfigPanelTypes'

const DEFAULT_TIMEOUT_MS = 30000

export interface ServerFormValues {
  name: string
  description: string
  transport: MCPTransportType
  enabled: boolean
  command: string
  args: string[]
  url: string
  headers: Record<string, string>
  timeout: string
  autoReconnect: boolean
}

interface ParameterSchemaLike {
  type?: string
  properties?: Record<string, { type?: string; description?: string }>
  required?: string[]
}

export function buildUnifiedTools(
  servers: MCPServerState[],
  registryTools: RegisteredTool[],
  toolsDbTools: BuiltInTool[],
): UnifiedTool[] {
  const mcpTools = servers
    .filter((server) => server.status === 'connected')
    .flatMap((server) => (server.tools || []).map((tool) => toUnifiedMCPTool(server, tool)))

  const frontendTools = registryTools.map(toUnifiedRegisteredTool)

  const backendTools = toolsDbTools.map(toUnifiedBuiltInTool)

  return [...mcpTools, ...frontendTools, ...backendTools]
}

export function toUnifiedMCPTool(server: MCPServerState, tool: MCPToolDefinition): UnifiedTool {
  return {
    id: `mcp-${server.config.id}-${tool.name}`,
    name: tool.name,
    description: tool.description || '',
    source: server.config.name,
    sourceType: 'mcp',
    parameters: extractMCPToolParameters(tool),
    rawTool: tool,
  }
}

export function toUnifiedRegisteredTool(tool: RegisteredTool): UnifiedTool {
  return {
    id: `frontend-${tool.metadata.id}`,
    name: tool.metadata.displayName,
    description: tool.definition.function.description,
    source: 'Built-in',
    sourceType: 'frontend',
    parameters: extractRegisteredToolParameters(tool),
    rawTool: tool,
  }
}

export function toUnifiedBuiltInTool(tool: BuiltInTool): UnifiedTool {
  return {
    id: `backend-${tool.tool.function.name}`,
    name: tool.tool.function.name,
    description: tool.tool.function.description,
    source: 'Semantic Router',
    sourceType: 'backend',
    parameters: extractBuiltInToolParameters(tool),
    rawTool: tool,
  }
}

export function filterAndSortUnifiedTools(
  tools: UnifiedTool[],
  searchValue: string,
  sourceFilter: ToolSourceFilter,
  sort: ToolSort,
): UnifiedTool[] {
  const search = searchValue.trim().toLowerCase()
  return tools
    .filter((tool) => sourceFilter === 'all' || tool.sourceType === sourceFilter)
    .filter((tool) => {
      if (!search) return true
      return [
        tool.name,
        tool.description,
        tool.source,
        tool.sourceType,
        ...tool.parameters.flatMap((parameter) => [parameter.name, parameter.description || '']),
      ].some((value) => value.toLowerCase().includes(search))
    })
    .sort((left, right) => {
      if (sort === 'source-asc') {
        return (
          left.source.localeCompare(right.source, undefined, { sensitivity: 'base' }) ||
          left.name.localeCompare(right.name, undefined, { sensitivity: 'base' })
        )
      }
      if (sort === 'parameters-desc') {
        return (
          right.parameters.length - left.parameters.length ||
          left.name.localeCompare(right.name, undefined, { sensitivity: 'base' })
        )
      }
      return left.name.localeCompare(right.name, undefined, { sensitivity: 'base' })
    })
}

export function filterAndSortServers(
  servers: MCPServerState[],
  filter: ServerFilter,
  searchValue: string,
  sort: ServerSort,
): MCPServerState[] {
  const search = searchValue.trim().toLowerCase()
  const statusRank: Record<MCPServerState['status'], number> = {
    connected: 0,
    connecting: 1,
    error: 2,
    disconnected: 3,
  }

  return servers
    .filter((server) => {
      if (filter === 'all') return true
      if (filter === 'connected') return server.status === 'connected'
      return server.status !== 'connected'
    })
    .filter((server) => {
      if (!search) return true
      const connection = server.config.connection
      return [
        server.config.name,
        server.config.description || '',
        server.config.transport,
        server.status,
        connection.command || '',
        connection.url || '',
        ...(server.tools || []).flatMap((tool) => [tool.name, tool.description || '']),
      ].some((value) => value.toLowerCase().includes(search))
    })
    .sort((left, right) => {
      if (sort === 'status') {
        return (
          statusRank[left.status] - statusRank[right.status] ||
          left.config.name.localeCompare(right.config.name, undefined, { sensitivity: 'base' })
        )
      }
      if (sort === 'tools-desc') {
        return (
          (right.tools?.length || 0) - (left.tools?.length || 0) ||
          left.config.name.localeCompare(right.config.name, undefined, { sensitivity: 'base' })
        )
      }
      return left.config.name.localeCompare(right.config.name, undefined, { sensitivity: 'base' })
    })
}

export function paginateMCPItems<T>(items: readonly T[], page: number, pageSize: number): T[] {
  const safePage = Math.max(1, page)
  return items.slice((safePage - 1) * pageSize, safePage * pageSize)
}

export function getMCPPageCount(itemCount: number, pageSize: number): number {
  return Math.max(1, Math.ceil(itemCount / pageSize))
}

export function getMCPVisibleRange(
  itemCount: number,
  page: number,
  pageSize: number,
): { start: number; end: number } {
  if (itemCount === 0) return { start: 0, end: 0 }
  const start = (Math.max(1, page) - 1) * pageSize + 1
  return { start, end: Math.min(itemCount, start + pageSize - 1) }
}

export function getTransportLabel(transport: MCPTransportType): string {
  switch (transport) {
    case 'stdio':
      return 'Stdio'
    case 'streamable-http':
      return 'HTTP'
    default:
      return transport
  }
}

function normalizeArgs(args: readonly string[]): string[] | undefined {
  const normalized = args.map((argument) => argument.trim()).filter(Boolean)
  return normalized.length > 0 ? normalized : undefined
}

function normalizeHeaders(
  headers: Readonly<Record<string, string>>,
): Record<string, string> | undefined {
  const normalized = Object.entries(headers)
    .map(([key, value]) => [key.trim(), value.trim()] as const)
    .filter(([key, value]) => Boolean(key && value))
  return normalized.length > 0 ? Object.fromEntries(normalized) : undefined
}

export function buildServerConfig(values: ServerFormValues): Omit<MCPServerConfig, 'id'> {
  return {
    name: values.name,
    description: values.description || undefined,
    transport: values.transport,
    enabled: values.enabled,
    connection:
      values.transport === 'stdio'
        ? {
            command: values.command,
            args: normalizeArgs(values.args),
          }
        : {
            url: values.url,
            headers: normalizeHeaders(values.headers),
          },
    options: {
      timeout: parseInt(values.timeout, 10) || DEFAULT_TIMEOUT_MS,
      autoReconnect: values.autoReconnect,
    },
  }
}

export function buildTestServerConfig(
  serverId: string | undefined,
  values: ServerFormValues,
): MCPServerConfig {
  return {
    id: serverId || 'test',
    name: values.name,
    description: values.description || undefined,
    transport: values.transport,
    enabled: values.enabled,
    connection:
      values.transport === 'stdio'
        ? {
            command: values.command,
            args: normalizeArgs(values.args),
          }
        : {
            url: values.url,
            headers: normalizeHeaders(values.headers),
          },
    options: {
      timeout: parseInt(values.timeout, 10) || DEFAULT_TIMEOUT_MS,
    },
  }
}

export function extractMCPToolParameters(tool: MCPToolDefinition): UnifiedToolParameter[] {
  return extractSchemaParameters(tool.inputSchema)
}

export function extractRegisteredToolParameters(tool: RegisteredTool): UnifiedToolParameter[] {
  return extractSchemaParameters(tool.definition.function.parameters)
}

export function extractBuiltInToolParameters(tool: BuiltInTool): UnifiedToolParameter[] {
  return extractSchemaParameters(tool.tool.function.parameters)
}

function extractSchemaParameters(schema?: ParameterSchemaLike): UnifiedToolParameter[] {
  if (!schema || schema.type !== 'object') {
    return []
  }

  const properties = schema.properties || {}
  const required = schema.required || []
  return Object.entries(properties).map(([name, property]) => ({
    name,
    type: property.type || 'any',
    description: property.description,
    required: required.includes(name),
  }))
}
