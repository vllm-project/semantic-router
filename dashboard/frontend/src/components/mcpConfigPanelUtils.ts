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
  UnifiedTool,
  UnifiedToolParameter,
} from './mcpConfigPanelTypes'

const DEFAULT_TIMEOUT_MS = 30000

interface ServerFormValues {
  name: string
  description: string
  transport: MCPTransportType
  enabled: boolean
  command: string
  args: string
  url: string
  headers: string
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
    .filter(server => server.status === 'connected')
    .flatMap(server =>
      (server.tools || []).map(tool => ({
        id: `mcp-${server.config.id}-${tool.name}`,
        name: tool.name,
        description: tool.description || '',
        source: server.config.name,
        sourceType: 'mcp' as const,
        parameters: extractMCPToolParameters(tool),
        rawTool: tool,
      })),
    )

  const frontendTools = registryTools.map(tool => ({
    id: `frontend-${tool.metadata.id}`,
    name: tool.metadata.displayName,
    description: tool.definition.function.description,
    source: 'Built-in',
    sourceType: 'frontend' as const,
    parameters: extractRegisteredToolParameters(tool),
    rawTool: tool,
  }))

  const backendTools = toolsDbTools.map(tool => ({
    id: `backend-${tool.tool.function.name}`,
    name: tool.tool.function.name,
    description: tool.tool.function.description,
    source: 'Semantic Router',
    sourceType: 'backend' as const,
    parameters: extractBuiltInToolParameters(tool),
    rawTool: tool,
  }))

  return [...mcpTools, ...frontendTools, ...backendTools]
}

export function filterUnifiedTools(tools: UnifiedTool[], searchValue: string): UnifiedTool[] {
  const search = searchValue.trim().toLowerCase()
  if (!search) {
    return tools
  }

  return tools.filter(tool =>
    tool.name.toLowerCase().includes(search) ||
    tool.description.toLowerCase().includes(search) ||
    tool.source.toLowerCase().includes(search),
  )
}

export function filterServers(
  servers: MCPServerState[],
  filter: ServerFilter,
): MCPServerState[] {
  if (filter === 'all') {
    return servers
  }
  if (filter === 'connected') {
    return servers.filter(server => server.status === 'connected')
  }
  return servers.filter(server => server.status !== 'connected')
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

export function parseHeaders(headersStr: string): Record<string, string> | undefined {
  if (!headersStr.trim()) {
    return undefined
  }

  const headers: Record<string, string> = {}
  for (const line of headersStr.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) {
      continue
    }

    const colonIndex = trimmed.indexOf(':')
    if (colonIndex === -1) {
      continue
    }

    const key = trimmed.slice(0, colonIndex).trim()
    const value = trimmed.slice(colonIndex + 1).trim()
    if (key && value) {
      headers[key] = value
    }
  }

  return Object.keys(headers).length > 0 ? headers : undefined
}

export function buildServerConfig(values: ServerFormValues): Omit<MCPServerConfig, 'id'> {
  return {
    name: values.name,
    description: values.description || undefined,
    transport: values.transport,
    enabled: values.enabled,
    connection: values.transport === 'stdio'
      ? {
          command: values.command,
          args: values.args.split('\n').filter(arg => arg.trim()),
        }
      : {
          url: values.url,
          headers: parseHeaders(values.headers),
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
    connection: values.transport === 'stdio'
      ? {
          command: values.command,
          args: values.args.split('\n').filter(arg => arg.trim()),
        }
      : {
          url: values.url,
          headers: parseHeaders(values.headers),
        },
    options: {
      timeout: parseInt(values.timeout, 10) || DEFAULT_TIMEOUT_MS,
    },
  }
}

export function toHeaderLines(headers?: Record<string, string>): string {
  if (!headers || Object.keys(headers).length === 0) {
    return ''
  }
  return Object.entries(headers)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n')
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
