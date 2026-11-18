import type {
  MCPServerConfig,
  MCPServerState,
  MCPToolDefinition,
  MCPTransportType,
} from '../tools/mcp'
import type { LucideIcon } from 'lucide-react'
import {
  AppWindow,
  Calculator,
  CalendarDays,
  CircleDot,
  Clock,
  CloudSun,
  Database,
  FilePen,
  FileText,
  FolderTree,
  GitCommitHorizontal,
  GitPullRequest,
  Globe,
  Image,
  Mail,
  Newspaper,
  Search,
  Terminal,
  Wrench,
} from 'lucide-react'
import type { RegisteredTool } from '../tools'
import type {
  BuiltInTool,
  ServerFilters,
  ToolSourceFilter,
  UnifiedTool,
  UnifiedToolParameter,
} from './mcpConfigPanelTypes'

const DEFAULT_TIMEOUT_MS = 30000
export const MCP_REDACTED_ARGUMENTS = '__VLLM_SR_REDACTED__'

export interface ServerFormValues {
  name: string
  description: string
  transport: MCPTransportType
  enabled: boolean
  command: string
  args: string[]
  preserveArguments: boolean
  url: string
  headers: Record<string, string>
  timeout: string
}

export function hiddenArgumentsBlockCommandChange(
  server: MCPServerConfig | null,
  command: string,
  preserveArguments: boolean,
): boolean {
  return Boolean(
    server?.transport === 'stdio' &&
      preserveArguments &&
      command !== (server.connection.command || ''),
  )
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

export function filterUnifiedTools(
  tools: UnifiedTool[],
  searchValue: string,
  sourceFilter: ToolSourceFilter,
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
    .sort((left, right) => left.name.localeCompare(right.name, undefined, { sensitivity: 'base' }))
}

/** Connected servers first, then the ones still trying, then the rest. */
export const SERVER_STATUS_RANK: Record<MCPServerState['status'], number> = {
  connected: 0,
  connecting: 1,
  error: 2,
  disconnected: 3,
}

export function filterServers(
  servers: MCPServerState[],
  searchValue: string,
  { transports, statuses }: ServerFilters,
): MCPServerState[] {
  const search = searchValue.trim().toLowerCase()
  return servers
    .filter((server) => transports.length === 0 || transports.includes(server.config.transport))
    .filter((server) => statuses.length === 0 || statuses.includes(server.status))
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
    .sort((left, right) =>
      left.config.name.localeCompare(right.config.name, undefined, { sensitivity: 'base' }),
    )
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

function normalizeArgs(args: readonly string[], preserveArguments: boolean): string[] | undefined {
  if (preserveArguments) return undefined
  const normalized = args.map((argument) => argument.trim()).filter(Boolean)
  return normalized
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
            args: normalizeArgs(values.args, values.preserveArguments),
          }
        : {
            url: values.url,
            // An explicit empty object tells the backend to clear all stored
            // headers; omission means that a client did not edit the field.
            headers: normalizeHeaders(values.headers) || {},
          },
    options: {
      timeout: parseInt(values.timeout, 10) || DEFAULT_TIMEOUT_MS,
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
            args: normalizeArgs(values.args, values.preserveArguments),
          }
        : {
            url: values.url,
            headers: normalizeHeaders(values.headers) || {},
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

const TOOL_ICON_RULES: Array<[RegExp, LucideIcon]> = [
  [/weather|forecast/, CloudSun],
  [/calc|math/, Calculator],
  [/e?mail/, Mail],
  [/calendar|event|slot|schedule/, CalendarDays],
  [/time|clock|date/, Clock],
  [/image|photo|picture/, Image],
  [/news/, Newspaper],
  [/pull_request|pull-request|\bpr\b/, GitPullRequest],
  [/issue/, CircleDot],
  [/commit|git/, GitCommitHorizontal],
  [/sql|query|table|database|\bdb\b/, Database],
  [/director|folder|list_files/, FolderTree],
  [/write|edit|save/, FilePen],
  [/file|read|doc/, FileText],
  [/page|browser|open_web|url/, AppWindow],
  [/search|find|lookup|web/, Search],
]

/** A recognisable icon for a tool from its name; a wrench when nothing fits. */
export function toolIcon(name: string): LucideIcon {
  const key = name.toLowerCase()
  return TOOL_ICON_RULES.find(([pattern]) => pattern.test(key))?.[1] ?? Wrench
}

export const TOOL_SOURCE_META: Record<
  UnifiedTool['sourceType'],
  { label: string; tone: 'brand' | 'neutral' | 'outline' }
> = {
  mcp: { label: 'MCP', tone: 'brand' },
  frontend: { label: 'Built-in', tone: 'neutral' },
  backend: { label: 'Router', tone: 'outline' },
}

export const TRANSPORT_META: Record<
  MCPTransportType,
  { label: string; icon: LucideIcon; hint: string }
> = {
  stdio: { label: 'stdio', icon: Terminal, hint: 'Runs as a local process next to the dashboard.' },
  'streamable-http': {
    label: 'HTTP',
    icon: Globe,
    hint: 'Streamable HTTP endpoint, usually ending in /mcp.',
  },
}

/** Where a server connects, for display: its command line or URL. */
export function serverTarget(config: MCPServerConfig): string {
  if (config.transport === 'stdio') {
    const args = (config.connection.args ?? []).filter((arg) => arg !== MCP_REDACTED_ARGUMENTS)
    const hidden = (config.connection.args ?? []).length !== args.length
    return [config.connection.command ?? '', ...args, hidden ? '…' : ''].filter(Boolean).join(' ')
  }
  return (config.connection.url ?? '').replace(MCP_REDACTED_ARGUMENTS, '•••')
}

export const SERVER_STATUS_META: Record<
  MCPServerState['status'],
  { label: string; tone: 'success' | 'info' | 'danger' | 'neutral'; pulse?: boolean }
> = {
  connected: { label: 'Connected', tone: 'success' },
  connecting: { label: 'Connecting', tone: 'info', pulse: true },
  error: { label: 'Error', tone: 'danger' },
  disconnected: { label: 'Not connected', tone: 'neutral' },
}
