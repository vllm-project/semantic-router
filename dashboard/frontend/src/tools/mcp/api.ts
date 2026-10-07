/**
 * MCP API Service
 * 与后端 MCP API 交互的服务层
 */

import type {
  MCPServerConfig,
  MCPServerState,
  MCPServersResponse,
  MCPToolsResponse,
  MCPTool,
  MCPToolResult,
  MCPTestConnectionResponse,
  MCPStreamChunk,
} from './types'

const API_BASE = '/api/mcp'

/**
 * 获取所有 MCP 服务器状态
 */
export async function getServers(signal?: AbortSignal): Promise<MCPServerState[]> {
  const response = await fetch(`${API_BASE}/servers`, { signal })
  if (!response.ok) {
    throw new Error(`Failed to get servers: ${response.statusText}`)
  }
  const data: MCPServersResponse = await response.json()
  return data.servers || []
}

/**
 * 创建 MCP 服务器配置
 */
export async function createServer(
  config: Omit<MCPServerConfig, 'id'> & { id?: string },
  signal?: AbortSignal,
): Promise<MCPServerConfig> {
  const response = await fetch(`${API_BASE}/servers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
    signal,
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to create server: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 更新 MCP 服务器配置
 */
export async function updateServer(
  id: string,
  config: Partial<MCPServerConfig>,
  signal?: AbortSignal,
): Promise<MCPServerConfig> {
  const response = await fetch(`${API_BASE}/servers/${encodeURIComponent(id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...config, id }),
    signal,
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to update server: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 删除 MCP 服务器配置
 */
export async function deleteServer(id: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API_BASE}/servers/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    signal,
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to delete server: ${response.statusText}`)
  }
}

/**
 * 连接到 MCP 服务器
 */
export async function connectServer(id: string, signal?: AbortSignal): Promise<MCPServerState> {
  const response = await fetch(`${API_BASE}/servers/${encodeURIComponent(id)}/connect`, {
    method: 'POST',
    signal,
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to connect: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 断开与 MCP 服务器的连接
 */
export async function disconnectServer(id: string, signal?: AbortSignal): Promise<MCPServerState> {
  const response = await fetch(`${API_BASE}/servers/${encodeURIComponent(id)}/disconnect`, {
    method: 'POST',
    signal,
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to disconnect: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 获取 MCP 服务器状态
 */
export async function getServerStatus(id: string, signal?: AbortSignal): Promise<MCPServerState> {
  const response = await fetch(`${API_BASE}/servers/${encodeURIComponent(id)}/status`, { signal })
  if (!response.ok) {
    throw new Error(`Failed to get server status: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 测试 MCP 服务器连接
 */
export async function testConnection(
  config: MCPServerConfig,
  signal?: AbortSignal,
): Promise<MCPTestConnectionResponse> {
  const response = await fetch(`${API_BASE}/servers/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
    signal,
  })
  if (!response.ok) {
    throw new Error(`Failed to test connection: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 获取所有可用的 MCP 工具
 */
export async function getTools(signal?: AbortSignal): Promise<MCPTool[]> {
  const response = await fetch(`${API_BASE}/tools`, { signal })
  if (!response.ok) {
    throw new Error(`Failed to get tools: ${response.statusText}`)
  }
  const data: MCPToolsResponse = await response.json()
  return data.tools || []
}

/**
 * 执行 MCP 工具
 */
export async function executeTool(
  serverId: string,
  toolName: string,
  args: unknown,
): Promise<MCPToolResult> {
  const response = await fetch(`${API_BASE}/tools/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      server_id: serverId,
      tool_name: toolName,
      arguments: args,
    }),
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to execute tool: ${response.statusText}`)
  }
  return response.json()
}

const MCP_STREAM_TYPES = new Set<MCPStreamChunk['type']>([
  'progress',
  'partial',
  'complete',
  'error',
])

interface MCPStreamParseState {
  eventType: string
  eventData: string
  sawComplete: boolean
  sawError: boolean
  result: unknown
  error?: string
}

function isAbortError(err: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true
  return err instanceof Error && err.name === 'AbortError'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function mcpStreamChunkFromSSE(eventName: string, rawData: string): MCPStreamChunk {
  const timestamp = Date.now()
  let parsed: unknown
  try {
    parsed = JSON.parse(rawData)
  } catch {
    return { type: 'partial', data: rawData, timestamp }
  }

  if (
    isRecord(parsed) &&
    typeof parsed.type === 'string' &&
    MCP_STREAM_TYPES.has(parsed.type as MCPStreamChunk['type'])
  ) {
    return {
      type: parsed.type as MCPStreamChunk['type'],
      data: parsed.data,
      progress: typeof parsed.progress === 'number' ? parsed.progress : undefined,
      timestamp,
    }
  }

  if (isRecord(parsed) && typeof parsed.error === 'string') {
    return { type: 'error', data: parsed.error, timestamp }
  }

  if (MCP_STREAM_TYPES.has(eventName as MCPStreamChunk['type'])) {
    return { type: eventName as MCPStreamChunk['type'], data: parsed, timestamp }
  }

  return { type: 'partial', data: parsed, timestamp }
}

function pushSSELine(state: MCPStreamParseState, line: string): MCPStreamChunk | undefined {
  const normalized = line.endsWith('\r') ? line.slice(0, -1) : line
  if (normalized.startsWith(':')) return undefined
  if (normalized.startsWith('event:')) {
    state.eventType = normalized.slice(6).trim()
    return undefined
  }
  if (normalized.startsWith('data:')) {
    let value = normalized.slice(5)
    if (value.startsWith(' ')) value = value.slice(1)
    state.eventData = state.eventData ? `${state.eventData}\n${value}` : value
    return undefined
  }
  if (normalized !== '' || !state.eventData) return undefined

  const chunk = mcpStreamChunkFromSSE(state.eventType, state.eventData)
  state.eventType = ''
  state.eventData = ''
  if (chunk.type === 'complete') {
    state.sawComplete = true
    state.result = chunk.data
  } else if (chunk.type === 'error') {
    state.sawError = true
    state.result = chunk.data
    state.error =
      typeof chunk.data === 'string' && chunk.data ? chunk.data : 'Tool execution failed'
  }
  return chunk
}

function streamTerminalResult(state: MCPStreamParseState): MCPToolResult {
  if (state.sawError || !state.sawComplete) {
    return {
      is_streaming: true,
      success: false,
      result: state.sawError ? state.result : undefined,
      error: state.error ?? 'Stream ended before a complete result',
    }
  }
  return {
    is_streaming: true,
    success: true,
    result: state.result,
  }
}

function cancelledStreamResult(): MCPToolResult {
  return {
    is_streaming: true,
    success: false,
    error: 'Streaming execution cancelled',
  }
}

/**
 * Stream MCP tool execution over SSE.
 *
 * Contract: each SSE `event` is progress | partial | complete | error; JSON `data`
 * is a StreamChunk and `data.type` carries the same meaning. Expect one terminal
 * complete or error event; EOF without complete is reported as success: false.
 */
export async function* executeToolStreaming(
  serverId: string,
  toolName: string,
  args: unknown,
  signal?: AbortSignal,
): AsyncGenerator<MCPStreamChunk, MCPToolResult, unknown> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}/tools/execute/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({
        server_id: serverId,
        tool_name: toolName,
        arguments: args,
      }),
      signal,
    })
  } catch (err) {
    if (isAbortError(err, signal)) return cancelledStreamResult()
    throw err
  }

  if (!response.ok) {
    return {
      is_streaming: true,
      success: false,
      error: `Streaming execution failed: ${response.statusText}`,
    }
  }

  if (!response.body) {
    throw new Error('Response body is null')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const state: MCPStreamParseState = {
    eventType: '',
    eventData: '',
    sawComplete: false,
    sawError: false,
    result: undefined,
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (value) buffer += decoder.decode(value, { stream: true })
      if (done) buffer += decoder.decode()
      const lines = buffer.split('\n')
      buffer = done ? '' : lines.pop() || ''

      for (const line of lines) {
        const chunk = pushSSELine(state, line)
        if (chunk) yield chunk
      }
      if (done) {
        const trailing = pushSSELine(state, '')
        if (trailing) yield trailing
        break
      }
    }
  } catch (err) {
    if (isAbortError(err, signal)) return cancelledStreamResult()
    throw err
  } finally {
    reader.releaseLock()
  }

  return streamTerminalResult(state)
}
