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
export async function getServers(): Promise<MCPServerState[]> {
  const response = await fetch(`${API_BASE}/servers`)
  if (!response.ok) {
    throw new Error(`Failed to get servers: ${response.statusText}`)
  }
  const data: MCPServersResponse = await response.json()
  return data.servers || []
}

/**
 * 创建 MCP 服务器配置
 */
export async function createServer(config: Omit<MCPServerConfig, 'id'>): Promise<MCPServerConfig> {
  const response = await fetch(`${API_BASE}/servers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
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
export async function updateServer(id: string, config: Partial<MCPServerConfig>): Promise<MCPServerConfig> {
  const response = await fetch(`${API_BASE}/servers/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...config, id }),
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
export async function deleteServer(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/servers/${id}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to delete server: ${response.statusText}`)
  }
}

/**
 * 连接到 MCP 服务器
 */
export async function connectServer(id: string): Promise<MCPServerState> {
  const response = await fetch(`${API_BASE}/servers/${id}/connect`, {
    method: 'POST',
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
export async function disconnectServer(id: string): Promise<MCPServerState> {
  const response = await fetch(`${API_BASE}/servers/${id}/disconnect`, {
    method: 'POST',
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
export async function getServerStatus(id: string): Promise<MCPServerState> {
  const response = await fetch(`${API_BASE}/servers/${id}/status`)
  if (!response.ok) {
    throw new Error(`Failed to get server status: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 测试 MCP 服务器连接
 */
export async function testConnection(config: MCPServerConfig): Promise<MCPTestConnectionResponse> {
  const response = await fetch(`${API_BASE}/servers/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!response.ok) {
    throw new Error(`Failed to test connection: ${response.statusText}`)
  }
  return response.json()
}

/**
 * 获取所有可用的 MCP 工具
 */
export async function getTools(): Promise<MCPTool[]> {
  const response = await fetch(`${API_BASE}/tools`)
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
  args: unknown
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

/**
 * 流式执行 MCP 工具
 */
export async function* executeToolStreaming(
  serverId: string,
  toolName: string,
  args: unknown,
  signal?: AbortSignal
): AsyncGenerator<MCPStreamChunk, MCPToolResult, unknown> {
  const response = await fetch(`${API_BASE}/tools/execute/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify({
      server_id: serverId,
      tool_name: toolName,
      arguments: args,
    }),
    signal,
  })

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
  let finalResult: unknown = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      let eventType = ''
      let eventData = ''

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          eventData = line.slice(5).trim()
        } else if (line === '' && eventData) {
          try {
            const chunk: MCPStreamChunk = {
              type: eventType as MCPStreamChunk['type'],
              data: JSON.parse(eventData),
              timestamp: Date.now(),
            }

            if (chunk.type === 'complete') {
              finalResult = chunk.data
            }

            yield chunk
          } catch {
            yield { type: 'partial', data: eventData, timestamp: Date.now() }
          }
          eventType = ''
          eventData = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  return {
    is_streaming: true,
    success: true,
    result: finalResult,
  }
}
