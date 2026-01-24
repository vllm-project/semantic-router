/**
 * MCP Tool Executor
 * 用于执行 MCP 工具的执行器
 */

import type { ToolExecutionContext, ToolResult } from '../types'
import type { MCPTool, MCPStreamChunk } from '../mcp/types'
import * as api from '../mcp/api'
import { parseMCPToolName } from '../mcp/mcpToolBridge'

/**
 * 执行 MCP 工具
 */
export async function executeMCPTool(
  toolName: string,
  args: unknown,
  _context: ToolExecutionContext
): Promise<ToolResult> {
  const parsed = parseMCPToolName(toolName)
  if (!parsed) {
    return {
      callId: '',
      name: toolName,
      content: null,
      error: `Invalid MCP tool name: ${toolName}`,
    }
  }

  const { serverId, toolName: actualToolName } = parsed

  try {
    const result = await api.executeTool(serverId, actualToolName, args)

    if (!result.success) {
      return {
        callId: '',
        name: toolName,
        content: null,
        error: result.error || 'Tool execution failed',
      }
    }

    return {
      callId: '',
      name: toolName,
      content: result.result,
    }
  } catch (error) {
    return {
      callId: '',
      name: toolName,
      content: null,
      error: error instanceof Error ? error.message : 'Unknown error',
    }
  }
}

/**
 * 流式执行 MCP 工具
 */
export async function* executeMCPToolStream(
  toolName: string,
  args: unknown,
  context: ToolExecutionContext
): AsyncGenerator<MCPStreamChunk, ToolResult, unknown> {
  const parsed = parseMCPToolName(toolName)
  if (!parsed) {
    return {
      callId: '',
      name: toolName,
      content: null,
      error: `Invalid MCP tool name: ${toolName}`,
    }
  }

  const { serverId, toolName: actualToolName } = parsed

  try {
    const generator = api.executeToolStreaming(
      serverId, 
      actualToolName, 
      args, 
      context.signal
    )

    let finalResult: unknown = null

    for await (const chunk of generator) {
      if (chunk.type === 'complete') {
        finalResult = chunk.data
      }
      yield chunk
    }

    return {
      callId: '',
      name: toolName,
      content: finalResult,
    }
  } catch (error) {
    return {
      callId: '',
      name: toolName,
      content: null,
      error: error instanceof Error ? error.message : 'Unknown error',
    }
  }
}

/**
 * 创建 MCP 工具执行器工厂
 */
export function createMCPExecutor(tool: MCPTool) {
  return async (args: unknown, _context: ToolExecutionContext) => {
    const result = await api.executeTool(tool.serverId, tool.name, args)
    
    if (!result.success) {
      throw new Error(result.error || 'Tool execution failed')
    }
    
    return result.result
  }
}
