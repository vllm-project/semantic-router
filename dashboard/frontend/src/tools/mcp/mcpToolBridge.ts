/**
 * MCP Tool Bridge
 * 将 MCP 工具转换为通用 Tool Registry 格式
 */

import type { MCPTool } from './types'
import type { ToolDefinition, ToolExecutor, RegisteredTool, ToolMetadata, ToolParameters } from '../types'
import * as api from './api'

const MCP_TOOL_PREFIX = 'mcp_'
const UUID_LIKE_SERVER_KEY = /^[a-f0-9-]+$/i

const SERVER_KEY_BY_ID: Record<string, string> = {
  [api.OPENCLAW_MCP_SERVER_ID]: api.OPENCLAW_MCP_TOOL_NAMESPACE,
}

const SERVER_ID_BY_KEY: Record<string, string> = {
  [api.OPENCLAW_MCP_TOOL_NAMESPACE]: api.OPENCLAW_MCP_SERVER_ID,
}

function getMCPServerKey(serverId: string): string {
  return SERVER_KEY_BY_ID[serverId] || serverId
}

function resolveMCPServerId(serverKey: string): string {
  return SERVER_ID_BY_KEY[serverKey] || serverKey
}

/**
 * 从原始 JSON Schema 生成增强的工具描述
 * 
 * 核心优势:
 * 1. 单一数据源: 直接从 tool.inputSchema 读取,不依赖中间转换层
 * 2. 自动同步: Schema 更新时描述自动更新
 * 3. 避免转换错误: 中间转换逻辑出错不影响描述准确性
 */
function buildEnhancedDescription(tool: MCPTool): string {
  const baseDescription = `[MCP: ${tool.serverName}] ${tool.description || tool.name}`

  // 如果没有参数定义,返回基础描述
  if (!tool.inputSchema?.properties) {
    return baseDescription
  }

  // 直接从原始 JSON Schema 生成参数描述
  const paramLines: string[] = []
  
  for (const [key, prop] of Object.entries(tool.inputSchema.properties)) {
    const isRequired = tool.inputSchema.required?.includes(key)
    const requiredMarker = isRequired ? 'required' : 'optional'
    const type = prop.type || 'unknown'
    const desc = prop.description || 'No description'
    
    paramLines.push(`- \`${key}\` (${type}, ${requiredMarker}): ${desc}`)
  }

  // 合并基础描述和参数列表
  if (paramLines.length > 0) {
    return `${baseDescription}\n\n**Parameters:**\n${paramLines.join('\n')}`
  }

  return baseDescription
}

/**
 * 将 MCP 工具转换为 OpenAI-compatible 工具定义
 */
export function mcpToolToDefinition(tool: MCPTool): ToolDefinition {
  // 转换 JSON Schema 为 ToolParameters (用于 OpenAI API 格式)
  const parameters: ToolParameters = {
    type: 'object',
    properties: {},
    required: [],
  }

  if (tool.inputSchema && tool.inputSchema.properties) {
    for (const [key, prop] of Object.entries(tool.inputSchema.properties)) {
      parameters.properties[key] = {
        type: prop.type as 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object',
        description: prop.description || '',
        default: prop.default,
        enum: prop.enum as string[] | undefined,
      }
    }
    parameters.required = tool.inputSchema.required || []
  }

  return {
    type: 'function',
    function: {
      name: getMCPToolId(tool.serverId, tool.name),
      // 使用从原始 JSON Schema 生成的增强描述
      description: buildEnhancedDescription(tool),
      parameters,
    },
  }
}

/**
 * 创建 MCP 工具执行器
 */
export function createMCPToolExecutor(tool: MCPTool): ToolExecutor {
  return async (args, _context) => {
    const result = await api.executeTool(tool.serverId, tool.name, args)
    
    if (!result.success) {
      throw new Error(result.error || 'Tool execution failed')
    }
    
    return result.result
  }
}

/**
 * 将 MCP 工具转换为 RegisteredTool
 */
export function mcpToolToRegisteredTool(tool: MCPTool): RegisteredTool {
  const metadata: ToolMetadata = {
    id: getMCPToolId(tool.serverId, tool.name),
    displayName: tool.name,
    category: 'custom',
    icon: 'mcp',
    enabled: true,
    version: '1.0.0',
  }

  return {
    metadata,
    definition: mcpToolToDefinition(tool),
    executor: createMCPToolExecutor(tool),
    formatResult: (result) => {
      if (typeof result === 'string') {
        return result
      }
      return JSON.stringify(result, null, 2)
    },
  }
}

/**
 * 解析 MCP 工具名称
 * 格式: mcp_{serverKey}_{toolName}
 */
export function parseMCPToolName(fullName: string): { serverId: string; toolName: string } | null {
  if (!fullName.startsWith(MCP_TOOL_PREFIX)) {
    return null
  }

  for (const [serverKey, serverId] of Object.entries(SERVER_ID_BY_KEY)) {
    const aliasPrefix = `${MCP_TOOL_PREFIX}${serverKey}_`
    if (fullName.startsWith(aliasPrefix)) {
      const toolName = fullName.slice(aliasPrefix.length)
      if (!toolName) {
        return null
      }

      return {
        serverId,
        toolName,
      }
    }
  }

  const remainder = fullName.slice(MCP_TOOL_PREFIX.length)
  const separatorIndex = remainder.indexOf('_')
  if (separatorIndex <= 0 || separatorIndex === remainder.length - 1) {
    return null
  }

  const serverKey = remainder.slice(0, separatorIndex)
  if (!UUID_LIKE_SERVER_KEY.test(serverKey)) {
    return null
  }

  return {
    serverId: resolveMCPServerId(serverKey),
    toolName: remainder.slice(separatorIndex + 1),
  }
}

/**
 * 检查是否是 MCP 工具
 */
export function isMCPTool(toolName: string): boolean {
  return toolName.startsWith(MCP_TOOL_PREFIX)
}

/**
 * 检查是否是内建 OpenClaw MCP 工具
 */
export function isOpenClawMCPToolName(toolName: string): boolean {
  const parsed = parseMCPToolName(toolName)
  return Boolean(parsed && parsed.serverId === api.OPENCLAW_MCP_SERVER_ID && parsed.toolName.startsWith('claw_'))
}

/**
 * 批量转换 MCP 工具
 */
export function convertMCPTools(tools: MCPTool[]): RegisteredTool[] {
  return tools.map(mcpToolToRegisteredTool)
}

/**
 * 获取 MCP 工具的完整 ID
 */
export function getMCPToolId(serverId: string, toolName: string): string {
  return `${MCP_TOOL_PREFIX}${getMCPServerKey(serverId)}_${toolName}`
}
