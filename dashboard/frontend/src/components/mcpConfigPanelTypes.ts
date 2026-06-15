import type { MCPToolDefinition } from '../tools/mcp'
import type { RegisteredTool } from '../tools'

export interface BuiltInToolParameter {
  type: string
  description?: string
  enum?: string[]
  default?: unknown
}

export interface BuiltInTool {
  tool: {
    type: string
    function: {
      name: string
      description: string
      parameters: {
        type: string
        properties: Record<string, BuiltInToolParameter>
        required?: string[]
      }
    }
  }
  description: string
  category: string
  tags: string[]
}

export interface UnifiedToolParameter {
  name: string
  type: string
  description?: string
  required: boolean
}

export interface UnifiedTool {
  id: string
  name: string
  description: string
  source: string
  sourceType: 'mcp' | 'frontend' | 'backend'
  parameters: UnifiedToolParameter[]
  rawTool: MCPToolDefinition | RegisteredTool | BuiltInTool
}

export type ServerFilter = 'all' | 'connected' | 'disconnected'

export const TOOLS_GRID_MAX_HEIGHT = 320
