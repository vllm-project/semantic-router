/**
 * MCP (Model Context Protocol) Module
 * 导出所有 MCP 相关功能
 */

// Types
export type {
  MCPTransportType,
  MCPServerConfig,
  MCPConnectionConfig,
  MCPSecurityConfig,
  MCPServerOptions,
  MCPServerStatus,
  MCPServerState,
  MCPToolDefinition,
  MCPTool,
  MCPToolExecuteRequest,
  MCPToolResult,
  MCPStreamChunk,
  ElicitationRequest,
  ElicitationResponse,
  MCPServersResponse,
  MCPToolsResponse,
  MCPTestConnectionResponse,
  JSONSchema,
  JSONSchemaProperty,
} from './types'

// API
export * as mcpApi from './api'

// Hooks
export { useMCPServers } from './useMCPServers'
export type { UseMCPServersReturn } from './useMCPServers'
export { useMCPToolSync } from './useMCPToolSync'
export type { UseMCPToolSyncOptions, UseMCPToolSyncReturn } from './useMCPToolSync'

// Bridge
export {
  mcpToolToDefinition,
  createMCPToolExecutor,
  mcpToolToRegisteredTool,
  parseMCPToolName,
  isMCPTool,
  convertMCPTools,
  getMCPToolId,
} from './mcpToolBridge'
