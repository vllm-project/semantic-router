/**
 * Tool Registry - Entry Point
 * 工具注册中心入口
 * 
 * Usage:
 * ```typescript
 * import { toolRegistry, useToolRegistry } from '@/tools'
 * 
 * // Get all enabled tool definitions for API call
 * const tools = toolRegistry.getDefinitions()
 * 
 * // Execute tool calls
 * const results = await toolRegistry.executeAll(toolCalls)
 * 
 * // In React components
 * const { tools, execute, isExecuting } = useToolRegistry()
 * ```
 */

// Core exports
export { 
  toolRegistry, 
  createTool, 
  registerTools, 
  unregisterTools,
  replaceAllTools,
  ToolRegistry,
} from './registry'

// Options and result types
export type { 
  RegisterToolsOptions, 
  RegisterToolsResult,
} from './registry'

// Type exports
export type {
  ToolDefinition,
  ToolCall,
  ToolResult,
  ToolExecutionContext,
  ExecuteAllOptions,
  ToolMetadata,
  RegisteredTool,
  ToolEvent,
  ToolEventType,
  ToolEventListener,
  ToolExecutor,
  ToolCardRenderer,
  ToolParameters,
  ToolParameterProperty,
  // Specific tool types
  WebSearchArgs,
  WebSearchResult,
  OpenWebArgs,
  OpenWebResult,
  CurrentTimeArgs,
  CurrentTimeResult,
  CalculateArgs,
  CalculateResult,
  WeatherArgs,
  WeatherResult,
} from './types'

// Built-in tools
export { webSearchTool } from './executors/webSearch'
export { openWebTool } from './executors/openWeb'
export { currentTimeTool } from './executors/currentTime'
export { calculateTool } from './executors/calculate'
export { weatherTool } from './executors/weather'

// MCP (Model Context Protocol) support
export * from './mcp'

// React hook
export { useToolRegistry } from './hooks/useToolRegistry'

// Initialize built-in tools
import { toolRegistry } from './registry'
import { webSearchTool } from './executors/webSearch'
import { openWebTool } from './executors/openWeb'
import { currentTimeTool } from './executors/currentTime'
import { calculateTool } from './executors/calculate'
import { weatherTool } from './executors/weather'

// Auto-register built-in tools
toolRegistry.register(webSearchTool)
toolRegistry.register(openWebTool)
toolRegistry.register(currentTimeTool)
toolRegistry.register(calculateTool)
toolRegistry.register(weatherTool)
