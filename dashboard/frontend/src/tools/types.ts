/**
 * Tool Registry - Type Definitions
 * 工具注册中心类型定义
 */

// ========== Tool Definition Types ==========

/**
 * OpenAI-compatible function parameter schema
 */
export interface ToolParameterProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object'
  description: string
  default?: unknown
  enum?: string[]
  items?: ToolParameterProperty
}

export interface ToolParameters {
  type: 'object'
  properties: Record<string, ToolParameterProperty>
  required: string[]
}

/**
 * OpenAI-compatible tool definition
 */
export interface ToolDefinition {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: ToolParameters
  }
}

// ========== Tool Execution Types ==========

/**
 * Tool call from LLM response
 */
export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
  status: 'pending' | 'running' | 'completed' | 'failed'
}

/**
 * Generic tool result - content can be any type depending on the tool
 */
export interface ToolResult<T = unknown> {
  callId: string
  name: string
  content: T
  error?: string
}

/**
 * Context passed to tool executor
 */
export interface ToolExecutionContext {
  /** Abort signal for cancellation */
  signal?: AbortSignal
  /** Callback to report execution progress (0-100) */
  onProgress?: (progress: number) => void
  /** Custom headers to include in API calls */
  headers?: Record<string, string>
  /** Timeout in milliseconds (default: 30000) */
  timeout?: number
}

/**
 * Options for batch execution
 */
export interface ExecuteAllOptions extends ToolExecutionContext {
  /** Maximum concurrent executions (default: 3) */
  concurrency?: number
  /** Continue execution even if some tools fail */
  continueOnError?: boolean
}

/**
 * Tool executor function signature
 */
export type ToolExecutor<TArgs = unknown, TResult = unknown> = (
  args: TArgs,
  context: ToolExecutionContext
) => Promise<TResult>

// ========== Tool Registration Types ==========

/**
 * UI component type for rendering tool card
 */
export type ToolCardRenderer = React.ComponentType<{
  toolCall: ToolCall
  toolResult?: ToolResult
  isExpanded: boolean
  onToggle: () => void
}>

/**
 * Tool metadata for registry
 */
export interface ToolMetadata {
  /** Unique tool identifier (same as function name) */
  id: string
  /** Display name for UI */
  displayName: string
  /** Tool category for grouping */
  category: 'search' | 'code' | 'file' | 'image' | 'custom'
  /** Icon name or component */
  icon?: string
  /** Whether tool is enabled */
  enabled: boolean
  /** Tool version */
  version: string
}

/**
 * Complete tool registration entry
 */
export interface RegisteredTool<TArgs = unknown, TResult = unknown> {
  /** Tool metadata */
  metadata: ToolMetadata
  /** OpenAI-compatible definition */
  definition: ToolDefinition
  /** Executor function */
  executor: ToolExecutor<TArgs, TResult>
  /** UI card renderer component */
  cardRenderer?: ToolCardRenderer
  /** Validate arguments before execution */
  validateArgs?: (args: unknown) => TArgs | never
  /** Transform result for display */
  formatResult?: (result: TResult) => string
}

// ========== Event Types ==========

export type ToolEventType = 
  | 'tool:registered'
  | 'tool:unregistered'
  | 'tool:enabled'
  | 'tool:disabled'
  | 'tool:execution:start'
  | 'tool:execution:progress'
  | 'tool:execution:complete'
  | 'tool:execution:error'

export interface ToolEvent {
  type: ToolEventType
  toolId: string
  timestamp: number
  data?: unknown
}

export type ToolEventListener = (event: ToolEvent) => void

// ========== Specific Tool Types ==========

// Web Search
export interface WebSearchArgs {
  query: string
  num_results?: number
}

export interface WebSearchResult {
  title: string
  url: string
  snippet: string
  domain: string
}
