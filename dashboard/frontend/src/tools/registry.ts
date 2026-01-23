/**
 * Tool Registry - Core Implementation
 * 工具注册中心核心实现
 * 
 * Features:
 * - Dynamic tool registration/unregistration
 * - Tool discovery and lookup
 * - Parallel tool execution
 * - Event-driven architecture
 * - Type-safe tool definitions
 */

import type {
  RegisteredTool,
  ToolDefinition,
  ToolCall,
  ToolResult,
  ToolExecutionContext,
  ToolMetadata,
  ToolEvent,
  ToolEventType,
  ToolEventListener,
  ExecuteAllOptions,
} from './types'

// Default configuration
const DEFAULT_TIMEOUT = 30000 // 30 seconds
const DEFAULT_CONCURRENCY = 10

class ToolRegistry {
  private tools: Map<string, RegisteredTool> = new Map()
  private listeners: Map<ToolEventType, Set<ToolEventListener>> = new Map()
  private activeExecutions: number = 0

  // ========== Registration Methods ==========

  /**
   * Register a new tool
   */
  register<TArgs, TResult>(tool: RegisteredTool<TArgs, TResult>): void {
    const toolId = tool.metadata.id

    if (this.tools.has(toolId)) {
      console.warn(`Tool "${toolId}" is already registered. Overwriting...`)
    }

    this.tools.set(toolId, tool as RegisteredTool)
    this.emit('tool:registered', toolId, { metadata: tool.metadata })
  }

  /**
   * Unregister a tool by ID
   */
  unregister(toolId: string): boolean {
    const removed = this.tools.delete(toolId)
    if (removed) {
      this.emit('tool:unregistered', toolId)
    }
    return removed
  }

  /**
   * Check if a tool is registered
   */
  has(toolId: string): boolean {
    return this.tools.has(toolId)
  }

  /**
   * Get a registered tool by ID
   */
  get<TArgs = unknown, TResult = unknown>(
    toolId: string
  ): RegisteredTool<TArgs, TResult> | undefined {
    return this.tools.get(toolId) as RegisteredTool<TArgs, TResult> | undefined
  }

  // ========== Discovery Methods ==========

  /**
   * Get all registered tools
   */
  getAll(): RegisteredTool[] {
    return Array.from(this.tools.values())
  }

  /**
   * Get all enabled tools
   */
  getEnabled(): RegisteredTool[] {
    return this.getAll().filter(t => t.metadata.enabled)
  }

  /**
   * Get tools by category
   */
  getByCategory(category: ToolMetadata['category']): RegisteredTool[] {
    return this.getAll().filter(t => t.metadata.category === category)
  }

  /**
   * Get OpenAI-compatible tool definitions for enabled tools
   */
  getDefinitions(): ToolDefinition[] {
    return this.getEnabled().map(t => t.definition)
  }

  /**
   * Get tool metadata list
   */
  getMetadataList(): ToolMetadata[] {
    return this.getAll().map(t => t.metadata)
  }

  // ========== Enable/Disable Methods ==========

  /**
   * Enable a tool
   */
  enable(toolId: string): boolean {
    const tool = this.tools.get(toolId)
    if (tool && !tool.metadata.enabled) {
      tool.metadata.enabled = true
      this.emit('tool:enabled', toolId)
      return true
    }
    return false
  }

  /**
   * Disable a tool
   */
  disable(toolId: string): boolean {
    const tool = this.tools.get(toolId)
    if (tool && tool.metadata.enabled) {
      tool.metadata.enabled = false
      this.emit('tool:disabled', toolId)
      return true
    }
    return false
  }

  // ========== Execution Methods ==========

  /**
   * Execute a single tool call with timeout
   */
  async execute(
    toolCall: ToolCall,
    context: ToolExecutionContext = {}
  ): Promise<ToolResult> {
    const toolId = toolCall.function.name
    const tool = this.tools.get(toolId)

    if (!tool) {
      return {
        callId: toolCall.id,
        name: toolId,
        content: null,
        error: `Unknown tool: ${toolId}`,
      }
    }

    if (!tool.metadata.enabled) {
      return {
        callId: toolCall.id,
        name: toolId,
        content: null,
        error: `Tool "${toolId}" is disabled`,
      }
    }

    // Parse arguments
    let args: unknown
    try {
      args = JSON.parse(toolCall.function.arguments || '{}')
    } catch (e) {
      return {
        callId: toolCall.id,
        name: toolId,
        content: null,
        error: `Failed to parse arguments: ${e instanceof Error ? e.message : 'Invalid JSON'}`,
      }
    }

    // Validate arguments if validator is provided
    if (tool.validateArgs) {
      try {
        args = tool.validateArgs(args)
      } catch (e) {
        return {
          callId: toolCall.id,
          name: toolId,
          content: null,
          error: `Invalid arguments: ${e instanceof Error ? e.message : 'Validation failed'}`,
        }
      }
    }

    // Execute with timeout
    this.emit('tool:execution:start', toolId, { callId: toolCall.id, args })
    this.activeExecutions++

    const timeout = context.timeout ?? DEFAULT_TIMEOUT

    try {
      const result = await this.executeWithTimeout(
        tool.executor(args, {
          ...context,
          onProgress: (progress) => {
            this.emit('tool:execution:progress', toolId, { callId: toolCall.id, progress })
            context.onProgress?.(progress)
          },
        }),
        timeout,
        context.signal
      )

      this.emit('tool:execution:complete', toolId, { callId: toolCall.id, result })

      return {
        callId: toolCall.id,
        name: toolId,
        content: result,
      }
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Execution failed'
      this.emit('tool:execution:error', toolId, { callId: toolCall.id, error })

      return {
        callId: toolCall.id,
        name: toolId,
        content: null,
        error,
      }
    } finally {
      this.activeExecutions--
    }
  }

  /**
   * Execute with timeout wrapper
   */
  private async executeWithTimeout<T>(
    promise: Promise<T>,
    timeout: number,
    signal?: AbortSignal
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Execution timeout after ${timeout}ms`))
      }, timeout)

      // Handle abort signal
      if (signal) {
        signal.addEventListener('abort', () => {
          clearTimeout(timeoutId)
          reject(new Error('Execution aborted'))
        })
      }

      promise
        .then((result) => {
          clearTimeout(timeoutId)
          resolve(result)
        })
        .catch((error) => {
          clearTimeout(timeoutId)
          reject(error)
        })
    })
  }

  /**
   * Execute multiple tool calls with concurrency control
   */
  async executeAll(
    toolCalls: ToolCall[],
    options: ExecuteAllOptions = {}
  ): Promise<ToolResult[]> {
    const { concurrency = DEFAULT_CONCURRENCY, continueOnError = true, ...context } = options

    if (toolCalls.length === 0) {
      return []
    }

    // If concurrency is high enough, just run all in parallel
    if (concurrency >= toolCalls.length) {
      return Promise.all(toolCalls.map(tc => this.execute(tc, context)))
    }

    // Semaphore-based concurrency control
    const results: ToolResult[] = new Array(toolCalls.length)
    let currentIndex = 0
    let hasError = false

    const executeNext = async (): Promise<void> => {
      while (currentIndex < toolCalls.length) {
        if (!continueOnError && hasError) {
          return
        }

        const index = currentIndex++
        const toolCall = toolCalls[index]

        try {
          results[index] = await this.execute(toolCall, context)
          if (results[index].error) {
            hasError = true
          }
        } catch (e) {
          hasError = true
          results[index] = {
            callId: toolCall.id,
            name: toolCall.function.name,
            content: null,
            error: e instanceof Error ? e.message : 'Execution failed',
          }
        }
      }
    }

    // Start concurrent workers
    const workers = Array(Math.min(concurrency, toolCalls.length))
      .fill(null)
      .map(() => executeNext())

    await Promise.all(workers)

    return results
  }

  /**
   * Execute multiple tool calls sequentially
   */
  async executeSequential(
    toolCalls: ToolCall[],
    context: ToolExecutionContext = {}
  ): Promise<ToolResult[]> {
    const results: ToolResult[] = []
    for (const tc of toolCalls) {
      if (context.signal?.aborted) {
        results.push({
          callId: tc.id,
          name: tc.function.name,
          content: null,
          error: 'Execution aborted',
        })
        break
      }
      results.push(await this.execute(tc, context))
    }
    return results
  }

  // ========== Event Methods ==========

  /**
   * Subscribe to tool events
   */
  on(eventType: ToolEventType, listener: ToolEventListener): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set())
    }
    this.listeners.get(eventType)!.add(listener)

    // Return unsubscribe function
    return () => {
      this.listeners.get(eventType)?.delete(listener)
    }
  }

  /**
   * Emit a tool event
   */
  private emit(type: ToolEventType, toolId: string, data?: unknown): void {
    const event: ToolEvent = {
      type,
      toolId,
      timestamp: Date.now(),
      data,
    }

    this.listeners.get(type)?.forEach(listener => {
      try {
        listener(event)
      } catch (e) {
        console.error(`Error in tool event listener:`, e)
      }
    })
  }

  // ========== Utility Methods ==========

  /**
   * Clear all registered tools
   */
  clear(): void {
    const toolIds = Array.from(this.tools.keys())
    this.tools.clear()
    toolIds.forEach(id => this.emit('tool:unregistered', id))
  }

  /**
   * Get number of currently active executions
   */
  getActiveExecutions(): number {
    return this.activeExecutions
  }

  /**
   * Get registry statistics
   */
  getStats(): {
    total: number
    enabled: number
    disabled: number
    activeExecutions: number
    byCategory: Record<string, number>
  } {
    const all = this.getAll()
    const byCategory: Record<string, number> = {}

    all.forEach(t => {
      byCategory[t.metadata.category] = (byCategory[t.metadata.category] || 0) + 1
    })

    return {
      total: all.length,
      enabled: all.filter(t => t.metadata.enabled).length,
      disabled: all.filter(t => !t.metadata.enabled).length,
      activeExecutions: this.activeExecutions,
      byCategory,
    }
  }
}

// ========== Singleton Instance ==========

export const toolRegistry = new ToolRegistry()

// ========== Helper Functions ==========

/**
 * Create a tool definition helper
 */
export function createTool<TArgs, TResult>(
  config: RegisteredTool<TArgs, TResult>
): RegisteredTool<TArgs, TResult> {
  return config
}

/**
 * Registration options for batch operations
 */
export interface RegisterToolsOptions {
  /** Skip already registered tools instead of overwriting */
  skipExisting?: boolean
  /** Stop on first error */
  stopOnError?: boolean
  /** Validate tool definitions before registering */
  validate?: boolean
}

/**
 * Registration result for tracking
 */
export interface RegisterToolsResult {
  success: string[]
  skipped: string[]
  failed: { id: string; error: string }[]
}

/**
 * Register multiple tools at once with options
 */
export function registerTools(
  tools: RegisteredTool[],
  options: RegisterToolsOptions = {}
): RegisterToolsResult {
  const { skipExisting = false, stopOnError = false, validate = true } = options
  
  const result: RegisterToolsResult = {
    success: [],
    skipped: [],
    failed: [],
  }

  // Deduplicate by tool ID (keep last occurrence)
  const uniqueTools = new Map<string, RegisteredTool>()
  tools.forEach(tool => {
    uniqueTools.set(tool.metadata.id, tool)
  })

  for (const tool of uniqueTools.values()) {
    const toolId = tool.metadata.id

    try {
      // Skip if already exists and skipExisting is true
      if (skipExisting && toolRegistry.has(toolId)) {
        result.skipped.push(toolId)
        continue
      }

      // Validate tool definition
      if (validate) {
        validateToolDefinition(tool)
      }

      toolRegistry.register(tool)
      result.success.push(toolId)
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Unknown error'
      result.failed.push({ id: toolId, error })

      if (stopOnError) {
        break
      }
    }
  }

  return result
}

/**
 * Validate tool definition
 */
function validateToolDefinition(tool: RegisteredTool): void {
  if (!tool.metadata?.id) {
    throw new Error('Tool metadata.id is required')
  }
  if (!tool.definition?.function?.name) {
    throw new Error('Tool definition.function.name is required')
  }
  if (tool.metadata.id !== tool.definition.function.name) {
    throw new Error(
      `Tool ID mismatch: metadata.id="${tool.metadata.id}" vs function.name="${tool.definition.function.name}"`
    )
  }
  if (typeof tool.executor !== 'function') {
    throw new Error('Tool executor must be a function')
  }
}

/**
 * Unregister multiple tools at once
 */
export function unregisterTools(toolIds: string[]): {
  removed: string[]
  notFound: string[]
} {
  const removed: string[] = []
  const notFound: string[] = []

  toolIds.forEach(id => {
    if (toolRegistry.unregister(id)) {
      removed.push(id)
    } else {
      notFound.push(id)
    }
  })

  return { removed, notFound }
}

/**
 * Replace all tools atomically
 */
export function replaceAllTools(
  tools: RegisteredTool[],
  options: RegisterToolsOptions = {}
): RegisterToolsResult {
  // Store current tools for rollback
  const previousTools = toolRegistry.getAll()
  
  // Clear all
  toolRegistry.clear()
  
  // Register new tools
  const result = registerTools(tools, { ...options, stopOnError: true })
  
  // Rollback if any failed and stopOnError
  if (result.failed.length > 0 && options.stopOnError) {
    toolRegistry.clear()
    previousTools.forEach(tool => toolRegistry.register(tool))
    throw new Error(
      `Failed to replace tools: ${result.failed.map(f => f.id).join(', ')}`
    )
  }
  
  return result
}

export { ToolRegistry }
