/**
 * React Hook for Tool Registry
 * 工具注册中心 React Hook
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { toolRegistry } from '../registry'
import type {
  ToolDefinition,
  ToolCall,
  ToolResult,
  ToolMetadata,
  ToolExecutionContext,
  ToolEvent,
} from '../types'

export interface UseToolRegistryOptions {
  /** Auto-subscribe to registry events */
  subscribe?: boolean
  /** Filter by categories */
  categories?: ToolMetadata['category'][]
  /** Only include enabled tools */
  enabledOnly?: boolean
}

export interface UseToolRegistryReturn {
  /** All registered tool definitions */
  definitions: ToolDefinition[]
  /** All tool metadata */
  metadata: ToolMetadata[]
  /** Currently executing tool IDs */
  executingTools: Set<string>
  /** Check if any tool is executing */
  isExecuting: boolean
  /** Execute a single tool call */
  execute: (toolCall: ToolCall, context?: ToolExecutionContext) => Promise<ToolResult>
  /** Execute multiple tool calls in parallel */
  executeAll: (toolCalls: ToolCall[], context?: ToolExecutionContext) => Promise<ToolResult[]>
  /** Enable a tool */
  enableTool: (toolId: string) => void
  /** Disable a tool */
  disableTool: (toolId: string) => void
  /** Refresh tool list */
  refresh: () => void
}

export function useToolRegistry(
  options: UseToolRegistryOptions = {}
): UseToolRegistryReturn {
  const { subscribe = true, categories, enabledOnly = true } = options

  // State for reactive updates
  const [version, setVersion] = useState(0)
  const [executingTools, setExecutingTools] = useState<Set<string>>(new Set())

  // Subscribe to registry events
  useEffect(() => {
    if (!subscribe) return

    const unsubscribers: (() => void)[] = []

    // Tool registration/unregistration events
    unsubscribers.push(
      toolRegistry.on('tool:registered', () => setVersion(v => v + 1))
    )
    unsubscribers.push(
      toolRegistry.on('tool:unregistered', () => setVersion(v => v + 1))
    )
    unsubscribers.push(
      toolRegistry.on('tool:enabled', () => setVersion(v => v + 1))
    )
    unsubscribers.push(
      toolRegistry.on('tool:disabled', () => setVersion(v => v + 1))
    )

    // Execution events
    unsubscribers.push(
      toolRegistry.on('tool:execution:start', (event: ToolEvent) => {
        setExecutingTools(prev => new Set([...prev, event.toolId]))
      })
    )
    unsubscribers.push(
      toolRegistry.on('tool:execution:complete', (event: ToolEvent) => {
        setExecutingTools(prev => {
          const next = new Set(prev)
          next.delete(event.toolId)
          return next
        })
      })
    )
    unsubscribers.push(
      toolRegistry.on('tool:execution:error', (event: ToolEvent) => {
        setExecutingTools(prev => {
          const next = new Set(prev)
          next.delete(event.toolId)
          return next
        })
      })
    )

    return () => {
      unsubscribers.forEach(unsub => unsub())
    }
  }, [subscribe])

  // Memoized definitions
  const definitions = useMemo(() => {
    // Trigger re-computation on version change
    void version

    let tools = enabledOnly
      ? toolRegistry.getEnabled()
      : toolRegistry.getAll()

    if (categories?.length) {
      tools = tools.filter(t => categories.includes(t.metadata.category))
    }

    return tools.map(t => t.definition)
  }, [version, categories, enabledOnly])

  // Memoized metadata
  const metadata = useMemo(() => {
    void version

    let tools = toolRegistry.getAll()

    if (categories?.length) {
      tools = tools.filter(t => categories.includes(t.metadata.category))
    }

    return tools.map(t => t.metadata)
  }, [version, categories])

  // Execute single tool
  const execute = useCallback(
    async (toolCall: ToolCall, context?: ToolExecutionContext) => {
      return toolRegistry.execute(toolCall, context)
    },
    []
  )

  // Execute all tools
  const executeAll = useCallback(
    async (toolCalls: ToolCall[], context?: ToolExecutionContext) => {
      return toolRegistry.executeAll(toolCalls, context)
    },
    []
  )

  // Enable tool
  const enableTool = useCallback((toolId: string) => {
    toolRegistry.enable(toolId)
  }, [])

  // Disable tool
  const disableTool = useCallback((toolId: string) => {
    toolRegistry.disable(toolId)
  }, [])

  // Force refresh
  const refresh = useCallback(() => {
    setVersion(v => v + 1)
  }, [])

  return {
    definitions,
    metadata,
    executingTools,
    isExecuting: executingTools.size > 0,
    execute,
    executeAll,
    enableTool,
    disableTool,
    refresh,
  }
}
