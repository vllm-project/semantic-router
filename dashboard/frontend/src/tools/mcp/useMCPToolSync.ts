/**
 * useMCPToolSync Hook
 * 同步 MCP 工具到 toolRegistry，使其可以在 Playground 中使用
 */

import { useEffect, useRef, useCallback } from 'react'
import { toolRegistry } from '../registry'
import { convertMCPTools } from './mcpToolBridge'
import type { MCPTool } from './types'
import * as api from './api'

export interface UseMCPToolSyncOptions {
  /** 是否启用同步 */
  enabled?: boolean
  /** 轮询间隔（毫秒），0 表示不轮询 */
  pollInterval?: number
}

export interface UseMCPToolSyncReturn {
  /** 同步的 MCP 工具数量 */
  syncedCount: number
  /** 手动刷新同步 */
  refresh: () => Promise<void>
}

/**
 * 同步 MCP 工具到全局 toolRegistry
 * 
 * 使用方式：
 * ```tsx
 * // 在 App 或顶层组件中调用
 * useMCPToolSync({ enabled: true, pollInterval: 30000 })
 * ```
 */
export function useMCPToolSync(
  options: UseMCPToolSyncOptions = {}
): UseMCPToolSyncReturn {
  const { enabled = true, pollInterval = 30000 } = options
  
  // 跟踪已注册的 MCP 工具 ID
  const registeredMCPToolIds = useRef<Set<string>>(new Set())
  const syncedCountRef = useRef<number>(0)

  /**
   * 同步 MCP 工具到 registry
   */
  const syncTools = useCallback(async () => {
    if (!enabled) return

    try {
      // 获取所有 MCP 工具
      const mcpTools: MCPTool[] = await api.getTools()
      
      // 转换为 RegisteredTool 格式
      const registeredTools = convertMCPTools(mcpTools)
      
      // 获取新工具的 ID 集合
      const newToolIds = new Set(registeredTools.map(t => t.metadata.id))
      
      // 找出需要移除的工具（之前注册但现在不存在的）
      const toRemove = Array.from(registeredMCPToolIds.current)
        .filter(id => !newToolIds.has(id))
      
      // 移除旧工具
      toRemove.forEach(id => {
        toolRegistry.unregister(id)
        registeredMCPToolIds.current.delete(id)
      })
      
      // 注册/更新新工具
      registeredTools.forEach(tool => {
        // 始终重新注册以确保工具定义是最新的
        toolRegistry.register(tool)
        registeredMCPToolIds.current.add(tool.metadata.id)
      })
      
      syncedCountRef.current = registeredTools.length
      
      if (registeredTools.length > 0) {
        console.log(`[MCP Sync] 已同步 ${registeredTools.length} 个 MCP 工具到 toolRegistry`)
      }
      if (toRemove.length > 0) {
        console.log(`[MCP Sync] 已移除 ${toRemove.length} 个过期的 MCP 工具`)
      }
    } catch (error) {
      console.error('[MCP Sync] 同步 MCP 工具失败:', error)
    }
  }, [enabled])

  /**
   * 手动刷新同步
   */
  const refresh = useCallback(async () => {
    await syncTools()
  }, [syncTools])

  // 初始同步和轮询
  useEffect(() => {
    if (!enabled) return

    const registeredIds = registeredMCPToolIds.current

    // 初始同步
    syncTools()

    // 设置轮询
    let intervalId: ReturnType<typeof setInterval> | null = null
    if (pollInterval > 0) {
      intervalId = setInterval(syncTools, pollInterval)
    }

    // 清理函数
    return () => {
      if (intervalId) {
        clearInterval(intervalId)
      }
      
      // 组件卸载时移除所有 MCP 工具
      registeredIds.forEach(id => {
        toolRegistry.unregister(id)
      })
      registeredIds.clear()
    }
  }, [enabled, pollInterval, syncTools])

  return {
    syncedCount: syncedCountRef.current,
    refresh,
  }
}

export default useMCPToolSync
