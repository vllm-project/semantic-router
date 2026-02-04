/**
 * useMCPServers Hook
 * 管理 MCP 服务器状态的 React Hook
 */

import { useState, useEffect, useCallback } from 'react'
import type { MCPServerConfig, MCPServerState, MCPTool } from './types'
import * as api from './api'

export interface UseMCPServersReturn {
  /** 所有服务器状态 */
  servers: MCPServerState[]
  /** 所有可用工具 */
  tools: MCPTool[]
  /** 是否正在加载 */
  loading: boolean
  /** 错误信息 */
  error: string | null
  
  // 服务器管理
  /** 刷新服务器列表 */
  refreshServers: () => Promise<void>
  /** 添加服务器 */
  addServer: (config: Omit<MCPServerConfig, 'id'>) => Promise<MCPServerConfig>
  /** 更新服务器 */
  updateServer: (id: string, config: Partial<MCPServerConfig>) => Promise<void>
  /** 删除服务器 */
  deleteServer: (id: string) => Promise<void>
  
  // 连接管理
  /** 连接服务器 */
  connect: (id: string) => Promise<void>
  /** 断开连接 */
  disconnect: (id: string) => Promise<void>
  /** 测试连接 */
  testConnection: (config: MCPServerConfig) => Promise<{ success: boolean; error?: string }>
  
  // 工具管理
  /** 刷新工具列表 */
  refreshTools: () => Promise<void>
}

/**
 * MCP 服务器管理 Hook
 */
export function useMCPServers(): UseMCPServersReturn {
  const [servers, setServers] = useState<MCPServerState[]>([])
  const [tools, setTools] = useState<MCPTool[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // 刷新服务器列表
  const refreshServers = useCallback(async () => {
    try {
      const data = await api.getServers()
      setServers(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load servers')
    }
  }, [])

  // 刷新工具列表
  const refreshTools = useCallback(async () => {
    try {
      const data = await api.getTools()
      setTools(data)
    } catch (err) {
      console.error('Failed to load tools:', err)
    }
  }, [])

  // 初始加载
  useEffect(() => {
    const init = async () => {
      setLoading(true)
      await Promise.all([refreshServers(), refreshTools()])
      setLoading(false)
    }
    init()
  }, [refreshServers, refreshTools])

  // 添加服务器
  const addServer = useCallback(async (config: Omit<MCPServerConfig, 'id'>) => {
    const newServer = await api.createServer(config)
    await refreshServers()
    return newServer
  }, [refreshServers])

  // 更新服务器
  const updateServer = useCallback(async (id: string, config: Partial<MCPServerConfig>) => {
    await api.updateServer(id, config)
    await refreshServers()
  }, [refreshServers])

  // 删除服务器
  const deleteServer = useCallback(async (id: string) => {
    await api.deleteServer(id)
    await Promise.all([refreshServers(), refreshTools()])
  }, [refreshServers, refreshTools])

  // 连接服务器
  const connect = useCallback(async (id: string) => {
    await api.connectServer(id)
    await Promise.all([refreshServers(), refreshTools()])
  }, [refreshServers, refreshTools])

  // 断开连接
  const disconnect = useCallback(async (id: string) => {
    await api.disconnectServer(id)
    await Promise.all([refreshServers(), refreshTools()])
  }, [refreshServers, refreshTools])

  // 测试连接
  const testConnection = useCallback(async (config: MCPServerConfig) => {
    return api.testConnection(config)
  }, [])

  return {
    servers,
    tools,
    loading,
    error,
    refreshServers,
    addServer,
    updateServer,
    deleteServer,
    connect,
    disconnect,
    testConnection,
    refreshTools,
  }
}
