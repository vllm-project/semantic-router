/**
 * useMCPServers Hook
 * 管理 MCP 服务器状态的 React Hook
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import {
  createLatestMCPRequestRunner,
  getMCPRequestErrorMessage,
  type LatestMCPRequestRunner,
} from './requestSupport'
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
  /** 工具目录错误信息 */
  toolsError: string | null

  // 服务器管理
  /** 刷新服务器列表 */
  refreshServers: (signal?: AbortSignal) => Promise<void>
  /** 添加服务器 */
  addServer: (config: Omit<MCPServerConfig, 'id'>, signal?: AbortSignal) => Promise<MCPServerConfig>
  /** 更新服务器 */
  updateServer: (
    id: string,
    config: Partial<MCPServerConfig>,
    signal?: AbortSignal,
  ) => Promise<void>
  /** 删除服务器 */
  deleteServer: (id: string, signal?: AbortSignal) => Promise<void>

  // 连接管理
  /** 连接服务器 */
  connect: (id: string, signal?: AbortSignal) => Promise<void>
  /** 断开连接 */
  disconnect: (id: string, signal?: AbortSignal) => Promise<void>
  /** 测试连接 */
  testConnection: (
    config: MCPServerConfig,
    signal?: AbortSignal,
  ) => Promise<{ success: boolean; error?: string }>

  // 工具管理
  /** 刷新工具列表 */
  refreshTools: (signal?: AbortSignal) => Promise<void>
}

/**
 * MCP 服务器管理 Hook
 */
export function useMCPServers(): UseMCPServersReturn {
  const [servers, setServers] = useState<MCPServerState[]>([])
  const [tools, setTools] = useState<MCPTool[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toolsError, setToolsError] = useState<string | null>(null)
  const serversRequestRef = useRef<LatestMCPRequestRunner | null>(null)
  const toolsRequestRef = useRef<LatestMCPRequestRunner | null>(null)
  if (!serversRequestRef.current) serversRequestRef.current = createLatestMCPRequestRunner()
  if (!toolsRequestRef.current) toolsRequestRef.current = createLatestMCPRequestRunner()

  // 刷新服务器列表
  const refreshServers = useCallback(async (externalSignal?: AbortSignal) => {
    await serversRequestRef.current?.run(
      (signal) => api.getServers(signal),
      {
        onSuccess: (data) => {
          setServers(data)
          setError(null)
        },
        onError: (requestError) => {
          setError(getMCPRequestErrorMessage(requestError, 'Failed to load MCP servers.'))
        },
      },
      externalSignal,
    )
  }, [])

  // 刷新工具列表
  const refreshTools = useCallback(async (externalSignal?: AbortSignal) => {
    await toolsRequestRef.current?.run(
      (signal) => api.getTools(signal),
      {
        onSuccess: (data) => {
          setTools(data)
          setToolsError(null)
        },
        onError: (requestError) => {
          setToolsError(getMCPRequestErrorMessage(requestError, 'Failed to load MCP tools.'))
        },
      },
      externalSignal,
    )
  }, [])

  // 初始加载
  useEffect(() => {
    let active = true
    const init = async () => {
      setLoading(true)
      await Promise.all([refreshServers(), refreshTools()])
      if (active) setLoading(false)
    }
    void init()
    return () => {
      active = false
      serversRequestRef.current?.cancel()
      toolsRequestRef.current?.cancel()
    }
  }, [refreshServers, refreshTools])

  // 添加服务器
  const addServer = useCallback(
    async (config: Omit<MCPServerConfig, 'id'>, signal?: AbortSignal) => {
      const newServer = await api.createServer(config, signal)
      await refreshServers(signal)
      return newServer
    },
    [refreshServers],
  )

  // 更新服务器
  const updateServer = useCallback(
    async (id: string, config: Partial<MCPServerConfig>, signal?: AbortSignal) => {
      await api.updateServer(id, config, signal)
      await refreshServers(signal)
    },
    [refreshServers],
  )

  // 删除服务器
  const deleteServer = useCallback(
    async (id: string, signal?: AbortSignal) => {
      await api.deleteServer(id, signal)
      await Promise.all([refreshServers(signal), refreshTools(signal)])
    },
    [refreshServers, refreshTools],
  )

  // 连接服务器
  const connect = useCallback(
    async (id: string, signal?: AbortSignal) => {
      await api.connectServer(id, signal)
      await Promise.all([refreshServers(signal), refreshTools(signal)])
    },
    [refreshServers, refreshTools],
  )

  // 断开连接
  const disconnect = useCallback(
    async (id: string, signal?: AbortSignal) => {
      await api.disconnectServer(id, signal)
      await Promise.all([refreshServers(signal), refreshTools(signal)])
    },
    [refreshServers, refreshTools],
  )

  // 测试连接
  const testConnection = useCallback(async (config: MCPServerConfig, signal?: AbortSignal) => {
    return api.testConnection(config, signal)
  }, [])

  return {
    servers,
    tools,
    loading,
    error,
    toolsError,
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
