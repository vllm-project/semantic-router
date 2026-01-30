/**
 * useAgentWebSocket - Hook for connecting to the E2B agent service via WebSocket
 */

import { useState, useCallback, useRef, useEffect } from 'react'

// WebSocket event types from the agent service
export interface AgentTraceMetadata {
  traceId: string
  inputTokensUsed: number
  outputTokensUsed: number
  duration: number
  numberOfSteps: number
  maxSteps: number
  completed: boolean
  final_state: 'success' | 'stopped' | 'max_steps_reached' | 'error' | 'sandbox_timeout' | null
  user_evaluation: 'success' | 'failed' | 'not_evaluated'
}

export interface AgentAction {
  function_name: string
  parameters: Record<string, unknown>
  original_string: string
  description: string
}

export interface AgentStepEvent {
  traceId: string
  stepId: string
  image: string // base64 screenshot
  thought: string | null
  actions: AgentAction[]
  error: string | null
  duration: number
  inputTokensUsed: number
  outputTokensUsed: number
  step_evaluation: 'like' | 'dislike' | 'neutral'
}

export interface AgentTrace {
  id: string
  timestamp: string
  instruction: string
  modelId: string
  isRunning: boolean
  steps: AgentStepEvent[]
  traceMetadata: AgentTraceMetadata
}

// WebSocket message types
export interface HeartbeatEvent {
  type: 'heartbeat'
  uuid: string
}

export interface AgentStartEvent {
  type: 'agent_start'
  agentTrace: AgentTrace
  status: 'max_sandboxes_reached' | 'success'
}

export interface AgentProgressEvent {
  type: 'agent_progress'
  agentStep: AgentStepEvent
  traceMetadata: AgentTraceMetadata
}

export interface AgentCompleteEvent {
  type: 'agent_complete'
  traceMetadata: AgentTraceMetadata
  final_state: 'success' | 'stopped' | 'max_steps_reached' | 'error' | 'sandbox_timeout'
}

export interface AgentErrorEvent {
  type: 'agent_error'
  error: string
}

export interface VncUrlSetEvent {
  type: 'vnc_url_set'
  vncUrl: string
}

export interface VncUrlUnsetEvent {
  type: 'vnc_url_unset'
}

export type WebSocketEvent =
  | HeartbeatEvent
  | AgentStartEvent
  | AgentProgressEvent
  | AgentCompleteEvent
  | AgentErrorEvent
  | VncUrlSetEvent
  | VncUrlUnsetEvent

// Connection state
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface UseAgentWebSocketOptions {
  wsUrl?: string
  autoConnect?: boolean
  onHeartbeat?: (uuid: string) => void
  onAgentStart?: (event: AgentStartEvent) => void
  onAgentProgress?: (event: AgentProgressEvent) => void
  onAgentComplete?: (event: AgentCompleteEvent) => void
  onAgentError?: (error: string) => void
  onVncUrlSet?: (url: string) => void
  onVncUrlUnset?: () => void
  onConnectionChange?: (state: ConnectionState) => void
}

export interface UseAgentWebSocketReturn {
  // Connection state
  connectionState: ConnectionState
  sessionId: string | null
  vncUrl: string | null

  // Current task state
  isRunning: boolean
  currentTrace: AgentTrace | null
  steps: AgentStepEvent[]
  metadata: AgentTraceMetadata | null
  error: string | null

  // Actions
  connect: () => void
  disconnect: () => void
  submitTask: (instruction: string, modelId: string) => void
  stopTask: () => void
}

export function useAgentWebSocket(options: UseAgentWebSocketOptions = {}): UseAgentWebSocketReturn {
  const {
    wsUrl = '/ws/agent',
    autoConnect = false,
    onHeartbeat,
    onAgentStart,
    onAgentProgress,
    onAgentComplete,
    onAgentError,
    onVncUrlSet,
    onVncUrlUnset,
    onConnectionChange,
  } = options

  // State
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [vncUrl, setVncUrl] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [currentTrace, setCurrentTrace] = useState<AgentTrace | null>(null)
  const [steps, setSteps] = useState<AgentStepEvent[]>([])
  const [metadata, setMetadata] = useState<AgentTraceMetadata | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Refs
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Update connection state and notify
  const updateConnectionState = useCallback((state: ConnectionState) => {
    setConnectionState(state)
    onConnectionChange?.(state)
  }, [onConnectionChange])

  // Handle incoming WebSocket message
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data: WebSocketEvent = JSON.parse(event.data)
      console.log('[AgentWS] Received event:', data.type)

      switch (data.type) {
        case 'heartbeat':
          setSessionId(data.uuid)
          onHeartbeat?.(data.uuid)
          break

        case 'agent_start':
          setCurrentTrace(data.agentTrace)
          setSteps([])
          setMetadata(data.agentTrace.traceMetadata)
          setIsRunning(true)
          setError(null)
          if (data.status === 'max_sandboxes_reached') {
            setError('Maximum sandboxes reached. Please try again later.')
            setIsRunning(false)
          }
          onAgentStart?.(data)
          break

        case 'agent_progress':
          setSteps(prev => [...prev, data.agentStep])
          setMetadata(data.traceMetadata)
          onAgentProgress?.(data)
          break

        case 'agent_complete':
          setMetadata(data.traceMetadata)
          setIsRunning(false)
          onAgentComplete?.(data)
          break

        case 'agent_error':
          setError(data.error)
          setIsRunning(false)
          onAgentError?.(data.error)
          break

        case 'vnc_url_set':
          setVncUrl(data.vncUrl)
          onVncUrlSet?.(data.vncUrl)
          break

        case 'vnc_url_unset':
          setVncUrl(null)
          onVncUrlUnset?.()
          break

        default:
          console.warn('[AgentWS] Unknown event type:', (data as { type: string }).type)
      }
    } catch (err) {
      console.error('[AgentWS] Failed to parse message:', err)
    }
  }, [onHeartbeat, onAgentStart, onAgentProgress, onAgentComplete, onAgentError, onVncUrlSet, onVncUrlUnset])

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    // Build WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const fullUrl = wsUrl.startsWith('ws') ? wsUrl : `${protocol}//${host}${wsUrl}`

    console.log('[AgentWS] Connecting to:', fullUrl)
    updateConnectionState('connecting')

    try {
      const ws = new WebSocket(fullUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[AgentWS] Connected')
        updateConnectionState('connected')
      }

      ws.onmessage = handleMessage

      ws.onerror = (err) => {
        console.error('[AgentWS] Error:', err)
        updateConnectionState('error')
        setError('WebSocket connection error')
      }

      ws.onclose = () => {
        console.log('[AgentWS] Disconnected')
        updateConnectionState('disconnected')
        setSessionId(null)
        setVncUrl(null)
        wsRef.current = null
      }
    } catch (err) {
      console.error('[AgentWS] Failed to connect:', err)
      updateConnectionState('error')
      setError('Failed to connect to agent service')
    }
  }, [wsUrl, handleMessage, updateConnectionState])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    updateConnectionState('disconnected')
    setSessionId(null)
    setVncUrl(null)
  }, [updateConnectionState])

  // Submit a task to the agent
  const submitTask = useCallback((instruction: string, modelId: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('WebSocket not connected')
      return
    }

    if (!sessionId) {
      setError('No session ID')
      return
    }

    console.log('[AgentWS] Submitting task:', instruction)

    const message = {
      type: 'user_task',
      trace: {
        id: sessionId,
        timestamp: new Date().toISOString(),
        instruction,
        modelId,
        isRunning: true,
        steps: [],
        traceMetadata: {
          traceId: sessionId,
          inputTokensUsed: 0,
          outputTokensUsed: 0,
          duration: 0,
          numberOfSteps: 0,
          maxSteps: 30,
          completed: false,
          final_state: null,
          user_evaluation: 'not_evaluated',
        },
      },
    }

    wsRef.current.send(JSON.stringify(message))
    setError(null)
  }, [sessionId])

  // Stop the current task
  const stopTask = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    if (!sessionId) {
      return
    }

    console.log('[AgentWS] Stopping task')

    const message = {
      type: 'stop_task',
      trace_id: sessionId,
    }

    wsRef.current.send(JSON.stringify(message))
  }, [sessionId])

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    // Connection state
    connectionState,
    sessionId,
    vncUrl,

    // Current task state
    isRunning,
    currentTrace,
    steps,
    metadata,
    error,

    // Actions
    connect,
    disconnect,
    submitTask,
    stopTask,
  }
}

export default useAgentWebSocket
