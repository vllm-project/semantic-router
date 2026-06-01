import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import {
  applyCollaborationOutboundEvent,
  applyRoomStreamEvent,
  type RoomMessage,
  type RoomStreamEvent,
  type RoomTransportMode,
  type StreamingParticipant,
  type WSOutboundMessage,
} from './clawRoomChatSupport'

interface UseClawRoomTransportParams {
  selectedRoomId: string
  fetchMessages: (roomId: string) => Promise<void>
  setMessages: Dispatch<SetStateAction<RoomMessage[]>>
  setError: Dispatch<SetStateAction<string | null>>
  upsertMessage: (message: RoomMessage) => void
  onRoomEvent?: (event: WSOutboundMessage) => void
}

export const useClawRoomTransport = ({
  selectedRoomId,
  fetchMessages,
  setMessages,
  setError,
  upsertMessage,
  onRoomEvent,
}: UseClawRoomTransportParams) => {
  const wsRef = useRef<WebSocket | null>(null)
  const sourceRef = useRef<EventSource | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const heartbeatTimerRef = useRef<number | null>(null)
  const onRoomEventRef = useRef(onRoomEvent)
  const [wsConnected, setWsConnected] = useState(false)
  const [transportMode, setTransportMode] = useState<RoomTransportMode>('connecting')
  const [streamingMessages, setStreamingMessages] = useState<Map<string, string>>(new Map())
  const [streamingParticipants, setStreamingParticipants] = useState<Map<string, StreamingParticipant>>(
    new Map()
  )

  useEffect(() => {
    onRoomEventRef.current = onRoomEvent
  }, [onRoomEvent])

  useEffect(() => {
    if (!selectedRoomId) {
      setMessages([])
      setWsConnected(false)
      setTransportMode('connecting')
      setStreamingMessages(new Map())
      setStreamingParticipants(new Map())
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      if (heartbeatTimerRef.current !== null) {
        window.clearInterval(heartbeatTimerRef.current)
        heartbeatTimerRef.current = null
      }
      reconnectAttemptsRef.current = 0
      return
    }

    let mounted = true

    const collaborationHandlers = {
      upsertMessage,
      setStreamingMessages,
      setStreamingParticipants,
      setError: (message: string) => setError(message),
    }

    const handleOutboundEvent = (payload: WSOutboundMessage) => {
      applyCollaborationOutboundEvent(payload, collaborationHandlers)
      onRoomEventRef.current?.(payload)
    }

    const closeSSE = () => {
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
    }

    const loadMessages = async () => {
      try {
        await fetchMessages(selectedRoomId)
        if (mounted) {
          setError(null)
        }
      } catch (err) {
        if (!mounted) return
        const message = err instanceof Error ? err.message : 'Failed to load messages'
        setError(message)
      }
    }

    const connectSSE = () => {
      if (!mounted || wsRef.current?.readyState === WebSocket.OPEN) {
        return
      }

      if (sourceRef.current) {
        sourceRef.current.close()
      }

      const source = new EventSource(`/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/stream`)
      sourceRef.current = source

      source.addEventListener('message', ((event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as RoomStreamEvent
          applyRoomStreamEvent(payload, collaborationHandlers)
          if (payload.message) {
            onRoomEventRef.current?.({
              type: payload.type === 'message' ? 'new_message' : payload.type,
              roomId: payload.roomId,
              message: payload.message,
            })
          }
        } catch {
          // ignore malformed stream events
        }
      }) as EventListener)

      source.addEventListener('message_updated', ((event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as RoomStreamEvent
          applyRoomStreamEvent(payload, collaborationHandlers)
          if (payload.message) {
            onRoomEventRef.current?.({
              type: 'message_updated',
              roomId: payload.roomId,
              message: payload.message,
            })
          }
        } catch {
          // ignore malformed stream events
        }
      }) as EventListener)

      source.onopen = () => {
        if (!mounted) return
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          setTransportMode('sse')
          setWsConnected(false)
        }
      }

      source.onerror = () => {
        source.close()
        if (!mounted) return
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          return
        }
        setTransportMode('connecting')
        if (reconnectTimerRef.current !== null) {
          window.clearTimeout(reconnectTimerRef.current)
        }
        reconnectTimerRef.current = window.setTimeout(connectSSE, 1500)
      }
    }

    const connectWebSocket = () => {
      if (!mounted) return

      if (wsRef.current) {
        wsRef.current.close()
      }
      closeSSE()
      if (heartbeatTimerRef.current !== null) {
        window.clearInterval(heartbeatTimerRef.current)
        heartbeatTimerRef.current = null
      }

      setTransportMode('connecting')

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/ws`
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mounted) return
        setWsConnected(true)
        setTransportMode('websocket')
        reconnectAttemptsRef.current = 0
        closeSSE()
        heartbeatTimerRef.current = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, 30000)
      }

      ws.onmessage = event => {
        try {
          const payload = JSON.parse(event.data) as WSOutboundMessage
          handleOutboundEvent(payload)
        } catch {
          // ignore malformed messages
        }
      }

      ws.onerror = () => {
        setWsConnected(false)
      }

      ws.onclose = () => {
        if (!mounted) return
        setWsConnected(false)
        if (heartbeatTimerRef.current !== null) {
          window.clearInterval(heartbeatTimerRef.current)
          heartbeatTimerRef.current = null
        }
        if (reconnectTimerRef.current !== null) {
          window.clearTimeout(reconnectTimerRef.current)
        }

        connectSSE()

        const baseDelay = 1000
        const maxDelay = 30000
        const delay = Math.min(baseDelay * Math.pow(2, reconnectAttemptsRef.current), maxDelay)
        reconnectAttemptsRef.current += 1
        reconnectTimerRef.current = window.setTimeout(() => {
          if (mounted) {
            connectWebSocket()
          }
        }, delay)
      }
    }

    void loadMessages()
    connectWebSocket()

    return () => {
      mounted = false
      setWsConnected(false)
      setTransportMode('connecting')
      reconnectAttemptsRef.current = 0
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      closeSSE()
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      if (heartbeatTimerRef.current !== null) {
        window.clearInterval(heartbeatTimerRef.current)
        heartbeatTimerRef.current = null
      }
    }
  }, [fetchMessages, selectedRoomId, setError, setMessages, upsertMessage])

  return {
    streamingMessages,
    streamingParticipants,
    transportMode,
    wsConnected,
    wsRef,
  }
}
