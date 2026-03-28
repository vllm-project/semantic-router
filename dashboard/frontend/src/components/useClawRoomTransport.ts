import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import {
  type RoomMessage,
  type RoomStreamEvent,
  type WSOutboundMessage,
} from './clawRoomChatSupport'

interface UseClawRoomTransportParams {
  selectedRoomId: string
  fetchMessages: (roomId: string) => Promise<void>
  setMessages: Dispatch<SetStateAction<RoomMessage[]>>
  setError: Dispatch<SetStateAction<string | null>>
  upsertMessage: (message: RoomMessage) => void
}

export const useClawRoomTransport = ({
  selectedRoomId,
  fetchMessages,
  setMessages,
  setError,
  upsertMessage,
}: UseClawRoomTransportParams) => {
  const wsRef = useRef<WebSocket | null>(null)
  const sourceRef = useRef<EventSource | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const heartbeatTimerRef = useRef<number | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [streamingMessages, setStreamingMessages] = useState<Map<string, string>>(new Map())

  useEffect(() => {
    if (!selectedRoomId) {
      setMessages([])
      setWsConnected(false)
      setStreamingMessages(new Map())
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

    const connectWebSocket = () => {
      if (!mounted) return

      if (wsRef.current) {
        wsRef.current.close()
      }
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
      if (heartbeatTimerRef.current !== null) {
        window.clearInterval(heartbeatTimerRef.current)
        heartbeatTimerRef.current = null
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/ws`
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mounted) return
        console.log('WebSocket connected to room:', selectedRoomId)
        setWsConnected(true)
        reconnectAttemptsRef.current = 0
        heartbeatTimerRef.current = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, 30000)
      }

      ws.onmessage = event => {
        try {
          const payload = JSON.parse(event.data) as WSOutboundMessage
          if (payload.type === 'new_message' && payload.message) {
            upsertMessage(payload.message)
            if (payload.message.id) {
              setStreamingMessages(previous => {
                const next = new Map(previous)
                next.delete(payload.message!.id)
                return next
              })
            }
          } else if (payload.type === 'message_chunk' && payload.messageId) {
            if (payload.chunk) {
              setStreamingMessages(previous => {
                const next = new Map(previous)
                const existing = next.get(payload.messageId!) || ''
                next.set(payload.messageId!, existing + payload.chunk)
                return next
              })
            }
          } else if (payload.type === 'error' && payload.error) {
            console.error('WebSocket error from server:', payload.error)
            setError(payload.error)
          }
        } catch {
          // ignore malformed messages
        }
      }

      ws.onerror = event => {
        console.error('WebSocket error:', event)
        setWsConnected(false)
      }

      ws.onclose = event => {
        if (!mounted) return
        console.log('WebSocket closed:', event.code, event.reason)
        setWsConnected(false)
        if (heartbeatTimerRef.current !== null) {
          window.clearInterval(heartbeatTimerRef.current)
          heartbeatTimerRef.current = null
        }
        if (reconnectTimerRef.current !== null) {
          window.clearTimeout(reconnectTimerRef.current)
        }
        const baseDelay = 1000
        const maxDelay = 30000
        const delay = Math.min(baseDelay * Math.pow(2, reconnectAttemptsRef.current), maxDelay)
        reconnectAttemptsRef.current += 1
        console.log(`WebSocket reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`)
        reconnectTimerRef.current = window.setTimeout(() => {
          if (mounted) {
            connectWebSocket()
          }
        }, delay)
      }
    }

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const connectSSE = () => {
      if (!mounted || wsRef.current?.readyState === WebSocket.OPEN) return

      if (sourceRef.current) {
        sourceRef.current.close()
      }

      const source = new EventSource(`/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/stream`)
      sourceRef.current = source

      source.addEventListener('message', ((event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as RoomStreamEvent
          if (payload.message) {
            upsertMessage(payload.message)
          }
        } catch {
          // ignore malformed stream events
        }
      }) as EventListener)

      source.onerror = () => {
        source.close()
        if (!mounted) return
        if (reconnectTimerRef.current !== null) {
          window.clearTimeout(reconnectTimerRef.current)
        }
        reconnectTimerRef.current = window.setTimeout(connectSSE, 1500)
      }
    }

    void loadMessages()
    connectWebSocket()

    return () => {
      mounted = false
      setWsConnected(false)
      reconnectAttemptsRef.current = 0
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
    }
  }, [fetchMessages, selectedRoomId, setError, setMessages, upsertMessage])

  return {
    streamingMessages,
    wsConnected,
    wsRef,
  }
}
