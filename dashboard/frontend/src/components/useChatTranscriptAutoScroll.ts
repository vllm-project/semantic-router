import { useCallback, useEffect, useMemo, useRef } from 'react'

import type { Message } from './ChatComponentTypes'

const BOTTOM_THRESHOLD_PX = 72

function isNearBottom(container: HTMLDivElement) {
  return container.scrollHeight - container.scrollTop - container.clientHeight <= BOTTOM_THRESHOLD_PX
}

function hasLiveAssistantActivity(message: Message) {
  return (
    message.role === 'assistant' &&
    (
      message.isStreaming === true ||
      message.toolCalls?.some(toolCall => toolCall.status === 'pending' || toolCall.status === 'running') === true
    )
  )
}

export function useChatTranscriptAutoScroll(messages: Message[]) {
  const containerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const shouldAutoFollowRef = useRef(true)

  const liveAssistantActivity = useMemo(
    () => messages.some(hasLiveAssistantActivity),
    [messages]
  )

  const syncFollowState = useCallback(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    shouldAutoFollowRef.current = isNearBottom(container)
  }, [])

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'auto') => {
    const container = containerRef.current
    if (!container) {
      return
    }

    container.scrollTo({
      top: container.scrollHeight,
      behavior,
    })
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    syncFollowState()

    const handleScroll = () => {
      syncFollowState()
    }

    container.addEventListener('scroll', handleScroll, { passive: true })

    return () => {
      container.removeEventListener('scroll', handleScroll)
    }
  }, [syncFollowState])

  useEffect(() => {
    if (messages.length === 0) {
      shouldAutoFollowRef.current = true
      return
    }

    if (shouldAutoFollowRef.current || liveAssistantActivity) {
      requestAnimationFrame(() => {
        scrollToBottom(liveAssistantActivity ? 'auto' : 'smooth')
      })
    }
  }, [liveAssistantActivity, messages, scrollToBottom])

  useEffect(() => {
    const content = contentRef.current
    if (!content || typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(() => {
      if (shouldAutoFollowRef.current) {
        scrollToBottom('auto')
      }
    })

    observer.observe(content)

    return () => {
      observer.disconnect()
    }
  }, [scrollToBottom])

  return {
    containerRef,
    contentRef,
  }
}
