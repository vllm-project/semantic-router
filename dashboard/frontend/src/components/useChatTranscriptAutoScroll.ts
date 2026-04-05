import { useCallback, useEffect, useMemo, useRef } from 'react'

import type { Message } from './ChatComponentTypes'

const BOTTOM_THRESHOLD_PX = 72
const TURN_TOP_OFFSET_PX = 16

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

export function useChatTranscriptAutoScroll(messages: Message[], conversationId: string) {
  const containerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const shouldAutoFollowRef = useRef(true)
  const anchoredUserMessageIdRef = useRef<string | null>(null)
  const shouldAnchorCurrentTurnRef = useRef(false)
  const isProgrammaticScrollRef = useRef(false)
  const programmaticScrollTimeoutRef = useRef<number | null>(null)
  const lastConversationIdRef = useRef<string | null>(null)

  const liveAssistantActivity = useMemo(
    () => messages.some(hasLiveAssistantActivity),
    [messages]
  )
  const lastUserMessage = useMemo(
    () => [...messages].reverse().find(message => message.role === 'user') ?? null,
    [messages]
  )

  const syncFollowState = useCallback(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    shouldAutoFollowRef.current = isNearBottom(container)
  }, [])

  const releaseAnchorToUser = useCallback(() => {
    shouldAnchorCurrentTurnRef.current = false
    isProgrammaticScrollRef.current = false
    syncFollowState()
  }, [syncFollowState])

  const beginProgrammaticScroll = useCallback(() => {
    isProgrammaticScrollRef.current = true

    if (programmaticScrollTimeoutRef.current !== null) {
      window.clearTimeout(programmaticScrollTimeoutRef.current)
    }

    programmaticScrollTimeoutRef.current = window.setTimeout(() => {
      isProgrammaticScrollRef.current = false
      programmaticScrollTimeoutRef.current = null
    }, 120)
  }, [])

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'auto') => {
    const container = containerRef.current
    if (!container) {
      return
    }

    beginProgrammaticScroll()
    container.scrollTo({
      top: container.scrollHeight,
      behavior,
    })
  }, [beginProgrammaticScroll])

  const scrollUserTurnToTop = useCallback((messageId: string, behavior: ScrollBehavior = 'smooth') => {
    const container = containerRef.current
    const content = contentRef.current
    if (!container || !content) {
      return
    }

    const target = Array.from(content.querySelectorAll<HTMLElement>('[data-message-id]'))
      .find(node => node.dataset.messageId === messageId)

    if (!target) {
      return
    }

    const targetRect = target.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    const nextTop = container.scrollTop + (targetRect.top - containerRect.top) - TURN_TOP_OFFSET_PX

    beginProgrammaticScroll()
    container.scrollTo({
      top: Math.max(0, nextTop),
      behavior,
    })
  }, [beginProgrammaticScroll])

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    syncFollowState()

    const handleScroll = () => {
      if (isProgrammaticScrollRef.current) {
        return
      }

      syncFollowState()
    }

    const handleUserScrollIntent = () => {
      releaseAnchorToUser()
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    container.addEventListener('wheel', handleUserScrollIntent, { passive: true })
    container.addEventListener('pointerdown', handleUserScrollIntent, { passive: true })
    container.addEventListener('touchmove', handleUserScrollIntent, { passive: true })

    return () => {
      container.removeEventListener('scroll', handleScroll)
      container.removeEventListener('wheel', handleUserScrollIntent)
      container.removeEventListener('pointerdown', handleUserScrollIntent)
      container.removeEventListener('touchmove', handleUserScrollIntent)
    }
  }, [releaseAnchorToUser, syncFollowState])

  useEffect(() => {
    return () => {
      if (programmaticScrollTimeoutRef.current !== null) {
        window.clearTimeout(programmaticScrollTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (lastConversationIdRef.current === conversationId) {
      return
    }

    lastConversationIdRef.current = conversationId
    anchoredUserMessageIdRef.current = lastUserMessage?.id ?? null
    shouldAnchorCurrentTurnRef.current = false
    shouldAutoFollowRef.current = true

    requestAnimationFrame(() => {
      scrollToBottom('auto')
    })
  }, [conversationId, lastUserMessage, scrollToBottom])

  useEffect(() => {
    if (messages.length === 0) {
      shouldAutoFollowRef.current = true
      anchoredUserMessageIdRef.current = null
      shouldAnchorCurrentTurnRef.current = false
      return
    }

    if (lastUserMessage && anchoredUserMessageIdRef.current !== lastUserMessage.id) {
      anchoredUserMessageIdRef.current = lastUserMessage.id
      shouldAnchorCurrentTurnRef.current = true
      shouldAutoFollowRef.current = false
      requestAnimationFrame(() => {
        scrollUserTurnToTop(lastUserMessage.id, 'auto')
      })
      return
    }

    if (shouldAnchorCurrentTurnRef.current && lastUserMessage) {
      requestAnimationFrame(() => {
        scrollUserTurnToTop(lastUserMessage.id, 'auto')
      })
      return
    }

    if (shouldAutoFollowRef.current && liveAssistantActivity) {
      requestAnimationFrame(() => {
        scrollToBottom('auto')
      })
    }
  }, [lastUserMessage, liveAssistantActivity, messages, scrollToBottom, scrollUserTurnToTop])

  useEffect(() => {
    const content = contentRef.current
    if (!content || typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(() => {
      if (shouldAnchorCurrentTurnRef.current && anchoredUserMessageIdRef.current) {
        scrollUserTurnToTop(anchoredUserMessageIdRef.current, 'auto')
        return
      }

      if (shouldAutoFollowRef.current) {
        scrollToBottom('auto')
      }
    })

    observer.observe(content)

    return () => {
      observer.disconnect()
    }
  }, [scrollToBottom, scrollUserTurnToTop])

  return {
    containerRef,
    contentRef,
  }
}
