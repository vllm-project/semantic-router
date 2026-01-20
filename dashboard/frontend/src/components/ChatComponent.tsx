import { useState, useRef, useEffect, useCallback } from 'react'
import styles from './ChatComponent.module.css'
import HeaderDisplay from './HeaderDisplay'
import MarkdownRenderer from './MarkdownRenderer'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'

// Choice represents a single model's response in ratings mode
interface Choice {
  content: string
  model?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  isStreaming?: boolean
  headers?: Record<string, string>
  // For ratings mode: multiple choices from different models
  choices?: Choice[]
}

interface ChatComponentProps {
  endpoint?: string
  defaultModel?: string
  defaultSystemPrompt?: string
  isFullscreenMode?: boolean
}

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  defaultModel = 'MoM',
  defaultSystemPrompt = 'You are a helpful assistant.',
  isFullscreenMode = false,
}: ChatComponentProps) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState(defaultModel)
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt)
  const [showSettings, setShowSettings] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const [showHeaderReveal, setShowHeaderReveal] = useState(false)
  const [pendingHeaders, setPendingHeaders] = useState<Record<string, string> | null>(null)
  const [isFullscreen] = useState(isFullscreenMode)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // When headers arrive, show HeaderReveal
  useEffect(() => {
    if (pendingHeaders && Object.keys(pendingHeaders).length > 0) {
      setShowHeaderReveal(true)
    }
  }, [pendingHeaders])

  // Toggle fullscreen mode by adding/removing class to body
  useEffect(() => {
    if (isFullscreen) {
      document.body.classList.add('playground-fullscreen')
    } else {
      document.body.classList.remove('playground-fullscreen')
    }

    return () => {
      document.body.classList.remove('playground-fullscreen')
    }
  }, [isFullscreen])

  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

  const handleThinkingComplete = useCallback(() => {
    // Thinking animation will be hidden when headers arrive
    // This callback is kept for ThinkingAnimation component compatibility
  }, [])

  const handleHeaderRevealComplete = useCallback(() => {
    setShowHeaderReveal(false)
    setPendingHeaders(null)
  }, [])

  const handleSend = async () => {
    const trimmedInput = inputValue.trim()
    if (!trimmedInput || isLoading) return

    setError(null)
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // Reset animation states and show thinking animation
    setPendingHeaders(null)
    setShowHeaderReveal(false)
    setShowThinking(true)

    const assistantMessageId = generateId()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      abortControllerRef.current = new AbortController()

      const chatMessages = [
        { role: 'system', content: systemPrompt },
        ...messages.map(m => ({ role: m.role, content: m.content })),
        { role: 'user', content: trimmedInput },
      ]

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages: chatMessages,
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`API error: ${response.status} - ${errorText}`)
      }

      // Extract key headers from response
      const responseHeaders: Record<string, string> = {}
      const headerKeys = [
        'x-vsr-selected-model',
        'x-vsr-selected-decision',
        'x-vsr-cache-hit',
        'x-vsr-selected-reasoning',
        'x-vsr-jailbreak-blocked',
        'x-vsr-pii-violation',
        'x-vsr-hallucination-detected',
        'x-vsr-fact-check-needed',
        'x-vsr-matched-keywords',
        'x-vsr-matched-embeddings',
        'x-vsr-matched-domains',
        'x-vsr-matched-fact-check',
        'x-vsr-matched-user-feedback',
        'x-vsr-matched-preference',
        'x-vsr-matched-language',
        // Looper headers
        'x-vsr-looper-model',
        'x-vsr-looper-models-used',
        'x-vsr-looper-iterations',
        'x-vsr-looper-algorithm',
      ]

      headerKeys.forEach(key => {
        const value = response.headers.get(key)
        if (value) {
          responseHeaders[key] = value
        }
      })

      // Store headers and immediately hide thinking animation
      if (Object.keys(responseHeaders).length > 0) {
        console.log('Headers received, hiding thinking animation')
        setPendingHeaders(responseHeaders)
        setShowThinking(false)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      // Track content for each choice (for ratings mode)
      const choiceContents: Map<number, { content: string; model?: string }> = new Map()
      // Check if this is ratings mode (multiple choices)
      let isRatingsMode = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              const choices = parsed.choices || []

              // Detect ratings mode (multiple choices)
              if (choices.length > 1) {
                isRatingsMode = true
              }

              // Process each choice
              for (const choice of choices) {
                const index = choice.index ?? 0
                const content = choice.delta?.content || ''
                const model = choice.model

                if (!choiceContents.has(index)) {
                  choiceContents.set(index, { content: '', model })
                }

                const current = choiceContents.get(index)!
                if (content) {
                  current.content += content
                }
                if (model && !current.model) {
                  current.model = model
                }
              }

              // Update message state
              if (isRatingsMode) {
                // Ratings mode: update choices array
                const choicesArray: Choice[] = []
                choiceContents.forEach((value, index) => {
                  choicesArray[index] = { content: value.content, model: value.model }
                })
                setMessages(prev =>
                  prev.map(m =>
                    m.id === assistantMessageId
                      ? { ...m, content: choicesArray[0]?.content || '', choices: choicesArray }
                      : m
                  )
                )
              } else {
                // Single choice mode
                const firstChoice = choiceContents.get(0)
                setMessages(prev =>
                  prev.map(m =>
                    m.id === assistantMessageId
                      ? { ...m, content: firstChoice?.content || '' }
                      : m
                  )
                )
              }
            } catch {
              // Skip malformed JSON chunks
            }
          }
        }
      }

      // Finalize message
      const finalChoices: Choice[] | undefined = isRatingsMode
        ? Array.from(choiceContents.entries())
            .sort(([a], [b]) => a - b)
            .map(([, v]) => ({ content: v.content, model: v.model }))
        : undefined

      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessageId
            ? {
                ...m,
                isStreaming: false,
                headers: Object.keys(responseHeaders).length > 0 ? responseHeaders : undefined,
                choices: finalChoices
              }
            : m
        )
      )
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      setMessages(prev => prev.filter(m => m.id !== assistantMessageId))
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleStop = () => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
  }

  const handleClear = () => {
    setMessages([])
    setError(null)
  }

  return (
    <>
      {/* Thinking Animation */}
      {showThinking && (
        <ThinkingAnimation onComplete={handleThinkingComplete} />
      )}

      {/* Header Reveal */}
      {showHeaderReveal && pendingHeaders && (
        <HeaderReveal
          headers={pendingHeaders}
          onComplete={handleHeaderRevealComplete}
          displayDuration={2000}
        />
      )}

      <div className={`${styles.container} ${isFullscreen ? styles.fullscreen : ''}`}>
      {showSettings && (
        <div className={styles.settings}>
          <div className={styles.settingsHeader}>
            <span className={styles.settingsTitle}>Settings</span>
            <button
              className={styles.iconButton}
              onClick={() => setShowSettings(false)}
              title="Close settings"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M1 1l12 12M13 1L1 13" strokeLinecap="round"/>
              </svg>
            </button>
          </div>
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>Model:</label>
            <input
              type="text"
              value={model}
              onChange={e => setModel(e.target.value)}
              className={styles.settingInput}
              placeholder="auto, gpt-4, etc."
            />
          </div>
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>System Prompt:</label>
            <textarea
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              className={styles.settingTextarea}
              rows={3}
              placeholder="You are a helpful assistant."
            />
          </div>
        </div>
      )}

      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
          <button
            className={styles.errorDismiss}
            onClick={() => setError(null)}
          >
            ×
          </button>
        </div>
      )}

      <div className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.3">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <h3>Start a conversation</h3>
            <p>Send a message to begin chatting with the mixture of models.</p>
          </div>
        ) : (
          <div className={styles.messages}>
            {messages.map(message => (
              <div
                key={message.id}
                className={`${styles.message} ${styles[message.role]}`}
              >
                <div className={styles.messageAvatar}>
                  {message.role === 'user' ? (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  ) : (
                    <img src="/vllm.png" alt="vLLM SR" className={styles.avatarImage} />
                  )}
                </div>
                <div className={styles.messageContent}>
                  <div className={styles.messageRole}>
                    {message.role === 'user' ? 'You' : 'vLLM SR'}
                  </div>
                  {/* Ratings mode: multiple choices */}
                  {message.role === 'assistant' && message.choices && message.choices.length > 1 ? (
                    <div className={styles.ratingsChoices}>
                      {message.choices.map((choice, idx) => (
                        <div key={idx} className={styles.choiceCard}>
                          <div className={styles.choiceHeader}>
                            <span className={styles.choiceModel}>{choice.model || `Model ${idx + 1}`}</span>
                            <span className={styles.choiceIndex}>Choice {idx + 1}</span>
                          </div>
                          <div className={styles.choiceContent}>
                            <MarkdownRenderer content={choice.content} />
                            {message.isStreaming && idx === 0 && (
                              <span className={styles.cursor}>▊</span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    /* Single choice mode */
                    <div className={styles.messageText}>
                      {message.role === 'assistant' && message.content ? (
                        <>
                          <MarkdownRenderer content={message.content} />
                          {message.isStreaming && (
                            <span className={styles.cursor}>▊</span>
                          )}
                        </>
                      ) : (
                        <>
                          {message.content || (message.isStreaming && (
                            <span className={styles.cursor}>▊</span>
                          ))}
                          {message.isStreaming && message.content && (
                            <span className={styles.cursor}>▊</span>
                          )}
                        </>
                      )}
                    </div>
                  )}
                  {message.role === 'assistant' && message.headers && (
                    <HeaderDisplay headers={message.headers} />
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className={styles.inputContainer}>
        <div className={styles.inputActions}>
          <button
            className={styles.inputActionButton}
            onClick={() => setShowSettings(!showSettings)}
            title="Settings"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="8" cy="8" r="2.5"/>
              <path d="M8 1v2M8 13v2M15 8h-2M3 8H1M13.5 2.5l-1.4 1.4M3.9 12.1l-1.4 1.4M13.5 13.5l-1.4-1.4M3.9 3.9L2.5 2.5" strokeLinecap="round"/>
            </svg>
          </button>
          <button
            className={styles.inputActionButton}
            onClick={handleClear}
            title="Clear chat"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 4h12M5.5 4V2.5h5V4M13 4v9.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4M6.5 7v4M9.5 7v4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send)"
          className={styles.input}
          rows={1}
          disabled={isLoading}
        />
        {isLoading ? (
          <button
            className={`${styles.sendButton} ${styles.stopButton}`}
            onClick={handleStop}
            title="Stop generating"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="1"/>
            </svg>
          </button>
        ) : (
          <button
            className={styles.sendButton}
            onClick={handleSend}
            disabled={!inputValue.trim()}
            title="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        )}
      </div>
    </div>
    </>
  )
}

export default ChatComponent

