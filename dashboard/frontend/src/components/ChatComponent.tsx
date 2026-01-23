import { useState, useRef, useEffect, useCallback, memo, useMemo } from 'react'
import styles from './ChatComponent.module.css'
import HeaderDisplay from './HeaderDisplay'
import MarkdownRenderer from './MarkdownRenderer'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'
import ThinkingBlock from './ThinkingBlock'
import ErrorBoundary from './ErrorBoundary'
import { useToolRegistry } from '../tools'
import { getTranslateAttr } from '../hooks/useNoTranslate'
import type { ToolCall, ToolResult, WebSearchResult } from '../tools'

// Copy button component for copying full response
const CopyResponseButton = ({ copied, onCopy }: { copied: boolean; onCopy: () => void }) => {
  return (
    <button
      className={styles.actionButton}
      onClick={onCopy}
      title={copied ? 'Copied!' : 'Copy'}
      aria-label={copied ? 'Copied!' : 'Copy'}
    >
      {copied ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="9" y="9" width="13" height="13" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  )
}

// Message action bar component
const MessageActionBar = ({ content }: { content: string }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!content) return
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(content)
      } else {
        const textArea = document.createElement('textarea')
        textArea.value = content
        textArea.style.position = 'fixed'
        textArea.style.left = '-9999px'
        document.body.appendChild(textArea)
        textArea.select()
        document.execCommand('copy')
        document.body.removeChild(textArea)
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }, [content])

  return (
    <div className={styles.messageActionBar}>
      <CopyResponseButton copied={copied} onCopy={handleCopy} />
    </div>
  )
}

// Greeting lines - defined outside component to maintain stable reference
const GREETING_LINES = [
  "Hi there, I am MoM :-)",
  "The System Intelligence for LLMs",
  "The World First Model-of-Models",
  "Open Source for Everyone",
  "How can I help you today?"
]

// Typing effect component for greeting with multiple lines
// Memoized to prevent re-renders when parent state changes (e.g., input typing)
const TypingGreeting = memo(({ lines }: { lines: string[] }) => {
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  const [displayedText, setDisplayedText] = useState('')
  const [isTyping, setIsTyping] = useState(true)

  useEffect(() => {
    if (currentLineIndex >= lines.length) return

    const currentLine = lines[currentLineIndex]
    let charIndex = 0
    setIsTyping(true)
    setDisplayedText('')

    const typingInterval = setInterval(() => {
      if (charIndex < currentLine.length) {
        setDisplayedText(currentLine.slice(0, charIndex + 1))
        charIndex++
      } else {
        clearInterval(typingInterval)
        setIsTyping(false)
        // Wait before moving to next line
        setTimeout(() => {
          if (currentLineIndex < lines.length - 1) {
            setCurrentLineIndex(prev => prev + 1)
          }
        }, 1500)
      }
    }, 60)

    return () => clearInterval(typingInterval)
  }, [currentLineIndex, lines])

  return (
    <div className={styles.typingGreeting} translate="no">
      <h2>
        {displayedText}
        {isTyping && <span className={styles.typingCursor}>|</span>}
      </h2>
    </div>
  )
})

// Choice represents a single model's response in ratings mode
interface Choice {
  content: string
  model?: string
}

// Re-export ToolCall and ToolResult types from tools module
// Local SearchResult alias for backward compatibility
type SearchResult = WebSearchResult

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  isStreaming?: boolean
  headers?: Record<string, string>
  // For ratings mode: multiple choices from different models
  choices?: Choice[]
  // Thinking process (content before assistantfinal tag)
  thinkingProcess?: string
  // Tool calls and results
  toolCalls?: ToolCall[]
  toolResults?: ToolResult[]
}

// Web Search Card Component
const WebSearchCard = ({ 
  toolCall, 
  toolResult,
  isExpanded,
  onToggle 
}: { 
  toolCall: ToolCall
  toolResult?: ToolResult
  isExpanded: boolean
  onToggle: () => void
}) => {
  // Safely parse arguments - may be incomplete during streaming
  let query = ''
  try {
    const args = JSON.parse(toolCall.function.arguments || '{}')
    query = args.query || ''
  } catch {
    // Arguments still streaming or invalid, show partial or empty
    const match = toolCall.function.arguments?.match(/"query"\s*:\s*"([^"]*)/)
    query = (match && match[1]) || 'Searching...'
  }
  
  // Safely get results - ensure it's an array
  const results = useMemo(() => {
    if (!toolResult?.content) return undefined
    if (Array.isArray(toolResult.content)) {
      return toolResult.content as SearchResult[]
    }
    // If content is a string (error message), return undefined
    return undefined
  }, [toolResult?.content])
  
  return (
    <div className={styles.webSearchCard}>
      <div className={styles.webSearchHeader} onClick={onToggle}>
        <div className={styles.webSearchIcon}>
          {toolCall.status === 'running' ? (
            <svg className={styles.searchSpinner} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          )}
        </div>
        <div className={styles.webSearchInfo}>
          <span className={styles.webSearchTitle}>
            {toolCall.status === 'running' ? 'Searching...' : 'Web Search'}
          </span>
          <span className={styles.webSearchQuery}>"{query}"</span>
        </div>
        <div className={styles.webSearchStatus}>
          {toolCall.status === 'completed' && results && (
            <span className={styles.webSearchCount}>{results.length} sources</span>
          )}
          <svg 
            className={`${styles.webSearchChevron} ${isExpanded ? styles.expanded : ''}`} 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </div>
      
      {isExpanded && toolCall.status === 'completed' && results && results.length > 0 && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourcePills}>
            {results.map((result, idx) => (
              <a 
                key={idx}
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.sourcePill}
                title={result.snippet}
              >
                <span className={styles.sourcePillNumber}>{idx + 1}</span>
                <span className={styles.sourcePillDomain}>{result.domain}</span>
              </a>
            ))}
          </div>
          <div className={styles.sourceDetails}>
            {results.map((result, idx) => (
              <div key={idx} className={styles.sourceItem}>
                <div className={styles.sourceItemHeader}>
                  <span className={styles.sourceItemNumber}>[{idx + 1}]</span>
                  <a 
                    href={result.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className={styles.sourceItemTitle}
                  >
                    {result.title}
                  </a>
                </div>
                <p className={styles.sourceItemSnippet}>{result.snippet}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {toolCall.status === 'running' && (
        <div className={styles.webSearchLoading}>
          <div className={styles.webSearchLoadingBar} />
        </div>
      )}
    </div>
  )
}

// Tool Toggle Component
const ToolToggle = ({ 
  enabled, 
  onToggle,
  disabled 
}: { 
  enabled: boolean
  onToggle: () => void
  disabled?: boolean
}) => {
  return (
    <button
      className={`${styles.toolToggle} ${enabled ? styles.toolToggleActive : ''}`}
      onClick={onToggle}
      disabled={disabled}
      title={enabled ? 'Web Search enabled' : 'Enable Web Search'}
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="8" />
        <path d="M21 21l-4.35-4.35" />
      </svg>
      <span>Search</span>
    </button>
  )
}

// Citation Link Component - renders [1], [2], etc. as clickable links
const CitationLink = ({ 
  number, 
  url, 
  title 
}: { 
  number: number
  url?: string
  title?: string 
}) => {
  const handleClick = (e: React.MouseEvent) => {
    if (url) {
      e.preventDefault()
      window.open(url, '_blank', 'noopener,noreferrer')
    }
  }

  return (
    <span
      className={styles.citationLink}
      onClick={handleClick}
      title={title || `Source ${number}`}
      role="button"
      tabIndex={0}
    >
      [{number}]
    </span>
  )
}

// Content with Citations - parses [1], [2] etc and renders as clickable links
const ContentWithCitations = ({ 
  content, 
  sources,
  isStreaming = false
}: { 
  content: string
  sources?: SearchResult[] | unknown
  isStreaming?: boolean
}) => {
  // Safely normalize sources to array
  const safeSources = useMemo(() => {
    if (!sources) return undefined
    if (Array.isArray(sources)) return sources as SearchResult[]
    return undefined
  }, [sources])

  // Disable translation during streaming to prevent DOM conflicts
  const translateAttr = getTranslateAttr(isStreaming)

  // Memoize the processed content to avoid re-parsing on every render
  const processedContent = useMemo(() => {
    // Safety check for content - always return consistent structure
    if (!content || typeof content !== 'string') {
      return null
    }

    // Parse content and replace [n] patterns with citation links
    const parseContentWithCitations = (text: string, keyPrefix: string): React.ReactNode[] => {
      const parts: React.ReactNode[] = []
      // Match [1], [2], [3] etc. - citation format
      const citationRegex = /\[(\d+)\]/g
      let lastIndex = 0
      let match
      let keyIndex = 0
      let iterationCount = 0
      const maxIterations = 1000 // Prevent infinite loop

      while ((match = citationRegex.exec(text)) !== null && iterationCount < maxIterations) {
        iterationCount++
        // Add text before the citation
        if (match.index > lastIndex) {
          parts.push(<span key={`${keyPrefix}-text-${keyIndex++}`}>{text.slice(lastIndex, match.index)}</span>)
        }

        const citationNumber = parseInt(match[1], 10)
        const source = safeSources?.[citationNumber - 1] // 1-indexed

        parts.push(
          <CitationLink
            key={`${keyPrefix}-citation-${keyIndex++}`}
            number={citationNumber}
            url={source?.url}
            title={source ? `${source.title} - ${source.domain}` : undefined}
          />
        )

        lastIndex = match.index + match[0].length
      }

      // Add remaining text
      if (lastIndex < text.length) {
        parts.push(<span key={`${keyPrefix}-text-${keyIndex++}`}>{text.slice(lastIndex)}</span>)
      }

      return parts
    }

    // If no sources, just render with MarkdownRenderer wrapped in consistent container
    if (!safeSources || safeSources.length === 0) {
      return <MarkdownRenderer content={content} />
    }

    // Check if content has citations
    const hasCitations = /\[\d+\]/.test(content)
    
    if (!hasCitations) {
      return <MarkdownRenderer content={content} />
    }

    // For content with citations, we need to handle it specially
    // Split by markdown blocks to preserve code blocks etc.
    const lines = content.split('\n')
    const processedLines: React.ReactNode[] = []
    let inCodeBlock = false
    let codeBlockContent = ''
    let codeBlockLang = ''

    lines.forEach((line, lineIndex) => {
      // Check for code block start/end
      if (line.startsWith('```')) {
        if (!inCodeBlock) {
          inCodeBlock = true
          codeBlockLang = line.slice(3).trim()
          codeBlockContent = ''
        } else {
          // End of code block - render as markdown
          processedLines.push(
            <div key={`code-${lineIndex}`} className={styles.codeBlockWrapper}>
              <MarkdownRenderer 
                content={`\`\`\`${codeBlockLang}\n${codeBlockContent}\`\`\``} 
              />
            </div>
          )
          inCodeBlock = false
          codeBlockLang = ''
        }
        return
      }

      if (inCodeBlock) {
        codeBlockContent += (codeBlockContent ? '\n' : '') + line
        return
      }

      // For regular lines, check for citations
      if (/\[\d+\]/.test(line)) {
        // Line has citations - render with citation links
        processedLines.push(
          <p key={`line-${lineIndex}`} className={styles.citationParagraph}>
            {parseContentWithCitations(line, `line-${lineIndex}`)}
          </p>
        )
      } else if (line.trim() === '') {
        // Empty line - add spacer div instead of br for consistent structure
        processedLines.push(<div key={`space-${lineIndex}`} className={styles.lineBreak} />)
      } else {
        // Regular line without citations - use markdown wrapped in div
        processedLines.push(
          <div key={`md-${lineIndex}`} className={styles.markdownLine}>
            <MarkdownRenderer content={line} />
          </div>
        )
      }
    })

    return <>{processedLines}</>
  }, [content, safeSources])

  // Always return consistent div structure
  // Disable translation during streaming to prevent DOM conflicts with browser translators
  return (
    <div className={styles.contentWithCitations} translate={translateAttr}>
      {processedContent}
    </div>
  )
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
  const [enableWebSearch, setEnableWebSearch] = useState(false)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Tool Registry integration
  const { definitions: toolDefinitions, executeAll: executeTools } = useToolRegistry({
    enabledOnly: true,
  })

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

  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`

  // Parse content to separate thinking process from final answer
  const parseStreamingContent = (fullContent: string): {
    hasAnalysis: boolean
    thinking: string
    hasFinal: boolean
    final: string
  } => {
    // Check if content contains "analysis" tag
    const hasAnalysis = /analysis/i.test(fullContent)

    // Look for assistantfinal tag (case insensitive)
    const finalMatch = fullContent.match(/assistantfinal(.*)$/is)

    if (finalMatch) {
      // Found assistantfinal - split content
      const finalIndex = fullContent.search(/assistantfinal/i)

      // Everything from "analysis" to "assistantfinal" is thinking
      const analysisMatch = fullContent.match(/analysis(.*?)assistantfinal/is)
      const thinking = analysisMatch ? analysisMatch[1].trim() : ''

      // Everything after assistantfinal is the final answer
      const final = fullContent.substring(finalIndex + 'assistantfinal'.length).trim()

      return { hasAnalysis, thinking, hasFinal: true, final }
    }

    // If we have analysis but no assistantfinal yet, all content after "analysis" is thinking
    if (hasAnalysis) {
      const analysisMatch = fullContent.match(/analysis(.*)$/is)
      const thinking = analysisMatch ? analysisMatch[1].trim() : ''
      return { hasAnalysis: true, thinking, hasFinal: false, final: '' }
    }

    // No special tags - treat as final answer
    return { hasAnalysis: false, thinking: '', hasFinal: false, final: fullContent }
  }

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

    // Reset animation states and show initial thinking animation (no content)
    setPendingHeaders(null)
    setShowHeaderReveal(false)
    setShowThinking(true)  // Show immediately when user sends message

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

      // Build request body
      const requestBody: Record<string, unknown> = {
        model,
        messages: chatMessages,
        stream: true,
      }

      // Add tools from registry if web search is enabled
      if (enableWebSearch && toolDefinitions.length > 0) {
        requestBody.tools = toolDefinitions
        requestBody.tool_choice = 'auto'
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
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
        'x-vsr-matched-latency',
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

      // Store headers and hide thinking animation, show HeaderReveal
      if (Object.keys(responseHeaders).length > 0) {
        console.log('Headers received, showing HeaderReveal')
        setPendingHeaders(responseHeaders)
        setShowThinking(false)  // Hide full-screen thinking animation
        setShowHeaderReveal(true)  // Show HeaderReveal
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
      // Track tool calls
      const toolCallsMap: Map<number, ToolCall> = new Map()
      let hasToolCalls = false

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
                const deltaToolCalls = choice.delta?.tool_calls

                // Handle tool calls
                if (deltaToolCalls && Array.isArray(deltaToolCalls)) {
                  hasToolCalls = true
                  for (const tc of deltaToolCalls) {
                    const tcIndex = tc.index ?? 0
                    if (!toolCallsMap.has(tcIndex)) {
                      toolCallsMap.set(tcIndex, {
                        id: tc.id || `tool-${tcIndex}`,
                        type: 'function',
                        function: {
                          name: tc.function?.name || '',
                          arguments: ''
                        },
                        status: 'running'
                      })
                    }
                    const existingTc = toolCallsMap.get(tcIndex)!
                    if (tc.function?.name) {
                      existingTc.function.name = tc.function.name
                    }
                    if (tc.function?.arguments) {
                      existingTc.function.arguments += tc.function.arguments
                    }
                    if (tc.id) {
                      existingTc.id = tc.id
                    }
                  }

                  // Update message with tool calls
                  const currentToolCalls = Array.from(toolCallsMap.values())
                  setMessages(prev =>
                    prev.map(m =>
                      m.id === assistantMessageId
                        ? { ...m, toolCalls: currentToolCalls }
                        : m
                    )
                  )
                }

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

              // Update message state (only if we have content, not just tool calls)
              if (!hasToolCalls || choiceContents.get(0)?.content) {
                if (isRatingsMode) {
                  // Ratings mode: update choices array
                  const choicesArray: Choice[] = []

                  choiceContents.forEach((value, index) => {
                    const parsed = parseStreamingContent(value.content)
                    choicesArray[index] = { content: parsed.final, model: value.model }
                  })

                  // Get thinking process from first choice
                  const firstChoiceForThinking = choiceContents.get(0)
                  const thinkingProcess = firstChoiceForThinking
                    ? parseStreamingContent(firstChoiceForThinking.content).thinking
                    : ''

                  setMessages(prev =>
                    prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            content: choicesArray[0]?.content || '',
                            choices: choicesArray,
                            thinkingProcess: thinkingProcess
                          }
                        : m
                    )
                  )
                } else {
                  // Single choice mode
                  const firstChoice = choiceContents.get(0)
                  if (firstChoice) {
                    const parsed = parseStreamingContent(firstChoice.content)

                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantMessageId
                          ? {
                              ...m,
                              content: parsed.final,
                              thinkingProcess: parsed.thinking,
                              isStreaming: !parsed.hasFinal  // Stop streaming when we hit assistantfinal
                            }
                          : m
                      )
                    )
                  }
                }
              }
            } catch {
              // Skip malformed JSON chunks
            }
          }
        }
      }

      // If we had tool calls, execute the tools using Tool Registry
      if (hasToolCalls) {
        // Get the final accumulated tool calls from the map
        const toolCalls = Array.from(toolCallsMap.values())
        
        // Mark all tools as running
        toolCalls.forEach(tc => { tc.status = 'running' })
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? { ...m, toolCalls: [...toolCalls] }
              : m
          )
        )

        // Execute all tools in parallel using Tool Registry
        const toolResults = await executeTools(toolCalls, {
          signal: abortControllerRef.current?.signal,
        })

        // Update tool statuses based on results
        toolResults.forEach(result => {
          const tc = toolCalls.find(t => t.id === result.callId)
          if (tc) {
            tc.status = result.error ? 'failed' : 'completed'
          }
        })

        // Update message with completed tool calls and results
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? { ...m, toolCalls: [...toolCalls], toolResults }
              : m
          )
        )

        // Auto-expand the first tool card
        if (toolCalls.length > 0) {
          setExpandedToolCards(new Set([toolCalls[0].id]))
        }

        // === CRITICAL: Send tool results back to model for final response ===
        // Build messages including tool call and tool results
        const messagesWithToolResults = [
          { role: 'system', content: systemPrompt },
          ...messages.map(m => ({ role: m.role, content: m.content })),
          { role: 'user', content: trimmedInput },
          // Assistant message with tool_calls
          {
            role: 'assistant',
            content: null,
            tool_calls: toolCalls.map(tc => ({
              id: tc.id,
              type: 'function',
              function: {
                name: tc.function.name,
                arguments: tc.function.arguments
              }
            }))
          },
          // Tool results
          ...toolResults.map(tr => ({
            role: 'tool',
            tool_call_id: tr.callId,
            content: typeof tr.content === 'string' 
              ? tr.content 
              : JSON.stringify(tr.content)
          }))
        ]

        // Make second API call to get final response
        const followUpResponse = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model,
            messages: messagesWithToolResults,
            stream: true,
          }),
          signal: abortControllerRef.current?.signal,
        })

        if (!followUpResponse.ok) {
          console.error('Follow-up API call failed:', followUpResponse.status, followUpResponse.statusText)
          // Don't throw - we already have tool results to show
        } else if (followUpResponse.body) {
          const followUpReader = followUpResponse.body.getReader()
          const followUpDecoder = new TextDecoder()
          let followUpContent = ''

          while (true) {
            const { done, value } = await followUpReader.read()
            if (done) break

            const chunk = followUpDecoder.decode(value, { stream: true })
            const lines = chunk.split('\n').filter(line => line.trim() !== '')

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') continue

                try {
                  const parsed = JSON.parse(data)
                  const delta = parsed.choices?.[0]?.delta

                  if (delta?.content) {
                    followUpContent += delta.content
                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantMessageId
                          ? { ...m, content: followUpContent }
                          : m
                      )
                    )
                  }
                } catch {
                  // Ignore parse errors
                }
              }
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

      // Streaming finished - no need to control ThinkingAnimation here
      // It was already hidden when headers arrived

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
        <ThinkingAnimation
          onComplete={handleThinkingComplete}
          thinkingProcess=""
        />
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
            <TypingGreeting lines={GREETING_LINES} />
          </div>
        ) : (
          <div className={styles.messages}>
            {messages.map(message => (
              <div
                key={message.id}
                className={`${styles.message} ${styles[message.role]}`}
                // Disable translation during streaming to prevent DOM conflicts
                translate={getTranslateAttr(message.isStreaming ?? false)}
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
                    <>
                      {/* Show tool calls if any */}
                      {message.toolCalls && message.toolCalls.length > 0 && (
                        <div className={styles.toolCallsContainer}>
                          {message.toolCalls.map(tc => (
                            <WebSearchCard
                              key={tc.id}
                              toolCall={tc}
                              toolResult={message.toolResults?.find(tr => tr.callId === tc.id)}
                              isExpanded={expandedToolCards.has(tc.id)}
                              onToggle={() => {
                                setExpandedToolCards(prev => {
                                  const next = new Set(prev)
                                  if (next.has(tc.id)) {
                                    next.delete(tc.id)
                                  } else {
                                    next.add(tc.id)
                                  }
                                  return next
                                })
                              }}
                            />
                          ))}
                        </div>
                      )}
                      {/* Show thinking block if available */}
                      {message.thinkingProcess && (
                        <ThinkingBlock
                          content={message.thinkingProcess}
                          isStreaming={message.isStreaming}
                        />
                      )}
                      <div className={styles.ratingsChoices}>
                        {message.choices.map((choice, idx) => (
                          <div key={idx} className={styles.choiceCard}>
                            <div className={styles.choiceHeader}>
                              <span className={styles.choiceModel}>{choice.model || `Model ${idx + 1}`}</span>
                              <span className={styles.choiceIndex}>Choice {idx + 1}</span>
                            </div>
                            <div className={styles.choiceContent}>
                              <ErrorBoundary>
                                <ContentWithCitations 
                                  content={choice.content}
                                  sources={
                                    message.toolResults?.find(tr => tr.name === 'search_web')?.content
                                  }
                                  isStreaming={message.isStreaming}
                                />
                              </ErrorBoundary>
                              {message.isStreaming && idx === 0 && (
                                <span className={styles.cursor}>▊</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    /* Single choice mode */
                    <>
                      {/* Show tool calls if any */}
                      {message.role === 'assistant' && message.toolCalls && message.toolCalls.length > 0 && (
                        <div className={styles.toolCallsContainer}>
                          {message.toolCalls.map(tc => (
                            <ErrorBoundary key={tc.id}>
                              <WebSearchCard
                                toolCall={tc}
                                toolResult={message.toolResults?.find(tr => tr.callId === tc.id)}
                                isExpanded={expandedToolCards.has(tc.id)}
                                onToggle={() => {
                                  setExpandedToolCards(prev => {
                                    const next = new Set(prev)
                                    if (next.has(tc.id)) {
                                      next.delete(tc.id)
                                    } else {
                                      next.add(tc.id)
                                    }
                                    return next
                                  })
                                }}
                              />
                            </ErrorBoundary>
                          ))}
                        </div>
                      )}
                      {/* Show thinking block if available */}
                      {message.role === 'assistant' && message.thinkingProcess && (
                        <ThinkingBlock
                          content={message.thinkingProcess}
                          isStreaming={message.isStreaming}
                        />
                      )}
                      <div className={styles.messageText}>
                        {message.role === 'assistant' && message.content ? (
                          <>
                            <ErrorBoundary>
                              <ContentWithCitations 
                                content={message.content} 
                                sources={
                                  message.toolResults?.find(tr => tr.name === 'search_web')?.content
                                }
                                isStreaming={message.isStreaming}
                              />
                            </ErrorBoundary>
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
                    </>
                  )}
                  {message.role === 'assistant' && message.headers && (
                    <HeaderDisplay headers={message.headers} />
                  )}
                  {message.role === 'assistant' && message.content && !message.isStreaming && (
                    <MessageActionBar content={message.content} />
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className={styles.inputContainer}>
        <div className={styles.inputToolbar}>
          <ToolToggle
            enabled={enableWebSearch}
            onToggle={() => setEnableWebSearch(!enableWebSearch)}
            disabled={isLoading}
          />
        </div>
        <div className={styles.inputWrapper}>
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything..."
            className={styles.input}
            rows={1}
            disabled={isLoading}
          />
          <div className={styles.inputActionsRow}>
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
            {isLoading ? (
              <button
                className={`${styles.sendButton} ${styles.stopButton}`}
                onClick={handleStop}
                title="Stop generating"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
              </button>
            ) : (
              <button
                className={styles.sendButton}
                onClick={handleSend}
                disabled={!inputValue.trim()}
                title="Send message"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
    </>
  )
}

export default ChatComponent

