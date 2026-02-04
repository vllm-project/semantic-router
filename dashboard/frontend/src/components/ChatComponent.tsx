import { useState, useRef, useEffect, useCallback, memo, useMemo } from 'react'
import styles from './ChatComponent.module.css'
import HeaderDisplay from './HeaderDisplay'
import MarkdownRenderer from './MarkdownRenderer'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'
import ThinkingBlock from './ThinkingBlock'
import ErrorBoundary from './ErrorBoundary'
import ReMoMResponsesDisplay from './ReMoMResponsesDisplay'
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

// ReMoM intermediate response structure
interface ReMoMIntermediateResp {
  model: string
  content: string
  reasoning?: string
  compacted_content?: string
  token_count?: number
}

interface ReMoMRoundResponse {
  round: number
  breadth: number
  responses: ReMoMIntermediateResp[]
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
  // Thinking process (from reasoning_content field)
  thinkingProcess?: string
  // Tool calls and results
  toolCalls?: ToolCall[]
  toolResults?: ToolResult[]
  // For ReMoM: intermediate responses from multi-round reasoning
  reasoning_mom_responses?: ReMoMRoundResponse[]
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
                <span className={styles.sourcePillDomain}>{(() => { try { return new URL(result.url).hostname } catch { return result.url } })()}</span>
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

// Open Web Card Component - displays webpage content extraction
const OpenWebCard = ({ 
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
  // Safely parse arguments
  let url = ''
  try {
    const args = JSON.parse(toolCall.function.arguments || '{}')
    url = args.url || ''
  } catch {
    const match = toolCall.function.arguments?.match(/"url"\s*:\s*"([^"]*)/)
    url = (match && match[1]) || 'Loading...'
  }

  // Extract domain from URL
  const domain = useMemo(() => {
    try {
      return new URL(url).hostname
    } catch {
      return url
    }
  }, [url])

  // Get result data
  const resultData = useMemo(() => {
    if (!toolResult?.content) return null
    if (typeof toolResult.content === 'object' && toolResult.content !== null) {
      const data = toolResult.content as { title?: string; content?: string; length?: number; truncated?: boolean }
      return data
    }
    return null
  }, [toolResult?.content])
  
  return (
    <div className={styles.webSearchCard}>
      <div className={styles.webSearchHeader} onClick={onToggle}>
        <div className={styles.webSearchIcon}>
          {toolCall.status === 'running' ? (
            <svg className={styles.searchSpinner} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          )}
        </div>
        <div className={styles.webSearchInfo}>
          <span className={styles.webSearchTitle}>
            {toolCall.status === 'running' ? 'Opening page...' : 'Web Page'}
          </span>
          <span className={styles.webSearchQuery}>{domain}</span>
        </div>
        <div className={styles.webSearchStatus}>
          {toolCall.status === 'completed' && resultData && (
            <span className={styles.webSearchCount}>
              {resultData.length ? `${Math.round(resultData.length / 1000)}k chars` : ''}
              {resultData.truncated ? ' (truncated)' : ''}
            </span>
          )}
          {toolCall.status === 'failed' && (
            <span className={styles.webSearchCount} style={{ color: 'var(--color-error)' }}>Failed</span>
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
      
      {isExpanded && toolCall.status === 'completed' && resultData && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourceDetails}>
            <div className={styles.sourceItem}>
              <div className={styles.sourceItemHeader}>
                <a 
                  href={url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className={styles.sourceItemTitle}
                >
                  {resultData.title || 'Untitled'}
                </a>
              </div>
              <div className={styles.openWebContent}>
                {resultData.content?.substring(0, 500)}
                {(resultData.content?.length || 0) > 500 && '...'}
              </div>
            </div>
          </div>
        </div>
      )}

      {isExpanded && toolCall.status === 'failed' && toolResult?.error && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourceDetails}>
            <div className={styles.sourceItem}>
              <p className={styles.sourceItemSnippet} style={{ color: 'var(--color-error)' }}>
                {toolResult.error}
              </p>
            </div>
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

// Generic Tool Card - routes to specific card based on tool type
const ToolCard = ({ 
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
  const toolName = toolCall.function.name

  if (toolName === 'search_web') {
    return (
      <WebSearchCard 
        toolCall={toolCall} 
        toolResult={toolResult} 
        isExpanded={isExpanded} 
        onToggle={onToggle} 
      />
    )
  }

  if (toolName === 'open_web') {
    return (
      <OpenWebCard 
        toolCall={toolCall} 
        toolResult={toolResult} 
        isExpanded={isExpanded} 
        onToggle={onToggle} 
      />
    )
  }

  // Fallback for unknown tools
  return (
    <div className={styles.webSearchCard}>
      <div className={styles.webSearchHeader} onClick={onToggle}>
        <div className={styles.webSearchIcon}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </svg>
        </div>
        <div className={styles.webSearchInfo}>
          <span className={styles.webSearchTitle}>{toolName}</span>
          <span className={styles.webSearchQuery}>{toolCall.status}</span>
        </div>
      </div>
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
      className={`${styles.inputActionButton} ${enabled ? styles.searchToggleActive : ''}`}
      onClick={onToggle}
      disabled={disabled}
      data-tooltip={enabled ? 'Web Search enabled' : 'Enable Web Search'}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M2 12h20" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
      </svg>
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
            title={source ? `${source.title} - ${(() => { try { return new URL(source.url).hostname } catch { return source.url } })()}` : undefined}
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
  isFullscreenMode?: boolean
}

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  isFullscreenMode = false,
}: ChatComponentProps) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const model = 'MoM' // Fixed to MoM
  const [error, setError] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const [showHeaderReveal, setShowHeaderReveal] = useState(false)
  const [pendingHeaders, setPendingHeaders] = useState<Record<string, string> | null>(null)
  const [isFullscreen] = useState(isFullscreenMode)
const [enableWebSearch, setEnableWebSearch] = useState(true)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Tool Registry integration
  // Search tools (controlled by web search toggle)
  const { definitions: searchToolDefinitions } = useToolRegistry({
    enabledOnly: true,
    categories: ['search'],
  })
  // Other tools (always available, not controlled by web search toggle)
  const { definitions: otherToolDefinitions, executeAll: executeTools } = useToolRegistry({
    enabledOnly: true,
    categories: ['code', 'file', 'image', 'custom'],
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

      // Build chat messages with proper tool call history
      // This ensures the model knows which tool calls have been completed
      type ChatMessage = {
        role: string
        content: string | null
        tool_calls?: Array<{
          id: string
          type: 'function'
          function: { name: string; arguments: string }
        }>
        tool_call_id?: string
      }

      const chatMessages: ChatMessage[] = []

      // Process each message for context
      // IMPORTANT: For history messages, we only include the final text content,
      // NOT the tool calls and results. This prevents context pollution where
      // the model might be confused by previous tool usage when answering new questions.
      for (const m of messages) {
        if (m.role === 'user') {
          chatMessages.push({ role: 'user', content: m.content })
        } else if (m.role === 'assistant') {
          // For assistant messages, only include the final text content
          // Don't include tool_calls or tool results in history
          // This keeps the context clean for new questions
          if (m.content) {
            chatMessages.push({ role: 'assistant', content: m.content })
          }
        }
      }

      // Add the new user message
      chatMessages.push({ role: 'user', content: trimmedInput })

      // Build request body
      const requestBody: Record<string, unknown> = {
        model,
        messages: chatMessages,
        stream: true,
      }

      // Add tools to request:
      // - Search tools: only when web search is enabled
      // - Other tools: always available
      const activeTools = [
        ...otherToolDefinitions,
        ...(enableWebSearch ? searchToolDefinitions : []),
      ]
      if (activeTools.length > 0) {
        requestBody.tools = activeTools
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
        'x-vsr-matched-context',
        'x-vsr-context-token-count',
        'x-vsr-matched-complexity',
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
      // Track content and reasoning for each choice (for ratings mode)
      const choiceContents: Map<number, { content: string; reasoningContent: string; model?: string }> = new Map()
      // Check if this is ratings mode (multiple choices)
      let isRatingsMode = false
      // Track tool calls
      const toolCallsMap: Map<number, ToolCall> = new Map()
      let hasToolCalls = false
      // Track ReMoM intermediate responses
      let reasoningMomResponses: ReMoMRoundResponse[] | undefined
      // Buffer for incomplete lines
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk
        const lines = buffer.split('\n')

        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              const choices = parsed.choices || []

              // Extract reasoning_mom_responses if present (ReMoM algorithm)
              if (parsed.reasoning_mom_responses) {
                reasoningMomResponses = parsed.reasoning_mom_responses
                console.log('[ReMoM] Extracted reasoning_mom_responses:', reasoningMomResponses)
              }

              // Detect ratings mode (multiple choices)
              if (choices.length > 1) {
                isRatingsMode = true
              }

              // Process each choice
              for (const choice of choices) {
                const index = choice.index ?? 0
                const content = choice.delta?.content || ''
                const reasoningContent = choice.delta?.reasoning_content || ''
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
                  choiceContents.set(index, { content: '', reasoningContent: '', model })
                }

                const current = choiceContents.get(index)!
                if (content) {
                  current.content += content
                }
                if (reasoningContent) {
                  current.reasoningContent += reasoningContent
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
                    choicesArray[index] = { content: value.content, model: value.model }
                  })

                  // Get thinking process (reasoning_content) from first choice
                  const firstChoice = choiceContents.get(0)
                  const thinkingProcess = firstChoice?.reasoningContent || ''

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
                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantMessageId
                          ? {
                              ...m,
                              content: firstChoice.content,
                              thinkingProcess: firstChoice.reasoningContent,
                              isStreaming: true
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

      // If we had tool calls, execute tools in a loop until model gives final answer
      if (hasToolCalls) {
        // Maximum tool call iterations to prevent infinite loops
const MAX_TOOL_ITERATIONS = 30
        let iteration = 0
        
        // Accumulated tool calls and results across all iterations
        let allToolCalls = Array.from(toolCallsMap.values())
        let allToolResults: ToolResult[] = []
        
        // Track final content from tool loop
        let finalContent = ''
        
        // Current conversation state for tool loop - use chatMessages as base (already built correctly)
        type ChatMessage = { role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string }
        let currentMessages: ChatMessage[] = [...chatMessages]

        while (iteration < MAX_TOOL_ITERATIONS) {
          iteration++
          console.log(`Tool iteration ${iteration}/${MAX_TOOL_ITERATIONS}`)

          // Get current tool calls to execute
          const currentToolCalls = iteration === 1 
            ? allToolCalls 
            : Array.from(toolCallsMap.values())
          
          if (currentToolCalls.length === 0) break

          // Mark all tools as running
          currentToolCalls.forEach(tc => { tc.status = 'running' })
          
          // Update UI with current tool calls
          const uiToolCalls = [...allToolCalls]
          if (iteration > 1) {
            // Add new tool calls from subsequent iterations
            currentToolCalls.forEach(tc => {
              if (!uiToolCalls.find(t => t.id === tc.id)) {
                uiToolCalls.push(tc)
              }
            })
            allToolCalls = uiToolCalls
          }
          
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, toolCalls: [...uiToolCalls] }
                : m
            )
          )

          // Execute all current tools in parallel
          const toolResults = await executeTools(currentToolCalls, {
            signal: abortControllerRef.current?.signal,
          })

          // Update tool statuses based on results
          toolResults.forEach(result => {
            const tc = currentToolCalls.find(t => t.id === result.callId)
            if (tc) {
              tc.status = result.error ? 'failed' : 'completed'
            }
          })
          
          // Accumulate results
          allToolResults = [...allToolResults, ...toolResults]

          // Update message with completed tool calls and all results
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, toolCalls: [...uiToolCalls], toolResults: allToolResults }
                : m
            )
          )

          // Auto-expand the first tool card
          if (uiToolCalls.length > 0 && expandedToolCards.size === 0) {
            setExpandedToolCards(new Set([uiToolCalls[0].id]))
          }

          // Build messages for next API call
          currentMessages = [
            ...currentMessages,
            // Assistant message with tool_calls
            {
              role: 'assistant',
              content: null,
              tool_calls: currentToolCalls.map(tc => ({
                id: tc.id,
                type: 'function',
                function: {
                  name: tc.function.name,
                  arguments: tc.function.arguments
                }
              }))
            },
            // Tool results (truncate long content to avoid exceeding model context)
            ...toolResults.map(tr => {
              const MAX_TOOL_RESULT_LENGTH = 15000 // ~4k tokens
              let content = typeof tr.content === 'string' 
                ? tr.content 
                : JSON.stringify(tr.content)
              
              // Truncate if too long
              if (content.length > MAX_TOOL_RESULT_LENGTH) {
                content = content.substring(0, MAX_TOOL_RESULT_LENGTH) + '\n\n...[Content truncated due to length]'
                console.log(`Tool result for ${tr.name} truncated from ${typeof tr.content === 'string' ? tr.content.length : JSON.stringify(tr.content).length} to ${MAX_TOOL_RESULT_LENGTH} chars`)
              }
              
              return {
                role: 'tool',
                tool_call_id: tr.callId,
                content
              }
            })
          ]

          // Make follow-up API call with tools enabled
          const followUpResponse = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model,
              messages: currentMessages,
              stream: true,
              // Keep tools enabled for multi-step tool usage (same logic as initial request)
              tools: activeTools,
              tool_choice: 'auto',
            }),
            signal: abortControllerRef.current?.signal,
          })

          if (!followUpResponse.ok) {
            console.error('Follow-up API call failed:', followUpResponse.status, followUpResponse.statusText)
            break
          }

          if (!followUpResponse.body) break

          const followUpReader = followUpResponse.body.getReader()
          const followUpDecoder = new TextDecoder()
          let followUpContent = ''
          let hasMoreToolCalls = false
          let streamFinishReason = ''
          
          // Reset tool calls map for this iteration
          toolCallsMap.clear()

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
                  const deltaToolCalls = delta?.tool_calls
                  const finishReason = parsed.choices?.[0]?.finish_reason

                  // Capture finish reason when present
                  if (finishReason) {
                    streamFinishReason = finishReason
                    console.log(`Iteration ${iteration} finish_reason: ${finishReason}, hasContent: ${followUpContent.length > 0}`)
                  }

                  // Check for new tool calls
                  if (deltaToolCalls && Array.isArray(deltaToolCalls)) {
                    hasMoreToolCalls = true
                    for (const tc of deltaToolCalls) {
                      const tcIndex = tc.index ?? 0
                      if (!toolCallsMap.has(tcIndex)) {
                        toolCallsMap.set(tcIndex, {
                          id: tc.id || `tool-${iteration}-${tcIndex}`,
                          type: 'function',
                          function: {
                            name: tc.function?.name || '',
                            arguments: ''
                          },
                          status: 'pending'
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
                  }

                  // Accumulate content
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

          // Save content from this iteration
          if (followUpContent) {
            finalContent = followUpContent
            console.log(`Iteration ${iteration} content: ${followUpContent.substring(0, 100)}`)
          }

          // Check if we should continue the loop
          if (streamFinishReason === 'tool_calls' && toolCallsMap.size > 0) {
            // Model wants to call more tools, continue loop
            console.log(`Model requested ${toolCallsMap.size} more tool call(s) (finish_reason: tool_calls), will continue loop`)
            continue
          } else if (streamFinishReason === 'stop' || streamFinishReason === 'length') {
            // Model finished, exit loop
            console.log(`Model finished (finish_reason: ${streamFinishReason}), exiting tool loop`)
            break
          } else if (!hasMoreToolCalls) {
            // No more tool calls, exit loop
            console.log('No more tool calls detected, exiting tool loop')
            break
          }
          
          console.log(`Default case: hasMoreToolCalls=${hasMoreToolCalls}, finish_reason=${streamFinishReason}, continuing`)
        }

        if (iteration >= MAX_TOOL_ITERATIONS) {
          console.warn('Reached maximum tool iterations, stopping')
        }
        
        // Ensure final content is set after all tool iterations
        console.log('Tool loop finished, final content length:', finalContent.length)
        if (finalContent) {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: finalContent }
                : m
            )
          )
        } else {
          // If no content after tool loop, generate a fallback summary from tool results
          console.warn('Tool loop finished but no content received from model, generating fallback summary')
          
          // Generate fallback content based on tool results
          let fallbackContent = ''
          if (allToolResults.length > 0) {
            const successResults = allToolResults.filter(tr => !tr.error)
            const failedResults = allToolResults.filter(tr => tr.error)
            
            if (successResults.length > 0) {
              fallbackContent = '基于搜索结果，以下是相关信息：\n\n'
              for (const tr of successResults) {
                if (typeof tr.content === 'string' && tr.content.length > 0) {
                  // 截取前 500 字符作为摘要
                  const summary = tr.content.length > 500 
                    ? tr.content.substring(0, 500) + '...' 
                    : tr.content
                  fallbackContent += summary + '\n\n'
                }
              }
            }
            
            if (failedResults.length > 0 && !fallbackContent) {
              fallbackContent = '抱歉，部分工具执行失败。请尝试重新查询或使用其他关键词。'
            }
          }
          
          if (!fallbackContent) {
            fallbackContent = '模型没有生成响应内容，请尝试重新提问。'
          }
          
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: fallbackContent }
                : m
            )
          )
        }
      }

      // Finalize message
      const finalChoices: Choice[] | undefined = isRatingsMode
        ? Array.from(choiceContents.entries())
            .sort(([a], [b]) => a - b)
            .map(([, v]) => ({ content: v.content, model: v.model }))
        : undefined

      // Get final thinking process from reasoning_content
      const finalThinkingProcess = choiceContents.get(0)?.reasoningContent || ''

      // Streaming finished - no need to control ThinkingAnimation here
      // It was already hidden when headers arrived

      console.log('[ReMoM] Setting reasoning_mom_responses:', reasoningMomResponses)
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessageId
            ? {
                ...m,
                isStreaming: false,
                headers: Object.keys(responseHeaders).length > 0 ? responseHeaders : undefined,
                choices: finalChoices,
                thinkingProcess: finalThinkingProcess || m.thinkingProcess,
                reasoning_mom_responses: reasoningMomResponses
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
                            <ToolCard
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
                      {/* Show tool calls if any - filter out failed ones for cleaner UX */}
                      {message.role === 'assistant' && message.toolCalls && message.toolCalls.length > 0 && (() => {
                        const successfulToolCalls = message.toolCalls.filter(tc => tc.status !== 'failed')
                        const failedCount = message.toolCalls.length - successfulToolCalls.length

                        if (successfulToolCalls.length === 0 && failedCount > 0) {
                          // All failed, show nothing or a minimal indicator
                          return null
                        }

                        return (
                          <div className={styles.toolCallsContainer}>
                            {successfulToolCalls.map(tc => (
                              <ErrorBoundary key={tc.id}>
                                <ToolCard
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
                        )
                      })()}
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
                  {message.role === 'assistant' && message.reasoning_mom_responses && (
                    <>
                      {console.log('[ReMoM] Rendering ReMoMResponsesDisplay for message:', message.id, 'rounds:', message.reasoning_mom_responses)}
                      <ReMoMResponsesDisplay rounds={message.reasoning_mom_responses} />
                    </>
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
        <div className={`${styles.inputWrapper} ${inputValue.trim() ? styles.hasContent : ''}`}>
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
              <ToolToggle
                enabled={enableWebSearch}
                onToggle={() => setEnableWebSearch(!enableWebSearch)}
                disabled={isLoading}
              />
              <button
                className={styles.inputActionButton}
                onClick={handleClear}
                data-tooltip="Clear chat"
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

