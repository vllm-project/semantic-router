import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import styles from './MarkdownRenderer.module.css'
import type { Components } from 'react-markdown'
import { copyText } from '../utils/clipboard'

interface MarkdownRendererProps {
  content: string
  allowImages?: boolean
}

// Copy button component for code blocks
const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!(await copyText(text))) return
    setCopied(true)
    window.setTimeout(() => setCopied(false), 2000)
  }, [text])

  return (
    <button
      type="button"
      className={styles.copyButton}
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy code'}
      aria-label={copied ? 'Copied!' : 'Copy code'}
      aria-live="polite"
    >
      {copied ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <rect x="9" y="9" width="13" height="13" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  )
}

// Extract text content from React children (handles nested highlight spans)
const extractTextContent = (node: React.ReactNode): string => {
  if (node == null || node === undefined) return ''
  if (typeof node === 'string') return node
  if (typeof node === 'number') return String(node)
  if (typeof node === 'boolean') return ''
  if (Array.isArray(node)) return node.map(extractTextContent).join('')
  // Handle React elements (e.g., <span> from syntax highlighting)
  if (typeof node === 'object') {
    // Check if it's a React element with props
    if ('props' in node && node.props) {
      return extractTextContent((node as React.ReactElement).props.children)
    }
    // Check if it has a toString method (fallback)
    if ('toString' in node && typeof node.toString === 'function') {
      const str = node.toString()
      if (str !== '[object Object]') return str
    }
  }
  return ''
}

const MarkdownRenderer = ({ content, allowImages = true }: MarkdownRendererProps) => {
  const components: Components = {
    // Customize code blocks
    code({ className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      const isInline = !className
      const codeText = extractTextContent(children).replace(/\n$/, '')

      return !isInline ? (
        <div className={styles.codeBlock}>
          <div className={styles.codeHeader}>
            {match && <span className={styles.codeLanguage}>{match[1]}</span>}
            <CopyButton text={codeText} />
          </div>
          <code className={className} {...props}>
            {children}
          </code>
        </div>
      ) : (
        <code className={styles.inlineCode} {...props}>
          {children}
        </code>
      )
    },
    // Customize links to open in new tab
    a({ children, href, ...props }) {
      const linkText = extractTextContent(children).trim()
      const isCitationLink = /^\[\d+\]$/.test(linkText)

      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={isCitationLink ? styles.citationLink : undefined}
          {...props}
        >
          {children}
        </a>
      )
    },
    img({ alt, ...props }) {
      if (!allowImages) {
        return (
          <span title="Images are not loaded from untrusted Markdown content">
            Image omitted{alt ? `: ${alt}` : ''}
          </span>
        )
      }
      return <img {...props} alt={alt ?? ''} />
    },
    // Customize tables
    table({ children, ...props }) {
      return (
        <div className={styles.tableWrapper}>
          <table {...props}>{children}</table>
        </div>
      )
    },
  }

  return (
    <div className={styles.markdown}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default MarkdownRenderer
