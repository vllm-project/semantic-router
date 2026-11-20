import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Check, Copy } from 'lucide-react'
import styles from './MarkdownRenderer.module.css'
import type { Components } from 'react-markdown'

interface MarkdownRendererProps {
  content: string
  allowImages?: boolean
}

// Copy button component for code blocks
const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!text) {
      console.warn('No text to copy')
      return
    }
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        // Fallback for older browsers or non-HTTPS
        const textArea = document.createElement('textarea')
        textArea.value = text
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
  }, [text])

  return (
    <button
      type="button"
      className={styles.copyButton}
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy code'}
      aria-label={copied ? 'Copied!' : 'Copy code'}
    >
      {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
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
          <span title="Images are not loaded from managed Recipe documentation">
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
