import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import styles from './MarkdownRenderer.module.css'
import type { Components } from 'react-markdown'

interface MarkdownRendererProps {
  content: string
}

const MarkdownRenderer = ({ content }: MarkdownRendererProps) => {
  const components: Components = {
    // Customize code blocks
    code({ className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      const isInline = !className

      return !isInline ? (
        <div className={styles.codeBlock}>
          {match && <div className={styles.codeLanguage}>{match[1]}</div>}
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
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
          {children}
        </a>
      )
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

