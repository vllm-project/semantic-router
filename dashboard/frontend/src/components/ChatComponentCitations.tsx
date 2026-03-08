import { useMemo } from 'react'

import { getTranslateAttr } from '../hooks/useNoTranslate'

import MarkdownRenderer from './MarkdownRenderer'
import styles from './ChatComponent.module.css'
import type { SearchResult } from './ChatComponentTypes'

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

export const ContentWithCitations = ({
  content,
  sources,
  isStreaming = false
}: {
  content: string
  sources?: SearchResult[] | unknown
  isStreaming?: boolean
}) => {
  const safeSources = useMemo(() => {
    if (!sources) return undefined
    if (Array.isArray(sources)) return sources as SearchResult[]
    return undefined
  }, [sources])

  const translateAttr = getTranslateAttr(isStreaming)

  const processedContent = useMemo(() => {
    if (!content || typeof content !== 'string') {
      return null
    }

    const parseContentWithCitations = (text: string, keyPrefix: string): React.ReactNode[] => {
      const parts: React.ReactNode[] = []
      const citationRegex = /\[(\d+)\]/g
      let lastIndex = 0
      let match
      let keyIndex = 0
      let iterationCount = 0
      const maxIterations = 1000

      while ((match = citationRegex.exec(text)) !== null && iterationCount < maxIterations) {
        iterationCount++
        if (match.index > lastIndex) {
          parts.push(<span key={`${keyPrefix}-text-${keyIndex++}`}>{text.slice(lastIndex, match.index)}</span>)
        }

        const citationNumber = parseInt(match[1], 10)
        const source = safeSources?.[citationNumber - 1]

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

      if (lastIndex < text.length) {
        parts.push(<span key={`${keyPrefix}-text-${keyIndex++}`}>{text.slice(lastIndex)}</span>)
      }

      return parts
    }

    if (!safeSources || safeSources.length === 0) {
      return <MarkdownRenderer content={content} />
    }

    if (!/\[\d+\]/.test(content)) {
      return <MarkdownRenderer content={content} />
    }

    const lines = content.split('\n')
    const processedLines: React.ReactNode[] = []
    let inCodeBlock = false
    let codeBlockContent = ''
    let codeBlockLang = ''

    lines.forEach((line, lineIndex) => {
      if (line.startsWith('```')) {
        if (!inCodeBlock) {
          inCodeBlock = true
          codeBlockLang = line.slice(3).trim()
          codeBlockContent = ''
        } else {
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

      if (/\[\d+\]/.test(line)) {
        processedLines.push(
          <p key={`line-${lineIndex}`} className={styles.citationParagraph}>
            {parseContentWithCitations(line, `line-${lineIndex}`)}
          </p>
        )
      } else if (line.trim() === '') {
        processedLines.push(<div key={`space-${lineIndex}`} className={styles.lineBreak} />)
      } else {
        processedLines.push(
          <div key={`md-${lineIndex}`} className={styles.markdownLine}>
            <MarkdownRenderer content={line} />
          </div>
        )
      }
    })

    return <>{processedLines}</>
  }, [content, safeSources])

  return (
    <div className={styles.contentWithCitations} translate={translateAttr}>
      {processedContent}
    </div>
  )
}
