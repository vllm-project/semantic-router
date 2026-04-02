import { useMemo } from 'react'

import { getTranslateAttr } from '../hooks/useNoTranslate'

import MarkdownRenderer from './MarkdownRenderer'
import styles from './ChatComponent.module.css'
import type { SearchResult } from './ChatComponentTypes'

const escapeMarkdownLinkTitle = (value: string): string => (
  value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
)

const injectCitationLinks = (content: string, sources: SearchResult[]): string => {
  const segments = content.split(/(```[\s\S]*?```)/g)

  return segments.map(segment => {
    if (segment.startsWith('```')) {
      return segment
    }

    return segment.replace(/\[(\d+)\]/g, (match, numberText: string) => {
      const citationNumber = Number.parseInt(numberText, 10)
      const source = sources[citationNumber - 1]

      if (!source?.url) {
        return match
      }

      const title = source.title?.trim()
      const titleFragment = title ? ` "${escapeMarkdownLinkTitle(title)}"` : ''
      return `[[${citationNumber}]](${source.url}${titleFragment})`
    })
  }).join('')
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

    if (isStreaming) {
      return <div className={styles.streamingCitationContent}>{content}</div>
    }

    const markdownContent = (!safeSources || safeSources.length === 0 || !/\[\d+\]/.test(content))
      ? content
      : injectCitationLinks(content, safeSources)

    return <MarkdownRenderer content={markdownContent} />
  }, [content, isStreaming, safeSources])

  return (
    <div className={styles.contentWithCitations} translate={translateAttr}>
      {processedContent}
    </div>
  )
}
