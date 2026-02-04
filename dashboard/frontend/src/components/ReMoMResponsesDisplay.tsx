import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import styles from './ReMoMResponsesDisplay.module.css'

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

interface ReMoMResponsesDisplayProps {
  rounds: ReMoMRoundResponse[]
}

const ReMoMResponsesDisplay = ({ rounds }: ReMoMResponsesDisplayProps) => {
  // Track expanded state for each response
  const [expandedResponses, setExpandedResponses] = useState<Set<string>>(new Set())

  const toggleResponse = (roundIndex: number, responseIndex: number) => {
    const key = `${roundIndex}-${responseIndex}`
    setExpandedResponses(prev => {
      const newSet = new Set(prev)
      if (newSet.has(key)) {
        newSet.delete(key)
      } else {
        newSet.add(key)
      }
      return newSet
    })
  }

  const isExpanded = (roundIndex: number, responseIndex: number) => {
    return expandedResponses.has(`${roundIndex}-${responseIndex}`)
  }

  if (!rounds || rounds.length === 0) {
    return null
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
        <span>ReMoM Test-Time Scaling</span>
        <span className={styles.badge}>{rounds.length} round{rounds.length > 1 ? 's' : ''}</span>
      </div>

      <div className={styles.rounds}>
        {rounds.map((round, roundIndex) => (
          <div key={roundIndex} className={styles.round}>
            <div className={styles.roundHeader}>
              <span className={styles.roundTitle}>Round {round.round}</span>
              <span className={styles.roundBreadth}>{round.breadth} parallel explore{round.breadth > 1 ? 's' : ''}</span>
            </div>

            <div className={styles.responses}>
              {round.responses.map((response, responseIndex) => {
                const expanded = isExpanded(roundIndex, responseIndex)
                return (
                  <div key={responseIndex} className={styles.responseCard}>
                    <button
                      className={styles.responseHeader}
                      onClick={() => toggleResponse(roundIndex, responseIndex)}
                    >
                      <div className={styles.responseInfo}>
                        <span className={styles.modelName}>{response.model}</span>
                        {response.token_count && (
                          <span className={styles.tokenCount}>{response.token_count} tokens</span>
                        )}
                      </div>
                      <svg
                        className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </button>

                    {expanded && (
                      <div className={styles.responseContent}>
                        {response.compacted_content && (
                          <div className={styles.contentSection}>
                            <div className={styles.contentLabel}>Compacted Content</div>
                            <div className={styles.contentText}>
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeHighlight]}
                              >
                                {response.compacted_content}
                              </ReactMarkdown>
                            </div>
                          </div>
                        )}
                        {response.reasoning && (
                          <div className={styles.contentSection}>
                            <div className={styles.contentLabel}>Reasoning</div>
                            <div className={styles.contentText}>
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeHighlight]}
                              >
                                {response.reasoning}
                              </ReactMarkdown>
                            </div>
                          </div>
                        )}
                        {response.content && !response.compacted_content && (
                          <div className={styles.contentSection}>
                            <div className={styles.contentLabel}>Full Content</div>
                            <div className={styles.contentText}>
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeHighlight]}
                              >
                                {response.content}
                              </ReactMarkdown>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ReMoMResponsesDisplay

