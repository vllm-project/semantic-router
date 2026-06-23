import React, { useState, useCallback } from 'react'
import styles from './FeedbackButtons.module.css'

const OUTCOMES_API = '/api/router/v1/router/outcomes'

export interface FeedbackSubmitParams {
  modelId: string
  verdict: 'good_fit' | 'underpowered'
  category?: string
  query?: string
}

interface FeedbackButtonsProps {
  /** Model ID for this response */
  modelId: string
  /** Router Replay record id for the routed response */
  replayId: string
  /** Optional category/decision name */
  category?: string
  /** Optional query for context */
  query?: string
  onSuccess?: () => void
  onError?: (message: string) => void
}

function ThumbUpIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={styles.icon}
    >
      <path d="M9 11.5V20H5.5A1.5 1.5 0 0 1 4 18.5v-5.5A1.5 1.5 0 0 1 5.5 11.5H9Z" />
      <path d="M9 11.5 12.3 4.9A1.8 1.8 0 0 1 15.7 6l-.7 5.5h4.2a1.8 1.8 0 0 1 1.8 2.1l-1 4.9a1.8 1.8 0 0 1-1.8 1.5H9" />
    </svg>
  )
}

function ThumbDownIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={styles.icon}
    >
      <path d="M15 12.5V4H18.5A1.5 1.5 0 0 1 20 5.5V11A1.5 1.5 0 0 1 18.5 12.5H15Z" />
      <path d="M15 12.5 11.7 19.1A1.8 1.8 0 0 1 8.3 18l.7-5.5H4.8A1.8 1.8 0 0 1 3 10.4l1-4.9A1.8 1.8 0 0 1 5.8 4H15" />
    </svg>
  )
}

const FeedbackButtons: React.FC<FeedbackButtonsProps> = ({
  modelId,
  replayId,
  category,
  query,
  onSuccess,
  onError,
}) => {
  // Which button is visually selected: 'up', 'down', or null
  const [selection, setSelection] = useState<'up' | 'down' | null>(null)
  // Whether feedback has been submitted to the API (only happens once)
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const clearError = () => setErrorMessage(null)

  const submitFeedback = useCallback(
    async (params: FeedbackSubmitParams) => {
      const metadata: Record<string, string> = {}
      if (params.category) {
        metadata.decision = params.category
      }
      if (params.query) {
        metadata.query = params.query
      }

      const body = {
        replay_id: replayId,
        source: 'user',
        target: 'model',
        target_ref: params.modelId,
        verdict: params.verdict,
        reason: params.verdict === 'good_fit' ? 'positive_feedback' : 'negative_feedback',
        score: 1,
        metadata,
      }

      const res = await fetch(OUTCOMES_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const err = data?.error
        const msg =
          err && typeof err === 'object' && typeof err.message === 'string'
            ? err.message
            : typeof data?.message === 'string'
              ? data.message
              : res.statusText
        throw new Error(msg || 'Feedback request failed')
      }
      return data
    },
    [replayId],
  )

  const handleClick = useCallback(
    async (direction: 'up' | 'down') => {
      if (loading) return

      // If same button clicked again, do nothing
      if (selection === direction) return

      // Toggle visual selection
      setSelection(direction)
      setErrorMessage(null)

      // Only submit to API once
      if (submitted) return

      // Determine if we can submit for this direction
      if (direction === 'up') {
        setLoading(true)
        try {
          await submitFeedback({ modelId, verdict: 'good_fit', category, query })
          setSubmitted(true)
          onSuccess?.()
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Failed to submit feedback'
          setErrorMessage(message)
          setSelection(null) // Reset on error so user can retry
          onError?.(message)
        } finally {
          setLoading(false)
        }
      } else if (direction === 'down') {
        setLoading(true)
        try {
          await submitFeedback({
            modelId,
            verdict: 'underpowered',
            category,
            query,
          })
          setSubmitted(true)
          onSuccess?.()
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Failed to submit feedback'
          setErrorMessage(message)
          setSelection(null)
          onError?.(message)
        } finally {
          setLoading(false)
        }
      }
    },
    [loading, selection, submitted, submitFeedback, modelId, category, query, onSuccess, onError],
  )

  return (
    <div className={styles.container}>
      <div className={styles.wrapper} role="group" aria-label="Feedback">
        <button
          type="button"
          className={`${styles.btn} ${styles.thumbsUp} ${selection === 'up' ? styles.selected : ''}`}
          onClick={() => handleClick('up')}
          disabled={loading}
          title={selection === 'up' ? 'Good response (selected)' : 'Good response'}
          aria-label={selection === 'up' ? 'Good response (selected)' : 'Good response'}
          aria-pressed={selection === 'up'}
        >
          {loading && selection === 'up' ? (
            <span className={styles.spinner} aria-hidden />
          ) : (
            <ThumbUpIcon />
          )}
        </button>
        <button
          type="button"
          className={`${styles.btn} ${styles.thumbsDown} ${selection === 'down' ? styles.selected : ''}`}
          onClick={() => handleClick('down')}
          disabled={loading}
          title={selection === 'down' ? 'Bad response (selected)' : 'Bad response'}
          aria-label={selection === 'down' ? 'Bad response (selected)' : 'Bad response'}
          aria-pressed={selection === 'down'}
        >
          {loading && selection === 'down' ? (
            <span className={styles.spinner} aria-hidden />
          ) : (
            <ThumbDownIcon />
          )}
        </button>
      </div>
      {submitted && <span className={styles.sentLabel}>Feedback Sent!</span>}
      {errorMessage && (
        <div className={styles.error} role="alert">
          {errorMessage}
          <button
            type="button"
            className={styles.dismissError}
            onClick={clearError}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      )}
    </div>
  )
}

export default FeedbackButtons
