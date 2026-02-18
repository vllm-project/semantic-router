import React, { useState, useCallback } from 'react'
import styles from './FeedbackButtons.module.css'

const FEEDBACK_API = '/api/router/api/v1/feedback'

export interface FeedbackSubmitParams {
  winnerModel: string
  loserModel?: string
  category?: string
  query?: string
  tie?: boolean
}

interface FeedbackButtonsProps {
  /** Model ID for this response */
  modelId: string
  /** Optional category/decision name */
  category?: string
  /** Optional query for context */
  query?: string
  /** When in A/B mode, other model IDs (thumbs down picks other as winner) */
  otherModelIds?: string[]
  onSuccess?: () => void
  onError?: (message: string) => void
}

const FeedbackButtons: React.FC<FeedbackButtonsProps> = ({
  modelId,
  category,
  query,
  otherModelIds = [],
  onSuccess,
  onError
}) => {
  // Which button is visually selected: 'up', 'down', or null
  const [selection, setSelection] = useState<'up' | 'down' | null>(null)
  // Whether feedback has been submitted to the API (only happens once)
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const clearError = () => setErrorMessage(null)

  const submitFeedback = useCallback(async (params: FeedbackSubmitParams) => {
    const body: Record<string, unknown> = {
      winner_model: params.winnerModel,
      decision_name: params.category || undefined,
      query: params.query || undefined,
      tie: params.tie ?? false
    }
    if (params.loserModel) body.loser_model = params.loserModel

    const res = await fetch(FEEDBACK_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })

    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      const err = data?.error
      const msg = (err && typeof err === 'object' && typeof err.message === 'string')
        ? err.message
        : (typeof data?.message === 'string' ? data.message : res.statusText)
      throw new Error(msg || 'Feedback request failed')
    }
    return data
  }, [])

  const handleClick = useCallback(async (direction: 'up' | 'down') => {
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
      // Thumbs up: winner = this model
      setLoading(true)
      try {
        await submitFeedback({ winnerModel: modelId, category, query })
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
    } else if (direction === 'down' && otherModelIds.length > 0) {
      // Thumbs down in A/B mode: other model wins, this model loses
      setLoading(true)
      try {
        await submitFeedback({
          winnerModel: otherModelIds[0],
          loserModel: modelId,
          category,
          query
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
    // Thumbs down in single-model mode: visual only.
    // API requires winner_model, which we can't provide without a comparison model.
  }, [loading, selection, submitted, submitFeedback, modelId, category, query, otherModelIds, onSuccess, onError])

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
            <span aria-hidden>👍</span>
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
            <span aria-hidden>👎</span>
          )}
        </button>
      </div>
      {submitted && (
        <span className={styles.sentLabel}>Feedback Sent!</span>
      )}
      {errorMessage && (
        <div className={styles.error} role="alert">
          {errorMessage}
          <button type="button" className={styles.dismissError} onClick={clearError} aria-label="Dismiss">×</button>
        </div>
      )}
    </div>
  )
}

export default FeedbackButtons
