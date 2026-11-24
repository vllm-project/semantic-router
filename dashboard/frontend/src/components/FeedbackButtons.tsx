import React, { useState, useCallback } from 'react'
import { Check, Loader2, ThumbsDown, ThumbsUp, X } from 'lucide-react'

import { cn } from '../ui'

const OUTCOMES_API = '/api/router/api/v1/observability/outcomes'

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
  /** Link to the matching Router Replay record after a successful rating. */
  insightsHref?: string
  /** Optional category/decision name */
  category?: string
  /** Optional query for context */
  query?: string
  onSuccess?: () => void
  onError?: (message: string) => void
}

const feedbackButton =
  'grid size-7 place-items-center rounded-md text-subtle-foreground outline-none transition-[color,background-color,transform] duration-150 hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 active:scale-95 disabled:cursor-default disabled:opacity-80 [&_svg]:size-4'

const FeedbackButtons: React.FC<FeedbackButtonsProps> = ({
  modelId,
  replayId,
  insightsHref,
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
        headers: {
          'Content-Type': 'application/json',
        },
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
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <div className="inline-flex items-center gap-0.5" role="group" aria-label="Feedback">
          <button
            type="button"
            className={cn(
              feedbackButton,
              selection === 'up' &&
                'bg-success-soft text-success-foreground hover:bg-success-soft hover:text-success-foreground',
            )}
            onClick={() => handleClick('up')}
            disabled={loading}
            title={selection === 'up' ? 'Good response (selected)' : 'Good response'}
            aria-label={selection === 'up' ? 'Good response (selected)' : 'Good response'}
            aria-pressed={selection === 'up'}
          >
            {loading && selection === 'up' ? (
              <Loader2 className="animate-spin" aria-hidden="true" />
            ) : (
              <ThumbsUp aria-hidden="true" />
            )}
          </button>
          <button
            type="button"
            className={cn(
              feedbackButton,
              selection === 'down' &&
                'bg-destructive-soft text-destructive-foreground hover:bg-destructive-soft hover:text-destructive-foreground',
            )}
            onClick={() => handleClick('down')}
            disabled={loading}
            title={selection === 'down' ? 'Bad response (selected)' : 'Bad response'}
            aria-label={selection === 'down' ? 'Bad response (selected)' : 'Bad response'}
            aria-pressed={selection === 'down'}
          >
            {loading && selection === 'down' ? (
              <Loader2 className="animate-spin" aria-hidden="true" />
            ) : (
              <ThumbsDown aria-hidden="true" />
            )}
          </button>
        </div>
        {submitted && (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-medium text-success-foreground animate-in fade-in-0 duration-300">
            <Check className="size-3.5" aria-hidden="true" />
            Feedback Sent!
            {insightsHref ? (
              <a
                href={insightsHref}
                className="font-normal text-muted-foreground underline underline-offset-2 hover:text-foreground"
              >
                View request
              </a>
            ) : null}
          </span>
        )}
      </div>
      {errorMessage && (
        <div className="flex items-center gap-1.5 text-xs text-destructive-foreground" role="alert">
          <span className="[overflow-wrap:anywhere]">{errorMessage}</span>
          <button
            type="button"
            className="grid size-5 shrink-0 place-items-center rounded text-destructive-foreground/80 outline-none hover:bg-destructive-soft hover:text-destructive-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
            onClick={clearError}
            aria-label="Dismiss"
          >
            <X className="size-3.5" aria-hidden="true" />
          </button>
        </div>
      )}
    </div>
  )
}

export default FeedbackButtons
