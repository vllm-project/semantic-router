import styles from './ConfigPageModelsSection.module.css'
import type { ModelLiveVerificationState } from './useModelLiveVerification'

interface ConfigPageModelLiveVerificationProps {
  model: string
  hasBackend: boolean
  allowed: boolean
  state: ModelLiveVerificationState
  onVerify: () => void
}

function formatVerificationTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export default function ConfigPageModelLiveVerification({
  model,
  hasBackend,
  allowed,
  state,
  onVerify,
}: ConfigPageModelLiveVerificationProps) {
  const pending = state.status === 'pending'
  const buttonLabel = pending
    ? 'Verifying…'
    : state.status === 'verified'
      ? 'Verify again'
      : state.status === 'failed'
        ? 'Retry'
        : 'Verify'

  return (
    <div className={styles.liveVerification} aria-live="polite">
      <div className={styles.liveVerificationHeading}>
        <span
          className={`${styles.liveVerificationDot} ${
            state.status === 'verified'
              ? styles.liveVerificationDotSuccess
              : state.status === 'failed'
                ? styles.liveVerificationDotError
                : pending
                  ? styles.liveVerificationDotPending
                  : ''
          }`}
          aria-hidden="true"
        />
        <span className={styles.liveVerificationLabel}>
          {!hasBackend
            ? 'No backend'
            : !allowed
              ? 'Run permission required'
              : state.status === 'verified'
                ? 'Live verified'
                : state.status === 'failed'
                  ? 'Verification failed'
                  : pending
                    ? 'Sending test query'
                    : 'Not live verified'}
        </span>
      </div>

      {state.status === 'verified' ? (
        <div className={styles.liveVerificationEvidence}>
          <span title={state.evidence.summary}>{state.evidence.summary}</span>
          <small title={`${state.evidence.providerModel} via ${state.evidence.backend}`}>
            {state.evidence.provider} · {state.evidence.latencyMs} ms ·{' '}
            {formatVerificationTime(state.evidence.verifiedAt)}
          </small>
        </div>
      ) : null}
      {state.status === 'failed' ? (
        <span className={styles.liveVerificationError} role="alert" title={state.message}>
          {state.message}
        </span>
      ) : null}

      <button
        type="button"
        className={styles.liveVerificationButton}
        disabled={!hasBackend || !allowed || pending}
        onClick={onVerify}
        aria-label={`${buttonLabel} ${model} with a real inference query`}
      >
        {buttonLabel}
      </button>
    </div>
  )
}
