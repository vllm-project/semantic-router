import styles from './ConfigPageModelsSection.module.css'
import type { ModelLiveVerificationState } from './useModelLiveVerification'

interface ConfigPageModelLiveVerificationProps {
  model: string
  hasBackend: boolean
  allowed: boolean
  state: ModelLiveVerificationState
  onVerify: () => void
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
    ? 'Checking'
    : state.status === 'verified'
      ? 'Live'
      : state.status === 'failed'
        ? 'Retry'
        : 'Check'

  return (
    <div className={styles.liveVerification} aria-live="polite">
      <button
        type="button"
        className={styles.liveVerificationButton}
        disabled={!hasBackend || !allowed || pending}
        onClick={onVerify}
        aria-label={`${buttonLabel} ${model} with a live inference query`}
        title={
          state.status === 'failed'
            ? state.message
            : state.status === 'verified'
              ? `Live · ${state.evidence.latencyMs} ms`
              : undefined
        }
      >
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
          {!hasBackend ? 'No backend' : !allowed ? 'Unavailable' : buttonLabel}
        </span>
      </button>
    </div>
  )
}
