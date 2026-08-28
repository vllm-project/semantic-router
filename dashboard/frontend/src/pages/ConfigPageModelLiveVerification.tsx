import styles from './ConfigPageModelsSection.module.css'
import type { ModelLiveVerificationState } from './useModelLiveVerification'
import ProductIcon from '../components/ProductIcon'

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
    ? 'Checking…'
    : state.status === 'verified'
      ? 'Check again'
      : state.status === 'failed'
        ? 'Check again'
        : 'Check'

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
                ? 'Live'
                : state.status === 'failed'
                  ? 'Unavailable'
                  : pending
                    ? 'Checking'
                    : 'Not checked'}
        </span>
      </div>

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
        <ProductIcon name="refresh" width={13} height={13} />
        {buttonLabel}
      </button>
    </div>
  )
}
