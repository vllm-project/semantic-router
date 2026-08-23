import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { AgentApprovalRequestPayload } from '../generated/managementApiContract'
import AgentInlineError from './AgentInlineError'
import ProductIcon from './ProductIcon'
import styles from './AgentPlayground.module.css'

interface AgentPublicationReviewDialogProps {
  approval: AgentApprovalRequestPayload
  canPublish: boolean
  error?: string | null
  publishing: boolean
  onClose: () => void
  onPublish: () => void
}

function itemCount(value: unknown): number | null {
  if (Array.isArray(value)) return value.length
  if (value && typeof value === 'object') return Object.keys(value).length
  return null
}

export default function AgentPublicationReviewDialog({
  approval,
  canPublish,
  error,
  publishing,
  onClose,
  onPublish,
}: AgentPublicationReviewDialogProps) {
  const titleId = `publication-review-${approval.planId}`
  const dialogRef = useAccessibleDialog<HTMLElement>({
    isOpen: true,
    onClose,
    dismissible: !publishing,
  })
  const expired = Date.parse(approval.expiresAt) <= Date.now()
  const recipe = approval.summary.recipeName || approval.summary.recipeId || 'Recipe'
  const entrypoint =
    approval.summary.entrypointName || approval.summary.entrypointId || 'Mixture-of-Models'
  const assignments = itemCount(approval.summary.assignments)
  const gates = itemCount(approval.summary.gateResults)
  const changed = approval.summary.changedResources ?? []
  const warnings = approval.summary.warnings ?? []

  return (
    <div
      className={styles.dialogBackdrop}
      onMouseDown={(event) => {
        if (!publishing && event.target === event.currentTarget) onClose()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.reviewDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <header className={styles.reviewDialogHeader}>
          <div className={styles.dialogLogo}>
            <img src="/vllm.png" alt="" />
          </div>
          <div>
            <span>Final review</span>
            <h2 id={titleId}>Publish {entrypoint}</h2>
            <p>Review, then publish.</p>
          </div>
          <button type="button" onClick={onClose} disabled={publishing} aria-label="Close review">
            <ProductIcon name="close" />
          </button>
        </header>

        <div className={styles.reviewIdentity}>
          <div>
            <span>Recipe</span>
            <strong>{recipe}</strong>
          </div>
          <div>
            <span>Mixture-of-Models</span>
            <strong>{entrypoint}</strong>
          </div>
          <div>
            <span>Plan</span>
            <strong title={approval.planDigest}>{approval.planDigest.slice(0, 12)}</strong>
          </div>
        </div>

        <div className={styles.reviewChecks}>
          <div>
            <ProductIcon name="topology" />
            <span>
              <strong>Topology</strong>
              <small>{approval.summary.topology ? 'Compiled' : 'Validated'}</small>
            </span>
          </div>
          <div>
            <ProductIcon name="model" />
            <span>
              <strong>Assignments</strong>
              <small>{assignments === null ? 'Verified' : `${assignments} verified`}</small>
            </span>
          </div>
          <div>
            <ProductIcon name="evaluation" />
            <span>
              <strong>Evaluation</strong>
              <small>{gates === null ? 'Passed' : `${gates} gates passed`}</small>
            </span>
          </div>
        </div>

        {changed.length > 0 ? (
          <section className={styles.reviewSection}>
            <h3>Changes</h3>
            <ul>
              {changed.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>
        ) : null}
        {warnings.length > 0 ? (
          <section className={`${styles.reviewSection} ${styles.reviewWarnings}`}>
            <h3>Before you publish</h3>
            <ul>
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </section>
        ) : null}

        {!canPublish ? (
          <p className={styles.reviewPermission}>
            An operator with publish access must confirm this plan.
          </p>
        ) : null}
        {expired ? (
          <p className={styles.reviewPermission}>
            This plan expired. Ask the Agent to prepare a fresh review.
          </p>
        ) : null}
        {error ? <AgentInlineError message={error} /> : null}

        <footer className={styles.reviewActions}>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={onClose}
            disabled={publishing}
          >
            Not yet
          </button>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={onPublish}
            disabled={!canPublish || expired || publishing}
            data-testid="agent-publish-confirm"
            data-dialog-initial-focus=""
          >
            {publishing ? 'Publishing…' : 'Publish'}
            {!publishing ? <ProductIcon name="arrow-right" /> : null}
          </button>
        </footer>
      </section>
    </div>
  )
}
