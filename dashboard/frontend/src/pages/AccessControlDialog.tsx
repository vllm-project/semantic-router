import { useState } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { copyText } from '../utils/clipboard'
import type {
  AccessAPIKey,
  AccessBudget,
  AccessGroup,
  AccessTeam,
  AccessUser,
  CreatedAccessAPIKey,
} from '../utils/inferenceAccessApi'
import AccessControlEditorFields from './AccessControlEditorFields'
import { ACCESS_EDITOR_TITLES } from './AccessControlFormSupport'
import type { AccessEditor } from './AccessControlPageSupport'
import styles from './AccessControlPage.module.css'

type EditorProps = {
  editor: AccessEditor
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  error: string
  saving: boolean
  onChange: (value: AccessEditor) => void
  onClose: () => void
  onSave: () => void
  selfService?: boolean
  selfUserId?: string
  secret?: never
}

type SecretProps = {
  secret: CreatedAccessAPIKey
  onClose: () => void
  onViewDetails: () => void
  editor?: never
  users?: never
  teams?: never
  keys?: never
  groups?: never
  budgets?: never
  error?: never
  saving?: never
  onChange?: never
  onSave?: never
  selfService?: never
  selfUserId?: never
}

type Props = EditorProps | SecretProps

export default function AccessControlDialog(props: Props) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose: props.onClose,
    dismissible: !('saving' in props && props.saving),
  })

  if (props.secret) {
    return (
      <div
        className={styles.modalBackdrop}
        onMouseDown={(event) => event.target === event.currentTarget && props.onClose()}
      >
        <section
          ref={dialogRef}
          className={`${styles.modal} ${styles.secretModal}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="api-secret-title"
          tabIndex={-1}
        >
          <button
            type="button"
            className={styles.modalClose}
            onClick={props.onClose}
            aria-label="Close"
          >
            ×
          </button>
          <div className={styles.secretIcon} aria-hidden="true">
            <img src="/vllm.png" alt="" />
          </div>
          <span className={styles.modalEyebrow}>API key created</span>
          <h2 id="api-secret-title">Your key is ready.</h2>
          <p>Copy it now and keep it somewhere safe.</p>
          <div className={styles.secretValue}>
            <code>{props.secret.secret}</code>
            <button
              type="button"
              aria-live="polite"
              onClick={() =>
                void copyText(props.secret.secret).then((success) => {
                  setCopyStatus(success ? 'copied' : 'failed')
                  window.setTimeout(() => setCopyStatus('idle'), 2200)
                })
              }
            >
              {copyStatus === 'copied' ? 'Copied' : copyStatus === 'failed' ? 'Try again' : 'Copy'}
            </button>
          </div>
          <div className={styles.secretMeta}>
            <span>
              <small>Key</small>
              {props.secret.name}
            </span>
            <span>
              <small>Owner</small>
              {props.secret.ownerType === 'team' ? 'Team' : 'Personal'}
            </span>
          </div>
          <div className={styles.secretActions}>
            <button type="button" className={styles.secondaryButton} onClick={props.onClose}>
              Done
            </button>
            <button type="button" className={styles.primaryButton} onClick={props.onViewDetails}>
              View details
            </button>
          </div>
        </section>
      </div>
    )
  }

  const {
    editor,
    users,
    teams,
    groups,
    budgets,
    error,
    saving,
    onChange,
    onClose,
    onSave,
    selfService = false,
    selfUserId = '',
  } = props
  const meta = ACCESS_EDITOR_TITLES[editor.kind]
  return (
    <div
      className={styles.modalBackdrop}
      onMouseDown={(event) => event.target === event.currentTarget && !saving && onClose()}
    >
      <section
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="access-dialog-title"
        tabIndex={-1}
      >
        <header className={styles.modalHeader}>
          <div className={styles.modalHeading}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span className={styles.modalEyebrow}>{meta.eyebrow}</span>
              <h2 id="access-dialog-title">{editor.value.id ? meta.edit : meta.create}</h2>
              <p>{meta.description}</p>
            </div>
          </div>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onClose}
            disabled={saving}
            aria-label="Close"
          >
            ×
          </button>
        </header>
        <div className={styles.modalBody}>
          {error ? (
            <div className={styles.modalError} role="alert">
              <span>!</span>
              <div>
                <strong>Couldn’t save</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}
          <AccessControlEditorFields
            editor={editor}
            users={users}
            teams={teams}
            groups={groups}
            budgets={budgets}
            keys={props.keys}
            selfService={selfService}
            selfUserId={selfUserId}
            onChange={onChange}
          />
        </div>
        <footer className={styles.modalFooter}>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button type="button" className={styles.primaryButton} onClick={onSave} disabled={saving}>
            {saving ? 'Saving…' : editor.value.id ? 'Save changes' : 'Create'}
          </button>
        </footer>
      </section>
    </div>
  )
}
