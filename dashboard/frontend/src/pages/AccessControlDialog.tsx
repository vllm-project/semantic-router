import { useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import ProductMomentDialog from '../components/ProductMomentDialog'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { copyText } from '../utils/clipboard'
import type { AccessAPIKey, AccessTeam, CreatedAccessAPIKey } from '../utils/inferenceAccessApi'
import AccessControlEditorFields from './AccessControlEditorFields'
import { ACCESS_EDITOR_TITLES } from './AccessControlFormSupport'
import type { AccessEditor } from './AccessControlPageSupport'
import type { AccessControlSelectorSources } from './accessControlSelectorSources'
import styles from './AccessControlPage.module.css'

type EditorProps = {
  editor: AccessEditor
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  selectors: AccessControlSelectorSources
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
  teams?: never
  keys?: never
  selectors?: never
  error?: never
  saving?: never
  onChange?: never
  onSave?: never
  selfService?: never
  selfUserId?: never
}

type Props = EditorProps | SecretProps

export default function AccessControlDialog(props: Props) {
  return 'onViewDetails' in props ? (
    <APIKeySecretDialog {...props} />
  ) : (
    <AccessEditorDialog {...props} />
  )
}

function APIKeySecretDialog({ secret, onClose, onViewDetails }: SecretProps) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  return (
    <ProductMomentDialog
      titleId="api-secret-title"
      eyebrow="API key created"
      title="Your key is ready."
      description="Copy it now and keep it somewhere safe."
      onClose={onClose}
      actions={[
        { label: 'Done', icon: 'check', tone: 'secondary', onClick: onClose },
        {
          label: 'View details',
          icon: 'chevron-right',
          tone: 'primary',
          onClick: onViewDetails,
          initialFocus: true,
        },
      ]}
    >
      <div className={styles.secretValue}>
        <code>{secret.secret}</code>
        <button
          type="button"
          aria-live="polite"
          onClick={() =>
            void copyText(secret.secret).then((success) => {
              setCopyStatus(success ? 'copied' : 'failed')
              window.setTimeout(() => setCopyStatus('idle'), 2200)
            })
          }
        >
          <ProductIcon
            name={copyStatus === 'copied' ? 'check' : copyStatus === 'failed' ? 'refresh' : 'copy'}
          />
          {copyStatus === 'copied' ? 'Copied' : copyStatus === 'failed' ? 'Try again' : 'Copy'}
        </button>
      </div>
      <div className={styles.secretMeta}>
        <span>
          <small>Key</small>
          {secret.name}
        </span>
        <span>
          <small>Owner</small>
          {secret.ownerType === 'team' ? 'Team' : 'Personal'}
        </span>
      </div>
    </ProductMomentDialog>
  )
}

function AccessEditorDialog(props: EditorProps) {
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose: props.onClose,
    dismissible: !props.saving,
  })

  const {
    editor,
    teams,
    selectors,
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
        aria-busy={saving}
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
            <ProductIcon name="close" />
          </button>
        </header>
        <div className={styles.modalBody}>
          {error ? (
            <div className={styles.modalError} role="alert">
              <ProductIcon name="alert" aria-hidden="true" />
              <div>
                <strong>Couldn’t save</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}
          <AccessControlEditorFields
            editor={editor}
            teams={teams}
            selectors={selectors}
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
            {!saving ? <ProductIcon name={editor.value.id ? 'check' : 'plus'} /> : null}
            {saving ? 'Saving…' : editor.value.id ? 'Save changes' : 'Create'}
          </button>
        </footer>
      </section>
    </div>
  )
}
