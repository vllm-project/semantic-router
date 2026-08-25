import { useState } from 'react'
import ConfirmDialog from '../components/ConfirmDialog'
import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { DashboardMember } from './AccessControlViewTypes'
import styles from './AccessControlPage.module.css'

export function DashboardAccessDialog({
  member,
  onClose,
  onChanged,
}: {
  member: DashboardMember
  onClose: () => void
  onChanged: () => void
}) {
  const [role, setRole] = useState(member.role)
  const [status, setStatus] = useState(member.status)
  const [password, setPassword] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [confirmingRemove, setConfirmingRemove] = useState(false)
  const dialogRef = useAccessibleDialog<HTMLFormElement>({
    isOpen: !confirmingRemove,
    onClose,
    dismissible: !saving,
  })
  const responseError = async (response: Response) =>
    (await response.text()) || `Request failed (${response.status})`
  const save = async () => {
    setSaving(true)
    setError('')
    try {
      const response = await fetch(`/api/admin/users/${member.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, status }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      if (password) {
        const passwordResponse = await fetch('/api/admin/users/password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId: member.id, password }),
        })
        if (!passwordResponse.ok) throw new Error(await responseError(passwordResponse))
      }
      onChanged()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not update Dashboard access')
    } finally {
      setSaving(false)
    }
  }
  const remove = async () => {
    setSaving(true)
    setError('')
    try {
      const response = await fetch(`/api/admin/users/${member.id}`, { method: 'DELETE' })
      if (!response.ok && response.status !== 204) throw new Error(await responseError(response))
      onChanged()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not remove Dashboard access')
      setConfirmingRemove(false)
    } finally {
      setSaving(false)
    }
  }

  if (confirmingRemove) {
    return (
      <ConfirmDialog
        isOpen
        title={`Remove ${member.name}’s login?`}
        description="They will no longer be able to sign in to this Dashboard. Their inference identity, Team memberships, and API keys are not deleted."
        eyebrow="Dashboard access"
        confirmLabel="Remove login"
        pending={saving}
        tone="danger"
        onCancel={() => setConfirmingRemove(false)}
        onConfirm={remove}
      />
    )
  }

  return (
    <div
      className={styles.modalBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !saving) onClose()
      }}
    >
      <form
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dashboard-access-title"
        tabIndex={-1}
        onSubmit={(event) => {
          event.preventDefault()
          void save()
        }}
      >
        <header className={styles.modalHeader}>
          <div className={styles.modalHeading}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span className={styles.modalEyebrow}>Dashboard access</span>
              <h2 id="dashboard-access-title">Manage {member.name}</h2>
              <p>Update their Dashboard role, sign-in status, or password.</p>
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
          <div className={styles.formGrid}>
            <label className={styles.formField}>
              <span>Role</span>
              <select
                value={role}
                onChange={(event) => setRole(event.target.value)}
                data-dialog-initial-focus
              >
                <option value="admin">Admin</option>
                <option value="write">Builder</option>
                <option value="read">Viewer</option>
              </select>
            </label>
            <label className={styles.formField}>
              <span>Status</span>
              <select value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
            </label>
            <label className={`${styles.formField} ${styles.formFieldWide}`}>
              <span>New password (optional)</span>
              <input
                type="password"
                minLength={9}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="9 characters or more"
                autoComplete="new-password"
              />
              <small>Leave blank to keep the current password.</small>
            </label>
          </div>
        </div>
        <footer className={styles.modalFooter}>
          <button
            type="button"
            className={styles.dangerButton}
            onClick={() => setConfirmingRemove(true)}
            disabled={saving}
          >
            <ProductIcon name="trash" /> Remove login
          </button>
          <span className={styles.footerSpacer} />
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={onClose}
            disabled={saving}
          >
            <ProductIcon name="close" />
            Cancel
          </button>
          <button type="submit" className={styles.primaryButton} disabled={saving}>
            {!saving ? <ProductIcon name="check" /> : null}
            {saving ? 'Saving…' : 'Save changes'}
          </button>
        </footer>
      </form>
    </div>
  )
}

export default DashboardAccessDialog
