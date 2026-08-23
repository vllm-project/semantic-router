import React, { useEffect, useMemo, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  absoluteInvitationURL,
  dashboardMemberInvitationApi,
  type DashboardMemberInvitation,
} from '../utils/dashboardMemberInvitations'
import { copyText } from '../utils/clipboard'
import type { AccessPickerSource } from './AccessAsyncResourcePicker'
import AccessAsyncResourcePicker from './AccessAsyncResourcePicker'
import type { AccessTeam } from '../utils/inferenceAccessApi'
import styles from './AccessControlPage.module.css'

interface Props {
  isOpen: boolean
  roleOptions: readonly string[]
  teamSource: AccessPickerSource<AccessTeam>
  onClose: () => void
  onCreated: () => void
}

export default function DashboardMemberInviteDialog({
  isOpen,
  roleOptions,
  teamSource,
  onClose,
  onCreated,
}: Props) {
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [role, setRole] = useState('read')
  const [teamId, setTeamId] = useState('')
  const [teamRole, setTeamRole] = useState<'member' | 'admin'>('member')
  const [expiresInHours, setExpiresInHours] = useState(168)
  const [sendEmail, setSendEmail] = useState(true)
  const [result, setResult] = useState<DashboardMemberInvitation | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen,
    onClose,
    dismissible: !submitting,
  })

  useEffect(() => {
    if (!isOpen) return
    setResult(null)
    setError('')
    setCopyStatus('idle')
    setEmail('')
    setName('')
    setTeamId('')
    setTeamRole('member')
  }, [isOpen])

  const invitationURL = useMemo(() => (result ? absoluteInvitationURL(result) : ''), [result])
  if (!isOpen) return null

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const invitation = await dashboardMemberInvitationApi.create({
        email: email.trim(),
        name: name.trim(),
        role,
        teamId: teamId || undefined,
        teamRole: teamId ? teamRole : undefined,
        expiresInHours,
        sendEmail,
      })
      setResult(invitation)
      onCreated()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not create invitation')
    } finally {
      setSubmitting(false)
    }
  }

  const copyInvitation = async () => {
    setCopyStatus((await copyText(invitationURL)) ? 'copied' : 'failed')
    window.setTimeout(() => setCopyStatus('idle'), 2200)
  }

  return (
    <div
      className={styles.modalBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !submitting) onClose()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="invite-title"
        tabIndex={-1}
      >
        {result ? (
          <>
            <button
              type="button"
              className={`${styles.modalClose} ${styles.inviteClose}`}
              onClick={onClose}
              aria-label="Close"
            >
              <ProductIcon name="close" />
            </button>
            <div className={styles.inviteResult}>
              <div className={styles.modalLogo} aria-hidden="true">
                <img src="/vllm.png" alt="" />
              </div>
              <span className={styles.modalEyebrow}>Invitation ready</span>
              <h2 id="invite-title">We’re ready for {result.name}.</h2>
              <p>
                {result.deliveryStatus === 'email_sent'
                  ? `A personal sign-up link is on its way to ${result.email}.`
                  : `Share this personal sign-up link with ${result.name}.`}
              </p>
              <div className={styles.inviteLink}>
                <input
                  value={invitationURL}
                  readOnly
                  aria-label="One-time invitation URL"
                  onFocus={(event) => event.currentTarget.select()}
                />
                <button type="button" onClick={() => void copyInvitation()}>
                  <ProductIcon name={copyStatus === 'copied' ? 'check' : 'copy'} />
                  {copyStatus === 'copied'
                    ? 'Copied'
                    : copyStatus === 'failed'
                      ? 'Select link'
                      : 'Copy link'}
                </button>
              </div>
              {result.deliveryError ? (
                <small className={styles.inviteDeliveryNote}>
                  Email is not configured, so the link is ready to share manually.
                </small>
              ) : null}
              <button type="button" className={styles.primaryButton} onClick={onClose}>
                <ProductIcon name="check" />
                Done
              </button>
            </div>
          </>
        ) : (
          <form onSubmit={submit}>
            <header className={styles.modalHeader}>
              <div className={styles.modalHeading}>
                <div className={styles.modalLogo} aria-hidden="true">
                  <img src="/vllm.png" alt="" />
                </div>
                <div>
                  <span className={styles.modalEyebrow}>Personal invitation</span>
                  <h2 id="invite-title">Invite a user</h2>
                  <p>
                    They’ll see a welcome made for them, choose a password, and enter the Dashboard.
                  </p>
                </div>
              </div>
              <button
                type="button"
                className={styles.modalClose}
                onClick={onClose}
                disabled={submitting}
                aria-label="Close"
              >
                <ProductIcon name="close" />
              </button>
            </header>
            <div className={styles.modalBody}>
              {error ? (
                <div className={styles.modalError} role="alert">
                  <span>!</span>
                  <div>
                    <strong>Couldn’t create invitation</strong>
                    <p>{error}</p>
                  </div>
                </div>
              ) : null}
              <div className={styles.formGrid}>
                <label className={styles.formField}>
                  <span>Name</span>
                  <input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="Ada Lovelace"
                    required
                    data-dialog-initial-focus
                  />
                </label>
                <label className={styles.formField}>
                  <span>Email</span>
                  <input
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    placeholder="ada@company.com"
                    required
                  />
                </label>
                <label className={styles.formField}>
                  <span>Dashboard role</span>
                  <select value={role} onChange={(event) => setRole(event.target.value)}>
                    {roleOptions.map((option) => (
                      <option key={option} value={option}>
                        {option === 'admin'
                          ? 'Admin · full control'
                          : option === 'write'
                            ? 'Builder · create and operate'
                            : 'Viewer · read only'}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={styles.formField}>
                  <span>Link expires</span>
                  <select
                    value={expiresInHours}
                    onChange={(event) => setExpiresInHours(Number(event.target.value))}
                  >
                    <option value={24}>In 24 hours</option>
                    <option value={168}>In 7 days</option>
                    <option value={336}>In 14 days</option>
                    <option value={720}>In 30 days</option>
                  </select>
                </label>
                <div className={styles.formField}>
                  <span>
                    Team <small>Optional</small>
                  </span>
                  <AccessAsyncResourcePicker
                    ariaLabel="Search invitation Team"
                    source={teamSource}
                    selectedIds={teamId ? [teamId] : []}
                    optional
                    optionalTitle="No Team"
                    optionalDescription="Assign one later"
                    placeholder="Search Team name"
                    emptyText="No Teams found"
                    onChange={(selectedIds) => setTeamId(selectedIds[0] || '')}
                  />
                </div>
                {teamId ? (
                  <label className={styles.formField}>
                    <span>Team role</span>
                    <select
                      value={teamRole}
                      onChange={(event) => setTeamRole(event.target.value as 'member' | 'admin')}
                    >
                      <option value="member">Member</option>
                      <option value="admin">Admin</option>
                    </select>
                  </label>
                ) : null}
                <label className={styles.toggleField}>
                  <input
                    type="checkbox"
                    checked={sendEmail}
                    onChange={(event) => setSendEmail(event.target.checked)}
                  />
                  <span>
                    <i />
                    <strong>Send by email</strong>
                    <small>A copyable link is always created.</small>
                  </span>
                </label>
              </div>
            </div>
            <footer className={styles.modalFooter}>
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={onClose}
                disabled={submitting}
              >
                <ProductIcon name="close" />
                Cancel
              </button>
              <button type="submit" className={styles.primaryButton} disabled={submitting}>
                <ProductIcon name="inbox" />
                {submitting ? 'Preparing…' : `Invite ${name.trim().split(/\s+/)[0] || 'user'}`}
              </button>
            </footer>
          </form>
        )}
      </section>
    </div>
  )
}
