import React, { useEffect, useMemo, useState } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  absoluteInvitationURL,
  dashboardMemberInvitationApi,
  type DashboardMemberInvitation,
} from '../utils/dashboardMemberInvitations'
import type { AccessTeam } from '../utils/inferenceAccessApi'
import { copyText } from '../utils/clipboard'
import styles from './AccessControlPage.module.css'

interface Props {
  isOpen: boolean
  roleOptions: readonly string[]
  teams: AccessTeam[]
  onClose: () => void
  onCreated: () => void
}

export default function DashboardMemberInviteDialog({
  isOpen,
  roleOptions,
  teams,
  onClose,
  onCreated,
}: Props) {
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [role, setRole] = useState('read')
  const [teamId, setTeamId] = useState('')
  const [teamRole, setTeamRole] = useState<'admin' | 'member'>('member')
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
              ×
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
                ×
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
                <label className={`${styles.formField} ${styles.formFieldWide}`}>
                  <span>Team (optional)</span>
                  <select
                    value={teamId}
                    onChange={(event) => setTeamId(event.target.value)}
                    data-dialog-initial-focus
                  >
                    <option value="">No team yet</option>
                    {teams.map((team) => (
                      <option key={team.id} value={team.id}>
                        {team.name}
                      </option>
                    ))}
                  </select>
                  <small>Team grants and quota apply to Playground and API usage.</small>
                </label>
                {teamId ? (
                  <fieldset className={`${styles.ownerSection} ${styles.formFieldWide}`}>
                    <legend>
                      Team role <small>Controls this Team only</small>
                    </legend>
                    <div className={styles.ownerChoices} role="radiogroup" aria-label="Team role">
                      <button
                        type="button"
                        role="radio"
                        aria-checked={teamRole === 'member'}
                        className={`${styles.ownerChoice} ${teamRole === 'member' ? styles.ownerChoiceActive : ''}`}
                        onClick={() => setTeamRole('member')}
                      >
                        <span>Member</span>
                        <small>Use Team models and quota</small>
                        <i>{teamRole === 'member' ? '✓' : ''}</i>
                      </button>
                      <button
                        type="button"
                        role="radio"
                        aria-checked={teamRole === 'admin'}
                        className={`${styles.ownerChoice} ${teamRole === 'admin' ? styles.ownerChoiceActive : ''}`}
                        onClick={() => setTeamRole('admin')}
                      >
                        <span>Team admin</span>
                        <small>Manage members and Team keys</small>
                        <i>{teamRole === 'admin' ? '✓' : ''}</i>
                      </button>
                    </div>
                  </fieldset>
                ) : null}
                <label className={styles.formField}>
                  <span>Name</span>
                  <input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="Ada Lovelace"
                    required
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
                Cancel
              </button>
              <button type="submit" className={styles.primaryButton} disabled={submitting}>
                {submitting ? 'Preparing…' : `Invite ${name.trim().split(/\s+/)[0] || 'user'}`}
              </button>
            </footer>
          </form>
        )}
      </section>
    </div>
  )
}
