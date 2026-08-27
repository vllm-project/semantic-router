import { useEffect, useMemo, useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './DashboardInviteDialog.module.css'

type DashboardRole = 'admin' | 'write' | 'read'

interface Invitation {
  id: string
  email: string
  name: string
  role: DashboardRole
  expiresAt: number
}

interface Props {
  isOpen: boolean
  onClose: () => void
}

const roles: Array<{ value: DashboardRole; label: string; description: string }> = [
  { value: 'read', label: 'Read', description: 'Explore configuration and results.' },
  { value: 'write', label: 'Builder', description: 'Create, tune, and deploy routing work.' },
  { value: 'admin', label: 'Admin', description: 'Manage the workspace and its members.' },
]

const responseError = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

async function copyText(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }
  const input = document.createElement('textarea')
  input.value = value
  input.style.position = 'fixed'
  input.style.opacity = '0'
  document.body.appendChild(input)
  input.select()
  const copied = document.execCommand('copy')
  input.remove()
  if (!copied) throw new Error('Copy is not available in this browser.')
}

export default function DashboardInviteDialog({ isOpen, onClose }: Props) {
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [role, setRole] = useState<DashboardRole>('read')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const [created, setCreated] = useState<{ invitation: Invitation; url: string } | null>(null)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen, onClose, dismissible: !pending })

  useEffect(() => {
    if (!isOpen) return
    setEmail('')
    setName('')
    setRole('read')
    setPending(false)
    setError('')
    setCopied(false)
    setCreated(null)
  }, [isOpen])

  const expiry = useMemo(
    () =>
      created
        ? new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
          }).format(new Date(created.invitation.expiresAt * 1000))
        : '',
    [created],
  )

  if (!isOpen) return null

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setPending(true)
    setError('')
    try {
      const response = await fetch('/api/admin/invitations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), name: name.trim(), role }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const payload = (await response.json()) as { invitation: Invitation; token: string }
      const url = new URL(
        `/invite/${encodeURIComponent(payload.token)}`,
        window.location.origin,
      ).toString()
      setCreated({ invitation: payload.invitation, url })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create invitation.')
    } finally {
      setPending(false)
    }
  }

  return (
    <div className={styles.overlay} onMouseDown={pending ? undefined : onClose}>
      <div
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="invite-dialog-title"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.identity}>
            <img src="/vllm.png" alt="" />
            <div>
              <span>{created ? 'Invitation ready' : 'Dashboard access'}</span>
              <h2 id="invite-dialog-title">
                {created ? `Welcome ${created.invitation.name}` : 'Invite user'}
              </h2>
            </div>
          </div>
          <button
            type="button"
            className={styles.close}
            onClick={onClose}
            aria-label="Close invitation"
          >
            ×
          </button>
        </header>

        {created ? (
          <div className={styles.ready}>
            <div className={styles.readyIntro}>
              <ProductIcon name="check" aria-hidden="true" />
              <div>
                <strong>A place in this workspace is reserved.</strong>
                <span>The link works once and expires {expiry}.</span>
              </div>
            </div>
            <div className={styles.linkBlock}>
              <span>One-time invitation URL</span>
              <code>{created.url}</code>
              <button
                type="button"
                onClick={() => {
                  void copyText(created.url)
                    .then(() => setCopied(true))
                    .catch((cause) =>
                      setError(cause instanceof Error ? cause.message : 'Copy failed.'),
                    )
                }}
              >
                <ProductIcon name="copy" aria-hidden="true" />
                {copied ? 'Copied' : 'Copy link'}
              </button>
            </div>
            {error ? (
              <div className={styles.error} role="alert">
                {error}
              </div>
            ) : null}
            <footer className={styles.footer}>
              <button type="button" className={styles.primary} onClick={onClose}>
                Done
              </button>
            </footer>
          </div>
        ) : (
          <form className={styles.form} onSubmit={submit}>
            <div className={styles.intro}>
              <strong>Make it personal.</strong>
              <span>We’ll reserve their name and email. They only choose a password.</span>
            </div>
            {error ? (
              <div className={styles.error} role="alert">
                {error}
              </div>
            ) : null}
            <div className={styles.grid}>
              <label>
                <span>Name</span>
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Ada Lovelace"
                  autoFocus
                  required
                />
              </label>
              <label>
                <span>Email</span>
                <input
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="ada@example.com"
                  required
                />
              </label>
            </div>
            <fieldset className={styles.roles}>
              <legend>Dashboard role</legend>
              {roles.map((option) => (
                <label
                  key={option.value}
                  className={role === option.value ? styles.roleSelected : ''}
                >
                  <input
                    type="radio"
                    name="role"
                    value={option.value}
                    checked={role === option.value}
                    onChange={() => setRole(option.value)}
                  />
                  <span>
                    <strong>{option.label}</strong>
                    <small>{option.description}</small>
                  </span>
                  <ProductIcon name="check" aria-hidden="true" />
                </label>
              ))}
            </fieldset>
            <footer className={styles.footer}>
              <button
                type="button"
                className={styles.secondary}
                onClick={onClose}
                disabled={pending}
              >
                Cancel
              </button>
              <button type="submit" className={styles.primary} disabled={pending}>
                <ProductIcon name="plus" aria-hidden="true" />
                {pending ? 'Creating…' : 'Create invitation'}
              </button>
            </footer>
          </form>
        )}
      </div>
    </div>
  )
}
