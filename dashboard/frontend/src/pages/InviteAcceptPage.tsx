import { FormEvent, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import { useAuth } from '../contexts/AuthContext'
import type { AuthUser } from '../contexts/authSession'
import AuthExperienceShell from './AuthExperienceShell'
import authStyles from './AuthExperienceShell.module.css'
import styles from './InviteAcceptPage.module.css'

interface InvitationInfo {
  email?: string
  name?: string
  role: 'admin' | 'write' | 'read'
  kind: 'personal' | 'shared'
  maxUses: number
  remainingUses: number
  expiresAt: number
}

const responseError = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

const invitationValidity = (expiresAt: number) => {
  const remainingHours = Math.max(0, Math.ceil((expiresAt * 1000 - Date.now()) / 3_600_000))
  if (remainingHours >= 48) return `${Math.ceil(remainingHours / 24)} days left`
  if (remainingHours >= 1) return `${remainingHours} hours left`
  return 'Expires soon'
}

export default function InviteAcceptPage() {
  const { token = '' } = useParams()
  const navigate = useNavigate()
  const { setSession } = useAuth()
  const [invitation, setInvitation] = useState<InvitationInfo | null>(null)
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!token) {
      setError('This invitation link is incomplete.')
      setLoading(false)
      return
    }
    void fetch(`/api/auth/invitations/${encodeURIComponent(token)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return response.json() as Promise<InvitationInfo>
      })
      .then(setInvitation)
      .catch((cause) =>
        setError(cause instanceof Error ? cause.message : 'This invitation is unavailable.'),
      )
      .finally(() => setLoading(false))
  }, [token])

  const isShared = invitation?.kind === 'shared'
  const firstName = useMemo(() => invitation?.name?.trim().split(/\s+/)[0] || '', [invitation])
  const passwordReady = password.length >= 9

  const accept = async (event: FormEvent) => {
    event.preventDefault()
    if (!invitation) return
    if (!passwordReady) {
      setError('Use at least 9 characters for your password.')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const response = await fetch(`/api/auth/invitations/${encodeURIComponent(token)}/accept`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password, email: email.trim(), name: name.trim() }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const payload = (await response.json()) as { token: string; user: AuthUser }
      setSession(payload.token, payload.user)
      navigate('/dashboard', { replace: true })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create your account.')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) return <ProductLoadingState label="Opening invitation" />

  const story = invitation ? (
    <>
      <div className={authStyles.brandBadge}>
        <img src="/vllm.png" alt="vLLM" />
        <span>Semantic Router</span>
      </div>
      <div className={authStyles.storyCopy}>
        <p className={authStyles.storyEyebrow}>Your invitation is here</p>
        <h1 className={`${authStyles.storyTitle} ${styles.inviteStoryTitle}`}>
          {isShared ? 'Your place is ready.' : `You’re in, ${firstName}.`}
        </h1>
        <p className={styles.storySlogan}>Build what one model can’t.</p>
        <p className={authStyles.storyDescription}>Your Mixture-of-Models workspace is ready.</p>
      </div>
      <div className={styles.invitationTicket}>
        <div className={styles.ticketTopline}>
          <span>Invitation</span>
          <strong>{invitationValidity(invitation.expiresAt)}</strong>
        </div>
        <div className={styles.ticketIdentity}>
          <small>{isShared ? 'Available places' : 'Reserved for'}</small>
          <strong>{isShared ? `${invitation.remainingUses} places` : invitation.name}</strong>
          <span>Dashboard · {invitation.role}</span>
        </div>
        <div className={styles.ticketExpiry}>
          <span>Valid until</span>
          <time dateTime={new Date(invitation.expiresAt * 1000).toISOString()}>
            {new Intl.DateTimeFormat('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
              hour: 'numeric',
              minute: '2-digit',
            }).format(new Date(invitation.expiresAt * 1000))}
          </time>
        </div>
      </div>
    </>
  ) : (
    <>
      <div className={authStyles.brandBadge}>
        <img src="/vllm.png" alt="vLLM" />
        <span>Semantic Router</span>
      </div>
      <div className={authStyles.storyCopy}>
        <p className={authStyles.storyEyebrow}>Invitation</p>
        <h1 className={authStyles.storyTitle}>This link is unavailable.</h1>
        <p className={authStyles.storyDescription}>
          Ask your administrator for a fresh invitation.
        </p>
      </div>
      <div className={authStyles.storyIdentity}>
        <span>One-time access</span>
        <small>Dashboard invitation</small>
      </div>
    </>
  )

  return (
    <AuthExperienceShell story={story}>
      {!invitation ? (
        <section className={authStyles.card}>
          <div className={authStyles.stageHeader}>
            <p className={authStyles.stageEyebrow}>Invitation unavailable</p>
            <h2 className={authStyles.stageTitle}>This link can’t be used.</h2>
            <p className={authStyles.stageDescription}>{error}</p>
          </div>
          <Link className={styles.backLink} to="/login">
            <ProductIcon name="arrow-left" aria-hidden="true" />
            Back to sign in
          </Link>
        </section>
      ) : (
        <form
          className={`${authStyles.card} ${styles.invitationForm}`}
          onSubmit={accept}
          aria-busy={submitting}
        >
          <div className={authStyles.stageHeader}>
            <p className={authStyles.stageEyebrow}>Your account</p>
            <h2 className={authStyles.stageTitle}>
              {isShared ? 'Create your account' : 'Choose your password'}
            </h2>
            <p className={authStyles.stageDescription}>
              {isShared ? 'Tell us who is joining.' : 'Your name and email are already reserved.'}
            </p>
          </div>
          {isShared ? (
            <div className={styles.identityInputs}>
              <div className={authStyles.inputBlock}>
                <label className={authStyles.label} htmlFor="invite-name">
                  Name
                </label>
                <input
                  id="invite-name"
                  className={authStyles.input}
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  autoComplete="name"
                  required
                  autoFocus
                  placeholder="Your name"
                />
              </div>
              <div className={authStyles.inputBlock}>
                <label className={authStyles.label} htmlFor="invite-email">
                  Email
                </label>
                <input
                  id="invite-email"
                  className={authStyles.input}
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  required
                  placeholder="you@example.com"
                />
              </div>
            </div>
          ) : (
            <div className={styles.prefilled}>
              <label>
                <span>Name</span>
                <strong>{invitation.name}</strong>
              </label>
              <label>
                <span>Email</span>
                <strong>{invitation.email}</strong>
              </label>
            </div>
          )}
          {error ? (
            <div className={authStyles.error} role="alert">
              {error}
            </div>
          ) : null}
          <div className={authStyles.inputBlock}>
            <label className={authStyles.label} htmlFor="invite-password">
              Password
            </label>
            <div className={styles.passwordInput}>
              <input
                id="invite-password"
                className={authStyles.input}
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={9}
                required
                autoComplete="new-password"
                autoFocus={!isShared}
                placeholder="9 characters or more"
              />
              <button type="button" onClick={() => setShowPassword((value) => !value)}>
                <ProductIcon name={showPassword ? 'eye-off' : 'eye'} aria-hidden="true" />
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
            <small
              className={`${styles.passwordHint} ${passwordReady ? styles.passwordReady : ''}`}
            >
              {passwordReady ? 'Looks good' : '9 characters minimum'}
            </small>
          </div>
          <button className={authStyles.primaryButton} type="submit" disabled={submitting}>
            {submitting ? 'Joining the workspace…' : 'Join the workspace'}
            <ProductIcon name="arrow-right" aria-hidden="true" />
          </button>
        </form>
      )}
    </AuthExperienceShell>
  )
}
