import { FormEvent, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'

import { useAuth } from '../contexts/AuthContext'
import type { AuthUser } from '../contexts/authSession'
import { markFirstAPIKeyOnboardingPending } from '../utils/firstAPIKeyOnboarding'
import AuthExperienceShell from './AuthExperienceShell'
import authStyles from './AuthExperienceShell.module.css'
import styles from './InviteAcceptPage.module.css'

interface InvitationInfo {
  email: string
  name: string
  role: string
  teamId?: string
  teamName?: string
  expiresAt: number
}

const responseError = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

const roleLabel = (role: string) =>
  role === 'admin'
    ? 'Workspace administrator'
    : role === 'write'
      ? 'Workspace builder'
      : 'Workspace viewer'

const invitationValidity = (expiresAt: number) => {
  const remainingHours = Math.max(0, Math.ceil((expiresAt * 1000 - Date.now()) / 3_600_000))
  if (remainingHours >= 48) return `${Math.ceil(remainingHours / 24)} days left`
  if (remainingHours >= 1) return `${remainingHours} hours left`
  return 'Expires soon'
}

export default function InviteAcceptPage() {
  const [params] = useSearchParams()
  const token = params.get('token') || ''
  const navigate = useNavigate()
  const { setSession } = useAuth()
  const [invitation, setInvitation] = useState<InvitationInfo | null>(null)
  const [password, setPassword] = useState('')
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
    void fetch(`/api/auth/invitations/info?token=${encodeURIComponent(token)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return response.json() as Promise<InvitationInfo>
      })
      .then(setInvitation)
      .catch((nextError) =>
        setError(
          nextError instanceof Error ? nextError.message : 'This invitation is unavailable.',
        ),
      )
      .finally(() => setLoading(false))
  }, [token])

  const firstName = useMemo(() => invitation?.name.trim().split(/\s+/)[0] || 'there', [invitation])
  const passwordReady = password.length >= 9

  const accept = async (event: FormEvent) => {
    event.preventDefault()
    if (!passwordReady) {
      setError('Use at least 9 characters for your password.')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const response = await fetch('/api/auth/invitations/accept', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const payload = (await response.json()) as { token: string; user: AuthUser }
      setSession(payload.token, payload.user)
      if (payload.user.permissions?.includes('access.self')) {
        markFirstAPIKeyOnboardingPending(payload.user.id)
      }
      navigate('/dashboard', { replace: true })
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not create your account')
    } finally {
      setSubmitting(false)
    }
  }

  const story = invitation ? (
    <>
      <div className={authStyles.brandBadge}>
        <img src="/vllm.png" alt="vLLM" />
        <span>Semantic Router</span>
      </div>
      <div className={authStyles.storyCopy}>
        <p className={authStyles.storyEyebrow}>Your invitation is here</p>
        <h1 className={`${authStyles.storyTitle} ${styles.inviteStoryTitle}`}>
          You’re in, {firstName}.
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
          <small>Reserved for</small>
          <strong>{invitation.name}</strong>
          <span>
            {roleLabel(invitation.role)}
            {invitation.teamName ? ` · ${invitation.teamName}` : ''}
          </span>
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
        <p className={authStyles.storyEyebrow}>{loading ? 'Opening invitation' : 'Invitation'}</p>
        <h1 className={authStyles.storyTitle}>
          {loading ? 'Just a moment.' : 'This link is unavailable.'}
        </h1>
        <p className={authStyles.storyDescription}>
          {loading
            ? 'Your welcome is almost ready.'
            : 'Ask your administrator for a fresh invitation.'}
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
      {loading ? (
        <section className={authStyles.card}>
          <div className={authStyles.stageHeader}>
            <p className={authStyles.stageEyebrow}>One moment</p>
            <h2 className={authStyles.stageTitle}>Opening your invitation…</h2>
          </div>
        </section>
      ) : !invitation ? (
        <section className={authStyles.card}>
          <div className={authStyles.stageHeader}>
            <p className={authStyles.stageEyebrow}>Invitation unavailable</p>
            <h2 className={authStyles.stageTitle}>This link can’t be used.</h2>
            <p className={authStyles.stageDescription}>{error}</p>
          </div>
          <Link className={styles.backLink} to="/login">
            Back to sign in
          </Link>
        </section>
      ) : (
        <form className={`${authStyles.card} ${styles.invitationForm}`} onSubmit={accept}>
          <div className={authStyles.stageHeader}>
            <p className={authStyles.stageEyebrow}>Your account</p>
            <h2 className={authStyles.stageTitle}>Choose your password</h2>
            <p className={authStyles.stageDescription}>Your name and email are already reserved.</p>
          </div>
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
          {error ? (
            <div className={authStyles.error} role="alert">
              {error}
            </div>
          ) : null}
          <label className={authStyles.inputBlock} htmlFor="invite-password">
            <span className={authStyles.label}>Password</span>
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
                autoFocus
                placeholder="9 characters or more"
              />
              <button type="button" onClick={() => setShowPassword((value) => !value)}>
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
            <small
              className={`${styles.passwordHint} ${passwordReady ? styles.passwordReady : ''}`}
            >
              {passwordReady ? 'Looks good' : '9 characters minimum'}
            </small>
          </label>
          <button className={authStyles.primaryButton} type="submit" disabled={submitting}>
            {submitting ? 'Joining the workspace…' : 'Join the workspace'}
          </button>
        </form>
      )}
    </AuthExperienceShell>
  )
}
