import { FormEvent, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Check, Mail, Ticket, Unlink, UserRound } from 'lucide-react'

import AuthBackdrop, { AuthCard, AuthCardHeader, AuthError } from '../components/AuthBackdrop'
import PasswordInput from '../components/PasswordInput'
import { useAuth } from '../contexts/AuthContext'
import type { AuthUser } from '../contexts/authSession'
import { Badge, Button, Field, IconTile, InputGroup, Label, Skeleton, cn } from '../ui'
import { MIN_PASSWORD_LENGTH, formatServerMessage } from './loginSupport'

interface InvitationInfo {
  email?: string
  name?: string
  role: 'admin' | 'write' | 'read'
  kind: 'personal' | 'shared'
  maxUses: number
  remainingUses: number
  expiresAt: number
}

const ROLE_LABELS: Record<InvitationInfo['role'], string> = {
  admin: 'Admin',
  write: 'Builder',
  read: 'Read',
}

const responseError = async (response: Response) =>
  formatServerMessage(await response.text(), `Request failed (${response.status})`)

const invitationValidity = (expiresAt: number) => {
  const remainingHours = Math.max(0, Math.ceil((expiresAt * 1000 - Date.now()) / 3_600_000))
  if (remainingHours >= 48) return `${Math.ceil(remainingHours / 24)} days left`
  if (remainingHours >= 1) return `${remainingHours} hours left`
  return 'Expires soon'
}

const formatExpiry = (expiresAt: number) =>
  new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(expiresAt * 1000))

function InvitationSummary({ invitation }: { invitation: InvitationInfo }) {
  const isShared = invitation.kind === 'shared'
  return (
    <div className="mb-5 flex items-center gap-3 rounded-lg border border-border bg-surface-subtle px-3.5 py-3">
      <IconTile icon={Ticket} tone="blue" size="sm" />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex items-center gap-2 text-[13px] font-medium text-foreground">
          {isShared
            ? `${invitation.remainingUses} of ${invitation.maxUses} seats left`
            : 'Personal invitation'}
          <Badge tone="brand" size="sm">
            {ROLE_LABELS[invitation.role] ?? invitation.role}
          </Badge>
        </span>
        <span className="text-xs text-muted-foreground">
          Valid until{' '}
          <time dateTime={new Date(invitation.expiresAt * 1000).toISOString()}>
            {formatExpiry(invitation.expiresAt)}
          </time>
        </span>
      </div>
      <span className="shrink-0 text-xs font-medium tabular text-subtle-foreground">
        {invitationValidity(invitation.expiresAt)}
      </span>
    </div>
  )
}

export default function InviteAcceptPage() {
  const { token = '' } = useParams()
  const navigate = useNavigate()
  const { setSession } = useAuth()
  const [invitation, setInvitation] = useState<InvitationInfo | null>(null)
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
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
  const passwordReady = password.length >= MIN_PASSWORD_LENGTH

  const accept = async (event: FormEvent) => {
    event.preventDefault()
    if (!invitation) return
    if (isShared && (!name.trim() || !email.trim().includes('@'))) {
      setError('Add your name and email so your administrator knows who joined.')
      return
    }
    if (!passwordReady) {
      setError(`Use at least ${MIN_PASSWORD_LENGTH} characters for your password.`)
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
      navigate('/', { replace: true })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create your account.')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <AuthBackdrop>
        <AuthCard busy>
          <div className="mb-6 flex flex-col items-center">
            <Skeleton className="mb-5 size-12 rounded-2xl" />
            <Skeleton className="h-5 w-48" />
            <Skeleton className="mt-2.5 h-3.5 w-60" />
          </div>
          <Skeleton className="mb-5 h-[62px] rounded-lg" />
          <div className="flex flex-col gap-4">
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
          </div>
          <span className="sr-only" role="status">
            Opening invitation
          </span>
        </AuthCard>
      </AuthBackdrop>
    )
  }

  if (!invitation) {
    return (
      <AuthBackdrop>
        <AuthCard note="Invitation links work once and expire. Ask your administrator for a fresh one.">
          <div className="flex flex-col items-center text-center">
            <IconTile icon={Unlink} tone="slate" size="lg" className="mb-5" />
            <h1 className="text-xl font-semibold tracking-[-0.02em]">
              This invitation can’t be used
            </h1>
            <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground [overflow-wrap:anywhere]">
              {error || 'This invitation is unavailable.'}
            </p>
            <Button asChild variant="outline" size="lg" className="mt-6 w-full">
              <Link to="/login">
                <ArrowLeft />
                Back to sign in
              </Link>
            </Button>
          </div>
        </AuthCard>
      </AuthBackdrop>
    )
  }

  return (
    <AuthBackdrop>
      <AuthCard note="This link works once. After joining, sign in with your email and this password.">
        <AuthCardHeader
          title={isShared ? 'Join Semantic Router' : `You’re in, ${firstName || 'welcome'}.`}
          description={
            isShared
              ? 'Tell us who is joining, then choose a password.'
              : 'Your name and email are reserved. Choose a password to finish.'
          }
        />

        <InvitationSummary invitation={invitation} />

        <form className="flex flex-col gap-4" onSubmit={accept} aria-busy={submitting} noValidate>
          {isShared ? (
            <>
              <Field label="Name" htmlFor="invite-name">
                <InputGroup
                  id="invite-name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  autoComplete="name"
                  placeholder="Your name"
                  leading={<UserRound />}
                  className="h-10"
                  autoFocus
                  required
                />
              </Field>
              <Field label="Email" htmlFor="invite-email">
                <InputGroup
                  id="invite-email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  placeholder="you@example.com"
                  leading={<Mail />}
                  className="h-10"
                  required
                />
              </Field>
            </>
          ) : (
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 rounded-lg border border-border px-3.5 py-3 text-[13px]">
              <dt className="text-subtle-foreground">Name</dt>
              <dd className="truncate font-medium text-foreground">{invitation.name}</dd>
              <dt className="text-subtle-foreground">Email</dt>
              <dd className="truncate text-foreground">{invitation.email}</dd>
            </dl>
          )}

          <div className="grid gap-1.5">
            <div className="flex items-center justify-between gap-3">
              <Label htmlFor="invite-password">Password</Label>
              <span
                className={cn(
                  'flex items-center gap-1 text-xs transition-colors',
                  passwordReady ? 'text-success-foreground' : 'text-subtle-foreground',
                )}
                aria-live="polite"
              >
                {passwordReady ? <Check className="size-3.5" aria-hidden="true" /> : null}
                {passwordReady ? 'Looks good' : `${MIN_PASSWORD_LENGTH} characters minimum`}
              </span>
            </div>
            <PasswordInput
              id="invite-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={MIN_PASSWORD_LENGTH}
              autoComplete="new-password"
              autoFocus={!isShared}
              placeholder={`${MIN_PASSWORD_LENGTH} characters or more`}
              required
            />
          </div>

          {error ? <AuthError>{error}</AuthError> : null}

          <Button type="submit" size="lg" className="mt-1 w-full" loading={submitting}>
            {submitting ? 'Joining the workspace…' : 'Join the workspace'}
            {submitting ? null : <ArrowRight />}
          </Button>
        </form>
      </AuthCard>
    </AuthBackdrop>
  )
}
