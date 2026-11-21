import React, { FormEvent, useEffect, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Mail, UserRound } from 'lucide-react'

import { preloadDashboardRoute } from '../app/routeLoaders'
import AuthBackdrop, { AuthCard, AuthCardHeader, AuthError } from '../components/AuthBackdrop'
import PasswordInput from '../components/PasswordInput'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import { Button, Field, InputGroup, Label, Skeleton } from '../ui'
import {
  MIN_PASSWORD_LENGTH,
  ROUTE_PRELOAD_GRACE_MS,
  formatServerMessage,
  resolvePostAuthTarget,
  settleWithin,
  targetPathname,
} from './loginSupport'

interface LocationState {
  from?: string
}

type BootstrapStatus = 'checking' | 'enabled' | 'disabled'

type BootstrapFormState = {
  name: string
  email: string
  password: string
}

type BootstrapStep = {
  key: 'name' | 'email' | 'password'
  label: string
  title: string
  description: string
}

const BOOTSTRAP_STEPS: BootstrapStep[] = [
  {
    key: 'name',
    label: 'Identity',
    title: 'Name your first administrator.',
    description: 'This name identifies who configures the first routing setup.',
  },
  {
    key: 'email',
    label: 'Access',
    title: 'Choose the administrator email.',
    description: 'This email owns the workspace and manages who else gets access.',
  },
  {
    key: 'password',
    label: 'Security',
    title: 'Secure the workspace.',
    description: 'Create a password, then continue to your workspace.',
  },
]

const ease = [0.16, 1, 0.3, 1] as const

function CheckingCard() {
  return (
    <AuthCard busy>
      <div className="mb-6 flex flex-col items-center">
        <Skeleton className="mb-5 size-12 rounded-2xl" />
        <Skeleton className="h-5 w-52" />
        <Skeleton className="mt-2.5 h-3.5 w-64" />
      </div>
      <div className="flex flex-col gap-4">
        <Skeleton className="h-10" />
        <Skeleton className="h-10" />
        <Skeleton className="mt-1 h-10" />
      </div>
      <span className="sr-only" role="status">
        Opening your workspace
      </span>
    </AuthCard>
  )
}

function BootstrapProgress({ index }: { index: number }) {
  return (
    <div className="mb-4 flex w-44 flex-col items-center gap-2">
      <ol aria-label="Administrator setup progress" className="flex w-full gap-1.5">
        {BOOTSTRAP_STEPS.map((step, stepIndex) => (
          <li
            key={step.key}
            aria-current={stepIndex === index ? 'step' : undefined}
            className="relative h-1 flex-1 overflow-hidden rounded-full bg-muted"
          >
            <span className="sr-only">
              {step.label}
              {stepIndex < index ? ' (complete)' : ''}
            </span>
            <motion.span
              aria-hidden="true"
              className="absolute inset-y-0 left-0 rounded-full bg-primary"
              initial={false}
              animate={{ width: stepIndex <= index ? '100%' : '0%' }}
              transition={{ duration: 0.45, ease }}
            />
          </li>
        ))}
      </ol>
      <span className="text-xs text-subtle-foreground">
        Step {index + 1} of {BOOTSTRAP_STEPS.length} · {BOOTSTRAP_STEPS[index].label}
      </span>
    </div>
  )
}

const LoginPage: React.FC = () => {
  const { isAuthenticated, isLoading, login, setSession } = useAuth()
  const { setupState, refreshSetupState } = useSetup()
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as LocationState | null)?.from ?? null

  const [loginEmail, setLoginEmail] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [bootstrapForm, setBootstrapForm] = useState<BootstrapFormState>({
    name: '',
    email: '',
    password: '',
  })

  const [bootstrapStatus, setBootstrapStatus] = useState<BootstrapStatus>('checking')
  const [bootstrapStepIndex, setBootstrapStepIndex] = useState(0)
  const [error, setError] = useState('')
  const [invalidField, setInvalidField] = useState<'email' | 'password' | null>(null)
  const [pending, setPending] = useState(false)

  const isFirstServe = Boolean(setupState?.setupMode)
  const targetAfterLogin = resolvePostAuthTarget(isFirstServe, from)
  const isBootstrapMode = bootstrapStatus === 'enabled'
  const currentStep = BOOTSTRAP_STEPS[bootstrapStepIndex] ?? BOOTSTRAP_STEPS[0]
  const isLastStep = bootstrapStepIndex === BOOTSTRAP_STEPS.length - 1
  const returnTo = !isFirstServe && targetAfterLogin !== '/' ? targetAfterLogin : null

  const navigateAfterAuth = async () => {
    const nextSetupState = await refreshSetupState()
    const target = resolvePostAuthTarget(nextSetupState?.setupMode ?? isFirstServe, from)
    await settleWithin(preloadDashboardRoute(targetPathname(target)), ROUTE_PRELOAD_GRACE_MS)
    // Keep the form pending until this navigation unmounts it. Clearing pending
    // on success would let the authenticated-session redirect race the navigation.
    navigate(target, { replace: true })
  }

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch('/api/auth/bootstrap/can-register', {
          method: 'GET',
        })
        if (!response.ok) {
          setBootstrapStatus('disabled')
          return
        }
        const payload = (await response.json()) as { canRegister: boolean }
        setBootstrapStatus(payload?.canRegister ? 'enabled' : 'disabled')
      } catch {
        setBootstrapStatus('disabled')
      }
    }

    void load()
  }, [])

  const showError = (message: string, field: 'email' | 'password' | null = null) => {
    setError(message)
    setInvalidField(field)
  }

  const validateBootstrapStep = () => {
    if (currentStep.key === 'name' && !bootstrapForm.name.trim()) {
      showError('Tell us what the workspace should call you before we continue.')
      return false
    }

    if (currentStep.key === 'email' && !bootstrapForm.email.trim().includes('@')) {
      showError('Add the admin email for this workspace.', 'email')
      return false
    }

    if (currentStep.key === 'password' && bootstrapForm.password.length < MIN_PASSWORD_LENGTH) {
      showError(`Use at least ${MIN_PASSWORD_LENGTH} characters for your password.`, 'password')
      return false
    }

    showError('')
    return true
  }

  const onSubmitLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!loginEmail.trim().includes('@')) {
      showError('Enter the email address of your dashboard account.', 'email')
      return
    }
    if (!loginPassword) {
      showError('Enter your password.', 'password')
      return
    }
    showError('')
    setPending(true)
    void preloadDashboardRoute(targetPathname(targetAfterLogin))
    try {
      await login(loginEmail.trim(), loginPassword)
      await navigateAfterAuth()
    } catch (err) {
      const fallback = 'Login failed. Please check credentials.'
      showError(err instanceof Error ? formatServerMessage(err.message, fallback) : fallback)
      setPending(false)
    }
  }

  const onSubmitBootstrap = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!isLastStep) {
      if (validateBootstrapStep()) {
        setBootstrapStepIndex((current) => current + 1)
      }
      return
    }

    if (!validateBootstrapStep()) {
      return
    }

    showError('')
    setPending(true)
    try {
      const response = await fetch('/api/auth/bootstrap/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: bootstrapForm.email.trim(),
          password: bootstrapForm.password,
          name: bootstrapForm.name.trim(),
        }),
      })
      if (!response.ok) {
        const message = await response.text()
        if (response.status === 409) {
          setBootstrapStatus('disabled')
          setLoginEmail(bootstrapForm.email.trim())
          throw new Error('The first admin is already registered. Sign in to continue.')
        }
        throw new Error(formatServerMessage(message, `Request failed: ${response.status}`))
      }
      const payload = (await response.json()) as {
        token: string
        user?: { id: string; email: string; name: string; role?: string }
      }
      setSession(payload.token, payload.user ?? null)
      await navigateAfterAuth()
    } catch (err) {
      showError(err instanceof Error ? err.message : 'Register failed.')
      setPending(false)
    }
  }

  if (isAuthenticated && !isLoading && !pending) {
    return <Navigate to={targetAfterLogin} replace />
  }

  return (
    <AuthBackdrop>
      {bootstrapStatus === 'checking' ? (
        <CheckingCard />
      ) : isBootstrapMode ? (
        <AuthCard note="This account owns the dashboard. Invite more people later from Settings, under Access.">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={currentStep.key}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease }}
            >
              <AuthCardHeader title={currentStep.title} description={currentStep.description}>
                <BootstrapProgress index={bootstrapStepIndex} />
              </AuthCardHeader>
            </motion.div>
          </AnimatePresence>

          <form
            className="flex flex-col gap-4"
            onSubmit={onSubmitBootstrap}
            autoComplete="on"
            noValidate
            aria-busy={pending || undefined}
          >
            {currentStep.key === 'name' ? (
              <Field label="What should we call you?" htmlFor="bootstrap-name">
                <InputGroup
                  id="bootstrap-name"
                  type="text"
                  name="name"
                  autoComplete="name"
                  value={bootstrapForm.name}
                  onChange={(event) =>
                    setBootstrapForm((current) => ({ ...current, name: event.target.value }))
                  }
                  placeholder="Ada Lovelace"
                  leading={<UserRound />}
                  className="h-10"
                  autoFocus
                  required
                />
              </Field>
            ) : null}

            {currentStep.key === 'email' ? (
              <Field label="Admin email" htmlFor="bootstrap-email">
                <InputGroup
                  id="bootstrap-email"
                  type="email"
                  name="email"
                  autoComplete="username"
                  value={bootstrapForm.email}
                  onChange={(event) =>
                    setBootstrapForm((current) => ({ ...current, email: event.target.value }))
                  }
                  placeholder="you@example.com"
                  leading={<Mail />}
                  className="h-10"
                  aria-invalid={invalidField === 'email' || undefined}
                  autoFocus
                  required
                />
              </Field>
            ) : null}

            {currentStep.key === 'password' ? (
              <>
                <div className="grid gap-1.5">
                  <div className="flex items-center justify-between gap-3">
                    <Label htmlFor="bootstrap-password">Password</Label>
                    <span className="text-xs text-subtle-foreground">
                      {MIN_PASSWORD_LENGTH} characters or more
                    </span>
                  </div>
                  <PasswordInput
                    id="bootstrap-password"
                    name="new-password"
                    autoComplete="new-password"
                    minLength={MIN_PASSWORD_LENGTH}
                    value={bootstrapForm.password}
                    onChange={(event) =>
                      setBootstrapForm((current) => ({ ...current, password: event.target.value }))
                    }
                    placeholder={`${MIN_PASSWORD_LENGTH} characters or more`}
                    aria-invalid={invalidField === 'password' || undefined}
                    autoFocus
                    required
                  />
                </div>
                <div className="flex flex-col gap-0.5 rounded-lg border border-border bg-surface-subtle px-3.5 py-3">
                  <span className="text-xs text-subtle-foreground">Ready to launch as</span>
                  <strong className="truncate text-[13px] font-medium text-foreground">
                    {bootstrapForm.name || 'Your first admin'}
                  </strong>
                  <span className="truncate text-xs text-muted-foreground">
                    {bootstrapForm.email || 'you@example.com'}
                  </span>
                </div>
              </>
            ) : null}

            <AnimatePresence initial={false}>
              {error ? <AuthError key="error">{error}</AuthError> : null}
            </AnimatePresence>

            <div className="mt-1 flex items-center gap-2">
              {bootstrapStepIndex > 0 ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="lg"
                  disabled={pending}
                  onClick={() => {
                    showError('')
                    setBootstrapStepIndex((current) => Math.max(0, current - 1))
                  }}
                >
                  <ArrowLeft />
                  Back
                </Button>
              ) : null}
              <Button
                type="submit"
                size="lg"
                className="flex-1"
                loading={pending}
                disabled={pending || isLoading}
              >
                {isLastStep
                  ? pending
                    ? 'Creating administrator…'
                    : 'Create admin and continue'
                  : 'Next'}
                {pending ? null : <ArrowRight />}
              </Button>
            </div>
          </form>
        </AuthCard>
      ) : (
        <AuthCard note="Sessions use an HttpOnly cookie. Applications call the router endpoint, not this dashboard.">
          <AuthCardHeader
            title="Sign in to Semantic Router"
            description="Use your dashboard account to manage routing."
          />

          {returnTo ? (
            <p className="-mt-2 mb-5 text-center text-xs text-subtle-foreground">
              You’ll continue to{' '}
              <span className="font-mono text-[11.5px] text-muted-foreground">{returnTo}</span>
            </p>
          ) : null}

          <form
            className="flex flex-col gap-4"
            onSubmit={onSubmitLogin}
            autoComplete="on"
            noValidate
            aria-busy={pending || undefined}
          >
            <Field label="Email" htmlFor="login-email">
              <InputGroup
                id="login-email"
                type="email"
                name="email"
                autoComplete="username"
                value={loginEmail}
                onChange={(event) => setLoginEmail(event.target.value)}
                placeholder="you@example.com"
                leading={<Mail />}
                className="h-10"
                aria-invalid={invalidField === 'email' || undefined}
                autoFocus
                required
              />
            </Field>

            <div className="grid gap-1.5">
              <div className="flex items-center justify-between gap-3">
                <Label htmlFor="login-password">Password</Label>
                <span className="text-xs text-subtle-foreground">
                  Forgot it? Ask an administrator.
                </span>
              </div>
              <PasswordInput
                id="login-password"
                name="password"
                autoComplete="current-password"
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
                placeholder="••••••••"
                aria-invalid={invalidField === 'password' || undefined}
                required
              />
            </div>

            <AnimatePresence initial={false}>
              {error ? <AuthError key="error">{error}</AuthError> : null}
            </AnimatePresence>

            <Button
              type="submit"
              size="lg"
              className="mt-1 w-full"
              loading={pending}
              disabled={pending || isLoading}
            >
              {pending ? 'Signing in…' : 'Continue'}
              {pending ? null : <ArrowRight />}
            </Button>
          </form>
        </AuthCard>
      )}
    </AuthBackdrop>
  )
}

export default LoginPage
