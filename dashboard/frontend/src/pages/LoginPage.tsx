import React, { FormEvent, useEffect, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import ColorBends from '../components/ColorBends'
import styles from './LoginPage.module.css'

interface LocationState {
  from?: string
}

const LoginPage: React.FC = () => {
  const { setupState, isLoading: setupLoading } = useSetup()
  const { isAuthenticated, isLoading, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as LocationState | null)?.from ?? null

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')

  const [error, setError] = useState('')
  const [canBootstrap, setCanBootstrap] = useState(false)
  const [registerMode, setRegisterMode] = useState(false)
  const [pending, setPending] = useState(false)

  const isFirstServe = Boolean(setupState?.setupMode)
  const targetAfterLogin = from ?? (isFirstServe ? '/setup' : '/dashboard')

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch('/api/auth/bootstrap/can-register', { method: 'GET' })
        if (response.ok) {
          const payload = (await response.json()) as { canRegister: boolean }
          setCanBootstrap(Boolean(payload?.canRegister))
        }
      } catch {
        // keep default false
      }
    }
    void load()
  }, [])

  const onSubmitLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    setPending(true)
    try {
      await login(email.trim(), password)
      navigate(targetAfterLogin, { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please check credentials.')
    } finally {
      setPending(false)
    }
  }

  const onSubmitRegister = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    setPending(true)
    try {
      const response = await fetch('/api/auth/bootstrap/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password, name }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `Request failed: ${response.status}`)
      }
      const payload = (await response.json()) as { token: string; user?: { id: string; email: string; name: string; role?: string } }
      localStorage.setItem('vsr_auth_token', payload.token)
      // force full reload to let AuthContext initialize state with new token
      window.location.href = targetAfterLogin
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Register failed.')
    } finally {
      setPending(false)
    }
  }

  if (isAuthenticated && !isLoading && !setupLoading) {
    return <Navigate to={targetAfterLogin} replace />
  }

  return (
    <div className={styles.container}>
      <div className={styles.backgroundEffect}>
        <ColorBends
          colors={['#76b900', '#00b4d8', '#ffffff']}
          rotation={20}
          speed={0.2}
          scale={1}
          frequency={1}
          warpStrength={1}
          mouseInfluence={1}
          parallax={0.5}
          noise={0.08}
          transparent
          autoRotate={0.8}
        />
      </div>

      <main className={styles.mainContent}>
        <form className={styles.card} onSubmit={registerMode ? onSubmitRegister : onSubmitLogin}>
          <div className={styles.heroBadge}>
            <img src="/vllm.png" alt="vLLM Logo" className={styles.badgeLogo} />
            <span>Welcome to vLLM Semantic Router</span>
          </div>

          <h1 className={styles.title}>{registerMode ? 'Create first admin' : 'Sign in'}</h1>
          <p className={styles.subtitle}>
            {registerMode
              ? 'No account exists yet. Register your first admin account.'
              : 'Use your dashboard account to continue.'}
          </p>

          <label className={styles.label}>Email</label>
          <input
            className={styles.input}
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
            required
          />

          <label className={styles.label}>Password</label>
          <input
            className={styles.input}
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="••••••••"
            required
          />

          {registerMode ? (
            <>
              <label className={styles.label}>Name</label>
              <input
                className={styles.input}
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Admin User"
                required
              />
            </>
          ) : null}

          {error ? <div className={styles.error}>{error}</div> : null}

          <button className={styles.button} type="submit" disabled={pending || setupLoading || isLoading}>
            {registerMode ? (pending ? 'Registering...' : 'Create admin') : (isLoading ? 'Signing in...' : 'Continue')}
          </button>

          {canBootstrap && !registerMode ? (
            <button className={styles.secondaryButton} type="button" onClick={() => setRegisterMode(true)}>
              First-time: register admin account
            </button>
          ) : null}

          {registerMode ? (
            <button
              className={styles.secondaryButton}
              type="button"
              onClick={() => setRegisterMode(false)}
            >
              Back to sign in
            </button>
          ) : null}

          <button className={styles.secondaryButton} type="button" onClick={() => navigate('/')}>
            Back to landing page
          </button>
        </form>
      </main>
    </div>
  )
}

export default LoginPage
