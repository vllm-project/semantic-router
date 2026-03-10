import React, { FormEvent, useMemo, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
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
  const [error, setError] = useState('')

  const isFirstServe = Boolean(setupState?.setupMode)
  const targetAfterLogin = useMemo(() => {
    if (from) return from
    return isFirstServe ? '/setup' : '/dashboard'
  }, [from, isFirstServe])

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    try {
      await login(email.trim(), password)
      navigate(targetAfterLogin, { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please check credentials.')
    }
  }

  if (isAuthenticated && !isLoading && !setupLoading) {
    return <Navigate to={targetAfterLogin} replace />
  }

  return (
    <div className={styles.page}>
      <form className={styles.card} onSubmit={onSubmit}>
        <h1>Sign in</h1>
        <p>Use your dashboard account to continue.</p>
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

        {error ? <div className={styles.error}>{error}</div> : null}

        <button className={styles.button} type="submit" disabled={isLoading || setupLoading}>
          {isLoading || setupLoading ? 'Signing in...' : 'Continue'}
        </button>
      </form>
    </div>
  )
}

export default LoginPage
