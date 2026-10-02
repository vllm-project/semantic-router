import React from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import ProductLoadingState from '../components/ProductLoadingState'
import SetupStatusPage from './SetupStatusPage'
import styles from './AppStatus.module.css'

/** Requires authentication; redirects to login with return path. */
const AuthGate: React.FC = () => {
  const { isAuthenticated, isLoading, sessionError, refreshSession } = useAuth()
  const location = useLocation()

  if (isLoading && !isAuthenticated) {
    return <ProductLoadingState label="Opening your workspace" />
  }

  if (!isAuthenticated) {
    if (sessionError) {
      return (
        <SetupStatusPage
          title="Unable to verify your session"
          description={sessionError}
          actionLabel="Retry"
          onAction={() => {
            void refreshSession()
          }}
        />
      )
    }
    const from = `${location.pathname}${location.search}${location.hash}`
    return <Navigate to="/login" state={{ from }} replace />
  }

  return (
    <>
      {sessionError && (
        <div className={styles.sessionNotice} role="alert">
          <p>{sessionError}</p>
          <button
            type="button"
            disabled={isLoading}
            onClick={() => {
              void refreshSession()
            }}
          >
            {isLoading ? 'Retrying…' : 'Retry'}
          </button>
        </div>
      )}
      <Outlet />
    </>
  )
}

export default AuthGate
