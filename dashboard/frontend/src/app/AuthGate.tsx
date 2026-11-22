import React from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import ProductLoadingState from '../components/ProductLoadingState'
import SetupStatusPage from './SetupStatusPage'
import { Alert, Button } from '../ui'

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
      {sessionError ? (
        <div className="fixed bottom-4 left-1/2 z-[60] w-[min(560px,calc(100vw-2rem))] -translate-x-1/2 shadow-lg">
          <Alert
            tone="warning"
            title="Your session could not be refreshed"
            actions={
              <Button
                size="sm"
                variant="outline"
                disabled={isLoading}
                onClick={() => {
                  void refreshSession()
                }}
              >
                {isLoading ? 'Retrying…' : 'Retry'}
              </Button>
            }
            className="bg-popover"
          >
            <span role="alert">{sessionError}</span>
          </Alert>
        </div>
      ) : null}
      <Outlet />
    </>
  )
}

export default AuthGate
