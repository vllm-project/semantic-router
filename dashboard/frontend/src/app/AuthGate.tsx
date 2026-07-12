import React from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import SetupStatusPage from './SetupStatusPage'

/** Requires authentication; redirects to login with return path. */
const AuthGate: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <SetupStatusPage
        title="Authenticating"
        description="Checking session state..."
        actionLabel="Retry"
        variant="loading"
        onAction={() => {
          window.location.reload()
        }}
      />
    )
  }

  if (!isAuthenticated) {
    const from = `${location.pathname}${location.search}${location.hash}`
    return <Navigate to="/login" state={{ from }} replace />
  }

  return <Outlet />
}

export default AuthGate
