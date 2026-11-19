import React from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useSetup } from '../contexts/SetupContext'

/** Sends every route to the first-run wizard while setup mode is active, and away from it afterwards. */
const AuthenticatedShell: React.FC = () => {
  const { setupState } = useSetup()
  const location = useLocation()
  const isSetupMode = setupState?.setupMode ?? false

  if (isSetupMode && location.pathname !== '/setup') {
    return <Navigate to="/setup" replace />
  }

  if (!isSetupMode && location.pathname === '/setup') {
    return <Navigate to="/" replace />
  }

  return <Outlet />
}

export default AuthenticatedShell
