import React from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import OnboardingGuide from '../components/OnboardingGuide'
import { useSetup } from '../contexts/SetupContext'

/** Setup-mode redirect + onboarding for normal routes. */
const AuthenticatedShell: React.FC = () => {
  const { setupState } = useSetup()
  const location = useLocation()
  const isSetupMode = setupState?.setupMode ?? false

  if (isSetupMode && location.pathname !== '/setup') {
    return <Navigate to="/setup" replace />
  }

  if (!isSetupMode && location.pathname === '/setup') {
    return <Navigate to="/dashboard" replace />
  }

  return (
    <>
      <Outlet />
      {!isSetupMode && location.pathname !== '/setup' && <OnboardingGuide />}
    </>
  )
}

export default AuthenticatedShell
