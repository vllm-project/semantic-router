import React, { useEffect, useState } from 'react'
import { Navigate, Outlet, useLocation, useNavigate } from 'react-router-dom'
import OnboardingGuide from '../components/OnboardingGuide'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import InviteCompletionDialog from '../pages/InviteCompletionDialog'
import { canAccessDashboardPath, canSelfManageInferenceAccess } from '../utils/accessControl'
import {
  isFirstAPIKeyOnboardingPending,
  markFirstAPIKeyOnboardingHandled,
} from '../utils/firstAPIKeyOnboarding'
import { inferenceAccessApi } from '../utils/inferenceAccessApi'

/** Setup-mode redirect + onboarding for normal routes. */
const AuthenticatedShell: React.FC = () => {
  const { setupState } = useSetup()
  const { user } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const isSetupMode = setupState?.setupMode ?? false
  const [showFirstKeyPrompt, setShowFirstKeyPrompt] = useState(false)

  useEffect(() => {
    let active = true
    if (!user || !canSelfManageInferenceAccess(user) || !isFirstAPIKeyOnboardingPending(user.id)) {
      setShowFirstKeyPrompt(false)
      return () => {
        active = false
      }
    }

    void inferenceAccessApi
      .selfKeys()
      .then((page) => {
        if (!active) return
        if (page.total > 0) {
          markFirstAPIKeyOnboardingHandled(user.id)
          setShowFirstKeyPrompt(false)
          return
        }
        setShowFirstKeyPrompt(true)
      })
      .catch(() => {
        if (active) setShowFirstKeyPrompt(false)
      })

    return () => {
      active = false
    }
  }, [user])

  const finishFirstKeyPrompt = () => {
    if (user) markFirstAPIKeyOnboardingHandled(user.id)
    setShowFirstKeyPrompt(false)
  }

  if (isSetupMode && location.pathname !== '/setup') {
    return <Navigate to="/setup" replace />
  }

  if (!isSetupMode && location.pathname === '/setup') {
    return <Navigate to="/dashboard" replace />
  }

  return (
    <>
      <Outlet />
      {!isSetupMode &&
        location.pathname !== '/setup' &&
        canAccessDashboardPath(user, '/config/models') &&
        !showFirstKeyPrompt && <OnboardingGuide />}
      {showFirstKeyPrompt && user ? (
        <InviteCompletionDialog
          firstName={user.name.trim().split(/\s+/)[0] || 'there'}
          onCreateKey={() => {
            finishFirstKeyPrompt()
            navigate('/access/api-keys?create=key&from=invitation')
          }}
          onExplore={finishFirstKeyPrompt}
        />
      ) : null}
    </>
  )
}

export default AuthenticatedShell
