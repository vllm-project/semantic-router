import React, { useEffect, useState } from 'react'
import { Navigate, Outlet, useLocation, useNavigate } from 'react-router-dom'
import OnboardingGuide from '../components/OnboardingGuide'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import InviteCompletionDialog from '../pages/InviteCompletionDialog'
import { canAccessDashboardPath, canSelfManageInferenceAccess } from '../utils/accessControl'
import {
  ensureFirstAPIKey,
  handoffFirstAPIKey,
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
  const [provisioningFirstKey, setProvisioningFirstKey] = useState(false)
  const [firstKeyError, setFirstKeyError] = useState('')

  useEffect(() => {
    let active = true
    if (!user || !canSelfManageInferenceAccess(user) || !isFirstAPIKeyOnboardingPending(user.id)) {
      setShowFirstKeyPrompt(false)
      setProvisioningFirstKey(false)
      return () => {
        active = false
      }
    }

    void inferenceAccessApi
      .selfKeys()
      .then((page) => {
        if (!active) return
        if (page.items[0]) {
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

  const createFirstKey = async () => {
    if (!user || provisioningFirstKey) return
    setProvisioningFirstKey(true)
    setFirstKeyError('')
    try {
      const teamPage = await inferenceAccessApi.selfTeams()
      const contextTeamId = teamPage.items[0]?.id
      const key = await ensureFirstAPIKey(user.name, {
        list: inferenceAccessApi.selfKeys,
        create: (name) =>
          inferenceAccessApi.createSelfKey(
            name,
            'user',
            user.id,
            contextTeamId,
          ),
      })
      const secret =
        'secret' in key ? key.secret : (await inferenceAccessApi.selfKeySecret(key.id)).secret
      markFirstAPIKeyOnboardingHandled(user.id)
      setShowFirstKeyPrompt(false)
      handoffFirstAPIKey({ ...key, secret })
      navigate(`/access/api-keys?onboarding=ready&created=${encodeURIComponent(key.id)}`, {
        replace: true,
      })
    } catch (error) {
      setFirstKeyError(error instanceof Error ? error.message : 'Could not prepare your API key')
    } finally {
      setProvisioningFirstKey(false)
    }
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
          busy={provisioningFirstKey}
          error={firstKeyError}
          onCreateKey={() => void createFirstKey()}
          onExplore={() => setShowFirstKeyPrompt(false)}
        />
      ) : null}
    </>
  )
}

export default AuthenticatedShell
