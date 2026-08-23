import React from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import InvitationWelcomeDialog from '../components/InvitationWelcomeDialog'
import OnboardingGuide from '../components/OnboardingGuide'
import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import { peekInvitationOnboarding } from '../utils/invitationOnboarding'

/** Capability-scoped onboarding for authenticated routes. */
const AuthenticatedShell: React.FC = () => {
  const { user } = useAuth()
  const navigate = useNavigate()
  const invitationOnboarding = user?.managementUserId
    ? peekInvitationOnboarding(user.managementUserId)
    : null
  return (
    <>
      <Outlet />
      {canAccessDashboardPath(user, '/config/models') && <OnboardingGuide />}
      {invitationOnboarding ? (
        <InvitationWelcomeDialog
          displayName={invitationOnboarding.displayName}
          onRevealKey={() => navigate('/access/api-keys?onboarding=invitation', { replace: true })}
        />
      ) : null}
    </>
  )
}

export default AuthenticatedShell
