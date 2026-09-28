import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import SetupStatusPage from './SetupStatusPage'

interface EvaluationAvailabilityRouteProps {
  available: boolean
  isLoading: boolean
  reason: string
  settingsError: string | null
  onRefreshAccess: () => void
  children: ReactNode
  featureName?: string
}

/** Keep a disabled or failed feature out of the normal workspace. */
export default function EvaluationAvailabilityRoute({
  available,
  isLoading,
  reason,
  settingsError,
  onRefreshAccess,
  children,
  featureName = 'Evaluation',
}: EvaluationAvailabilityRouteProps) {
  const navigate = useNavigate()

  if (isLoading) {
    return (
      <SetupStatusPage
        title={`Checking ${featureName}`}
        description={`Confirming that the ${featureName} service initialized successfully.`}
        actionLabel=""
        onAction={() => undefined}
        variant="loading"
      />
    )
  }

  if (settingsError) {
    return (
      <SetupStatusPage
        title="Unable to check Evaluation access"
        description={settingsError}
        actionLabel="Refresh access"
        onAction={onRefreshAccess}
      />
    )
  }

  if (!available) {
    return (
      <SetupStatusPage
        title={`${featureName} is not available`}
        description={reason || `${featureName} is not available for this deployment.`}
        actionLabel="Return to Dashboard"
        onAction={() => navigate('/dashboard', { replace: true })}
      />
    )
  }

  return children
}
