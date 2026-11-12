import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import SetupStatusPage from './SetupStatusPage'
import {
  isCapabilityUsable,
  PROBE_PENDING,
  type DashboardCapability,
} from '../contexts/dashboardCapabilities'

interface EvaluationAvailabilityRouteProps {
  capability: DashboardCapability | undefined
  isLoading: boolean
  error: string | null
  onRefreshAccess: () => void
  children: ReactNode
  featureName?: string
}

/** Keep a disabled or failed feature out of the normal workspace. */
export default function EvaluationAvailabilityRoute({
  capability,
  isLoading,
  error,
  onRefreshAccess,
  children,
  featureName = 'Evaluation',
}: EvaluationAvailabilityRouteProps) {
  const navigate = useNavigate()

  if (isLoading || capability?.reason === PROBE_PENDING) {
    return (
      <SetupStatusPage
        title={`Checking ${featureName}`}
        description={`Confirming that the ${featureName} service is available.`}
        actionLabel=""
        onAction={() => undefined}
        variant="loading"
      />
    )
  }

  if (error || !capability) {
    return (
      <SetupStatusPage
        title={`Unable to check ${featureName} access`}
        description={error ?? `${featureName} availability was not reported.`}
        actionLabel="Refresh access"
        onAction={onRefreshAccess}
      />
    )
  }

  if (!isCapabilityUsable(capability)) {
    return (
      <SetupStatusPage
        title={`${featureName} is not available`}
        description={capability.remediation || `${featureName} is not available for this deployment.`}
        actionLabel="Return to Dashboard"
        onAction={() => navigate('/dashboard', { replace: true })}
      />
    )
  }

  return children
}
