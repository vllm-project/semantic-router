import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import SetupStatusPage from './SetupStatusPage'

interface EvaluationAvailabilityRouteProps {
  available: boolean
  isLoading: boolean
  reason: string
  children: ReactNode
}

/** Keep a disabled or failed Evaluation service out of the normal workspace. */
export default function EvaluationAvailabilityRoute({
  available,
  isLoading,
  reason,
  children,
}: EvaluationAvailabilityRouteProps) {
  const navigate = useNavigate()

  if (isLoading) {
    return (
      <SetupStatusPage
        title="Checking Evaluation"
        description="Confirming that the Evaluation service initialized successfully."
        actionLabel=""
        onAction={() => undefined}
        variant="loading"
      />
    )
  }

  if (!available) {
    return (
      <SetupStatusPage
        title="Evaluation is not available"
        description={reason || 'Evaluation is not available for this deployment.'}
        actionLabel="Return to Dashboard"
        onAction={() => navigate('/dashboard', { replace: true })}
      />
    )
  }

  return children
}
