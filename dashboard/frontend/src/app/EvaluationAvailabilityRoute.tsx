import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlaskConical } from 'lucide-react'
import { Button, EmptyState, PageContainer } from '../ui'
import RouteLoadingFallback from './RouteLoadingFallback'
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

/** Explains an unavailable feature inside the shell, so navigation stays available. */
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
    return <RouteLoadingFallback label={`Checking ${featureName} availability`} />
  }

  if (error || !capability) {
    return (
      <PageContainer>
        <EmptyState
          icon={FlaskConical}
          tone="orange"
          title={`Unable to check ${featureName} access`}
          description={error ?? `${featureName} availability was not reported.`}
          actions={<Button onClick={onRefreshAccess}>Refresh access</Button>}
        />
      </PageContainer>
    )
  }

  if (!isCapabilityUsable(capability)) {
    return (
      <PageContainer>
        <EmptyState
          icon={FlaskConical}
          tone="orange"
          title={`${featureName} is not available`}
          description={
            capability.remediation || `${featureName} is not available for this deployment.`
          }
          actions={
            <>
              <Button variant="outline" onClick={onRefreshAccess}>
                Check again
              </Button>
              <Button onClick={() => navigate('/', { replace: true })}>Go to Home</Button>
            </>
          }
        />
      </PageContainer>
    )
  }

  return children
}
