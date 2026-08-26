import type { ConfigSection } from '../components/ConfigNav'
import { useAuth } from '../contexts/AuthContext'
import { useSystemStatus } from '../contexts/SystemStatusContext'
import AppShellLayout from './AppShellLayout'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { loadStatusPage } from './routeLoaders'

interface StatusRouteProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
}

/**
 * Status is public. Healthy authenticated sessions retain the product shell;
 * degraded sessions receive the same status surface without exposing routes
 * that depend on Router authority.
 */
export default function StatusRoute({ configSection, setConfigSection }: StatusRouteProps) {
  const { isAuthenticated, user } = useAuth()
  const { routingAccess } = useSystemStatus()
  const page = <RecoverableLazyRoute loader={loadStatusPage} routeLabel="Status" />
  const identityReady = !user?.managementIdentityStatus || user.managementIdentityStatus === 'ready'

  if (!isAuthenticated || !identityReady || routingAccess !== 'operational') return page
  return (
    <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
      {page}
    </AppShellLayout>
  )
}
