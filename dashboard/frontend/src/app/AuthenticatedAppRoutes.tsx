import React from 'react'
import { Navigate, Route, useLocation } from 'react-router-dom'
import type { ConfigSection } from '../components/ConfigNav'
import AppShellLayout from './AppShellLayout'
import { ConfigSectionRoute } from './ConfigSectionRoutes'
import {
  redirectRouteDefinitions,
  shellRouteDefinitions,
  type ShellRouteDefinition,
  type ShellRoutePage,
} from './routeManifest'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { canAccessDashboardPath, type PermissionUser } from '../utils/accessControl'
import { useSystemStatus } from '../contexts/SystemStatusContext'
import {
  loadAccessControlPage,
  loadBuilderPage,
  loadDashboardPage,
  loadEvaluationPage,
  loadFleetSimFleetsPage,
  loadFleetSimOverviewPage,
  loadFleetSimRunsPage,
  loadFleetSimWorkloadsPage,
  loadInsightsPage,
  loadInsightsRecordPage,
  loadMLSetupPage,
  loadMonitoringPage,
  loadOpenClawPage,
  loadPlaygroundFullscreenPage,
  loadPlaygroundPage,
  loadTopologyPage,
  loadTracingPage,
} from './routeLoaders'

interface AuthenticatedAppRoutesProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  canUseMLSetup: boolean
  user: PermissionUser | null
}

const shellPageElements: Record<ShellRoutePage, React.ReactElement> = {
  'access-control': (
    <RecoverableLazyRoute loader={loadAccessControlPage} routeLabel="Access Control" />
  ),
  builder: <RecoverableLazyRoute loader={loadBuilderPage} routeLabel="Config Builder" />,
  dashboard: <RecoverableLazyRoute loader={loadDashboardPage} routeLabel="Dashboard" />,
  evaluation: <RecoverableLazyRoute loader={loadEvaluationPage} routeLabel="Evaluation" />,
  'fleet-sim': <RecoverableLazyRoute loader={loadFleetSimOverviewPage} routeLabel="Fleet Sim" />,
  'fleet-sim-fleets': <RecoverableLazyRoute loader={loadFleetSimFleetsPage} routeLabel="Fleets" />,
  'fleet-sim-runs': (
    <RecoverableLazyRoute loader={loadFleetSimRunsPage} routeLabel="Simulation runs" />
  ),
  'fleet-sim-workloads': (
    <RecoverableLazyRoute loader={loadFleetSimWorkloadsPage} routeLabel="Workloads" />
  ),
  insights: <RecoverableLazyRoute loader={loadInsightsPage} routeLabel="Insights" />,
  'insights-record': (
    <RecoverableLazyRoute loader={loadInsightsRecordPage} routeLabel="Insight record" />
  ),
  monitoring: <RecoverableLazyRoute loader={loadMonitoringPage} routeLabel="Monitoring" />,
  openclaw: <RecoverableLazyRoute loader={loadOpenClawPage} routeLabel="OpenClaw" />,
  playground: <RecoverableLazyRoute loader={loadPlaygroundPage} routeLabel="Playground" />,
  topology: <RecoverableLazyRoute loader={loadTopologyPage} routeLabel="Topology" />,
  tracing: <RecoverableLazyRoute loader={loadTracingPage} routeLabel="Tracing" />,
}

const renderShellContent = (
  route: Pick<ShellRouteDefinition, 'hideAccountControl' | 'hideHeaderOnMobile'>,
  element: React.ReactElement,
  configSection: ConfigSection,
  setConfigSection: (section: ConfigSection) => void,
) => (
  <AppShellLayout
    configSection={configSection}
    setConfigSection={setConfigSection}
    hideHeaderOnMobile={route.hideHeaderOnMobile}
    hideAccountControl={route.hideAccountControl}
  >
    {element}
  </AppShellLayout>
)

const renderShellElement = (
  route: ShellRouteDefinition,
  configSection: ConfigSection,
  setConfigSection: (section: ConfigSection) => void,
) => renderShellContent(route, shellPageElements[route.page], configSection, setConfigSection)

interface AuthorizedShellRouteProps {
  route: ShellRouteDefinition
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  user: PermissionUser | null
}

const AuthorizedFullscreenRoute: React.FC<{
  children: React.ReactElement
  user: PermissionUser | null
}> = ({ children, user }) => {
  const { pathname } = useLocation()
  const { routingAccess } = useSystemStatus()
  return routingAccess === 'operational' && canAccessDashboardPath(user, pathname) ? (
    children
  ) : (
    <Navigate to="/dashboard" replace />
  )
}

const AuthorizedShellRoute: React.FC<AuthorizedShellRouteProps> = ({
  route,
  configSection,
  setConfigSection,
  user,
}) => {
  const { pathname } = useLocation()
  const { routingAccess } = useSystemStatus()
  const routingReady = route.page === 'dashboard' || routingAccess === 'operational'
  return routingReady && canAccessDashboardPath(user, pathname) ? (
    renderShellElement(route, configSection, setConfigSection)
  ) : (
    <Navigate to="/dashboard" replace />
  )
}

const AuthorizedMLSetupRoute: React.FC<{
  canUseMLSetup: boolean
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  user: PermissionUser | null
}> = ({ canUseMLSetup, configSection, setConfigSection, user }) => {
  const { routingAccess } = useSystemStatus()
  if (
    routingAccess !== 'operational' ||
    !canUseMLSetup ||
    !canAccessDashboardPath(user, '/ml-setup')
  ) {
    return <Navigate to="/dashboard" replace />
  }
  return renderShellContent(
    {},
    <RecoverableLazyRoute loader={loadMLSetupPage} routeLabel="ML setup" />,
    configSection,
    setConfigSection,
  )
}

export const renderAuthenticatedAppRoutes = ({
  configSection,
  setConfigSection,
  canUseMLSetup,
  user,
}: AuthenticatedAppRoutesProps): React.ReactElement => (
  <>
    {shellRouteDefinitions.map((route) => (
      <Route
        key={route.path}
        path={route.path}
        element={
          <AuthorizedShellRoute
            route={route}
            configSection={configSection}
            setConfigSection={setConfigSection}
            user={user}
          />
        }
      />
    ))}
    <Route
      path="/config"
      element={
        <ConfigSectionRoute configSection={configSection} setConfigSection={setConfigSection} />
      }
    />
    <Route
      path="/config/:section"
      element={
        <ConfigSectionRoute configSection={configSection} setConfigSection={setConfigSection} />
      }
    />
    {redirectRouteDefinitions.map((route) => (
      <Route key={route.path} path={route.path} element={<Navigate to={route.to} replace />} />
    ))}
    <Route
      path="/playground/fullscreen"
      element={
        <AuthorizedFullscreenRoute user={user}>
          <RecoverableLazyRoute
            loader={loadPlaygroundFullscreenPage}
            routeLabel="Fullscreen playground"
          />
        </AuthorizedFullscreenRoute>
      }
    />
    <Route
      path="/ml-setup"
      element={
        <AuthorizedMLSetupRoute
          canUseMLSetup={canUseMLSetup}
          configSection={configSection}
          setConfigSection={setConfigSection}
          user={user}
        />
      }
    />
    <Route path="*" element={<Navigate to="/dashboard" replace />} />
  </>
)
