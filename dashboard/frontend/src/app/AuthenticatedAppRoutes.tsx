import React from 'react'
import { Navigate, Route, useLocation } from 'react-router-dom'
import type { ConfigSection } from '../components/ConfigNav'
import AppShellLayout from './AppShellLayout'
import {
  ConfigSectionRoute,
  KnowledgeBaseRoute,
  LegacyTaxonomyRedirect,
} from './ConfigSectionRoutes'
import {
  fallbackRouteTarget,
  redirectRouteDefinitions,
  shellRouteDefinitions,
  type ShellRouteDefinition,
  type ShellRoutePage,
} from './routeManifest'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { canAccessDashboardPath, type PermissionUser } from '../utils/accessControl'
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
  loadKnowledgeMapPage,
  loadLogsPage,
  loadMLSetupPage,
  loadMonitoringPage,
  loadOpenClawPage,
  loadPlaygroundFullscreenPage,
  loadPlaygroundPage,
  loadPluginOperationsPage,
  loadSetupWizardPage,
  loadStatusPage,
  loadTopologyPage,
  loadTracingPage,
} from './routeLoaders'

interface AuthenticatedAppRoutesProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  canUseMLSetup: boolean
  user: PermissionUser | null
  setupMode: boolean
}

const shellPageElements: Record<ShellRoutePage, React.ReactElement> = {
  'access-control': (
    <RecoverableLazyRoute loader={loadAccessControlPage} routeLabel="Access Control" />
  ),
  builder: <RecoverableLazyRoute loader={loadBuilderPage} routeLabel="Config Builder" />,
  clawos: <RecoverableLazyRoute loader={loadOpenClawPage} routeLabel="OpenClaw" />,
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
  logs: <RecoverableLazyRoute loader={loadLogsPage} routeLabel="Logs" />,
  monitoring: <RecoverableLazyRoute loader={loadMonitoringPage} routeLabel="Monitoring" />,
  playground: <RecoverableLazyRoute loader={loadPlaygroundPage} routeLabel="Playground" />,
  plugins: (
    <RecoverableLazyRoute loader={loadPluginOperationsPage} routeLabel="Plugin Operations" />
  ),
  status: <RecoverableLazyRoute loader={loadStatusPage} routeLabel="Status" />,
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

const AuthorizedShellRoute: React.FC<AuthorizedShellRouteProps> = ({
  route,
  configSection,
  setConfigSection,
  user,
}) => {
  const { pathname } = useLocation()
  return canAccessDashboardPath(user, pathname) ? (
    renderShellElement(route, configSection, setConfigSection)
  ) : (
    <Navigate to="/dashboard" replace />
  )
}

export const renderAuthenticatedAppRoutes = ({
  configSection,
  setConfigSection,
  canUseMLSetup,
  user,
  setupMode,
}: AuthenticatedAppRoutesProps): React.ReactElement => (
  <>
    <Route
      path="/setup"
      element={<RecoverableLazyRoute loader={loadSetupWizardPage} routeLabel="Setup" />}
    />
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
      path="/knowledge-bases/:name/map"
      element={
        canAccessDashboardPath(user, '/knowledge-bases/map') ? (
          <RecoverableLazyRoute loader={loadKnowledgeMapPage} routeLabel="Knowledge map" />
        ) : (
          <Navigate to="/dashboard" replace />
        )
      }
    />
    <Route
      path="/knowledge-bases/:view"
      element={
        <KnowledgeBaseRoute configSection={configSection} setConfigSection={setConfigSection} />
      }
    />
    <Route path="/taxonomy/:view" element={<LegacyTaxonomyRedirect />} />
    <Route
      path="/playground/fullscreen"
      element={
        <RecoverableLazyRoute
          loader={loadPlaygroundFullscreenPage}
          routeLabel="Fullscreen playground"
        />
      }
    />
    <Route
      path="/ml-setup"
      element={
        canUseMLSetup ? (
          renderShellContent(
            {},
            <RecoverableLazyRoute loader={loadMLSetupPage} routeLabel="ML setup" />,
            configSection,
            setConfigSection,
          )
        ) : (
          <Navigate to="/dashboard" replace />
        )
      }
    />
    <Route path="*" element={<Navigate to={fallbackRouteTarget(setupMode)} replace />} />
  </>
)
