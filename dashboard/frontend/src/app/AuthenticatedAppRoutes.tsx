import React from 'react'
import { Navigate, Route } from 'react-router-dom'
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
  loadAccountSecurityPage,
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
  loadSecurityPolicyPage,
  loadSetupWizardPage,
  loadStatusPage,
  loadTopologyPage,
  loadTracingPage,
  loadUsersPage,
} from './routeLoaders'

interface AuthenticatedAppRoutesProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  canUseMLSetup: boolean
  user: PermissionUser | null
  setupMode: boolean
}

const shellPageElements: Record<ShellRoutePage, React.ReactElement> = {
  'account-security': (
    <RecoverableLazyRoute loader={loadAccountSecurityPage} routeLabel="Password & security" />
  ),
  builder: <RecoverableLazyRoute loader={loadBuilderPage} routeLabel="Config Builder" />,
  clawos: <RecoverableLazyRoute loader={loadOpenClawPage} routeLabel="ClawOS" />,
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
  security: <RecoverableLazyRoute loader={loadSecurityPolicyPage} routeLabel="Security" />,
  status: <RecoverableLazyRoute loader={loadStatusPage} routeLabel="Status" />,
  topology: <RecoverableLazyRoute loader={loadTopologyPage} routeLabel="Topology" />,
  tracing: <RecoverableLazyRoute loader={loadTracingPage} routeLabel="Tracing" />,
  users: <RecoverableLazyRoute loader={loadUsersPage} routeLabel="Users" />,
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
          canAccessDashboardPath(user, route.path) ? (
            renderShellElement(route, configSection, setConfigSection)
          ) : (
            <Navigate to="/dashboard" replace />
          )
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
