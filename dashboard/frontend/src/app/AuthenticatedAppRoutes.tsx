import React, { Suspense, lazy } from 'react'
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
import RouteLoadingFallback from './RouteLoadingFallback'

const BuilderPage = lazy(() => import('../pages/BuilderPage'))
const DashboardPage = lazy(() => import('../pages/DashboardPage'))
const EvaluationPage = lazy(() => import('../pages/EvaluationPage'))
const FleetSimFleetsPage = lazy(() => import('../pages/FleetSimFleetsPage'))
const FleetSimOverviewPage = lazy(() => import('../pages/FleetSimOverviewPage'))
const FleetSimRunsPage = lazy(() => import('../pages/FleetSimRunsPage'))
const FleetSimWorkloadsPage = lazy(() => import('../pages/FleetSimWorkloadsPage'))
const InsightsPage = lazy(() => import('../pages/InsightsPage'))
const InsightsRecordPage = lazy(() => import('../pages/InsightsRecordPage'))
const KnowledgeMapPage = lazy(() => import('../pages/KnowledgeMapPage'))
const LogsPage = lazy(() => import('../pages/LogsPage'))
const MLSetupPage = lazy(() => import('../pages/MLSetupPage'))
const MonitoringPage = lazy(() => import('../pages/MonitoringPage'))
const OpenClawPage = lazy(() => import('../pages/OpenClawPage'))
const PlaygroundFullscreenPage = lazy(() => import('../pages/PlaygroundFullscreenPage'))
const PlaygroundPage = lazy(() => import('../pages/PlaygroundPage'))
const SecurityPolicyPage = lazy(() => import('../pages/SecurityPolicyPage'))
const SetupWizardPage = lazy(() => import('../pages/SetupWizardPage'))
const StatusPage = lazy(() => import('../pages/StatusPage'))
const TopologyPage = lazy(() => import('../pages/TopologyPage'))
const TracingPage = lazy(() => import('../pages/TracingPage'))
const UsersPage = lazy(() => import('../pages/UsersPage'))

interface AuthenticatedAppRoutesProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  canUseMLSetup: boolean
  setupMode: boolean
}

const shellPageElements: Record<ShellRoutePage, React.ReactElement> = {
  builder: <BuilderPage />,
  clawos: <OpenClawPage />,
  dashboard: <DashboardPage />,
  evaluation: <EvaluationPage />,
  'fleet-sim': <FleetSimOverviewPage />,
  'fleet-sim-fleets': <FleetSimFleetsPage />,
  'fleet-sim-runs': <FleetSimRunsPage />,
  'fleet-sim-workloads': <FleetSimWorkloadsPage />,
  insights: <InsightsPage />,
  'insights-record': <InsightsRecordPage />,
  logs: <LogsPage />,
  monitoring: <MonitoringPage />,
  playground: <PlaygroundPage />,
  security: <SecurityPolicyPage />,
  status: <StatusPage />,
  topology: <TopologyPage />,
  tracing: <TracingPage />,
  users: <UsersPage />,
}

const withRouteSuspense = (element: React.ReactElement) => (
  <Suspense fallback={<RouteLoadingFallback />}>
    {element}
  </Suspense>
)

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
    {withRouteSuspense(element)}
  </AppShellLayout>
)

const renderShellElement = (
  route: ShellRouteDefinition,
  configSection: ConfigSection,
  setConfigSection: (section: ConfigSection) => void,
) => renderShellContent(
  route,
  shellPageElements[route.page],
  configSection,
  setConfigSection,
)

export const renderAuthenticatedAppRoutes = ({
  configSection,
  setConfigSection,
  canUseMLSetup,
  setupMode,
}: AuthenticatedAppRoutesProps): React.ReactElement => (
  <>
    <Route path="/setup" element={withRouteSuspense(<SetupWizardPage />)} />
    {shellRouteDefinitions.map((route) => (
      <Route
        key={route.path}
        path={route.path}
        element={renderShellElement(route, configSection, setConfigSection)}
      />
    ))}
    <Route
      path="/config"
      element={(
        <ConfigSectionRoute
          configSection={configSection}
          setConfigSection={setConfigSection}
        />
      )}
    />
    <Route
      path="/config/:section"
      element={(
        <ConfigSectionRoute
          configSection={configSection}
          setConfigSection={setConfigSection}
        />
      )}
    />
    {redirectRouteDefinitions.map((route) => (
      <Route
        key={route.path}
        path={route.path}
        element={<Navigate to={route.to} replace />}
      />
    ))}
    <Route path="/knowledge-bases/:name/map" element={withRouteSuspense(<KnowledgeMapPage />)} />
    <Route
      path="/knowledge-bases/:view"
      element={(
        <KnowledgeBaseRoute
          configSection={configSection}
          setConfigSection={setConfigSection}
        />
      )}
    />
    <Route path="/taxonomy/:view" element={<LegacyTaxonomyRedirect />} />
    <Route
      path="/playground/fullscreen"
      element={withRouteSuspense(<PlaygroundFullscreenPage />)}
    />
    <Route
      path="/ml-setup"
      element={(
        canUseMLSetup ? (
          renderShellContent(
            {},
            <MLSetupPage />,
            configSection,
            setConfigSection,
          )
        ) : (
          <Navigate to="/dashboard" replace />
        )
      )}
    />
    <Route path="*" element={<Navigate to={fallbackRouteTarget(setupMode)} replace />} />
  </>
)
