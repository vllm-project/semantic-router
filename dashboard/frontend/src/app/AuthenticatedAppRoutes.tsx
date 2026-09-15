import React from 'react'
import { Navigate, Route } from 'react-router-dom'
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
import EvaluationAvailabilityRoute from './EvaluationAvailabilityRoute'
import { canAccessDashboardPath, type PermissionUser } from '../utils/accessControl'
import {
  loadBuilderPage,
  loadConfigSchemaReferencePage,
  loadDashboardPage,
  loadEvaluationPage,
  loadInsightsPage,
  loadInsightsRecordPage,
  loadKnowledgeMapPage,
  loadLogsPage,
  loadMLSetupPage,
  loadModelHubPage,
  loadMonitoringPage,
  loadOpenClawPage,
  loadPlaygroundFullscreenPage,
  loadPlaygroundPage,
  loadSetupWizardPage,
  loadStatusPage,
  loadTopologyPage,
  loadTracingPage,
  loadUsersPage,
} from './routeLoaders'

interface AuthenticatedAppRoutesProps {
  canUseMLSetup: boolean
  user: PermissionUser | null
  setupMode: boolean
  settingsLoading: boolean
  evaluationAvailable: boolean
  evaluationUnavailableReason: string
}

const shellPageElements: Record<ShellRoutePage, React.ReactElement> = {
  builder: <RecoverableLazyRoute loader={loadBuilderPage} routeLabel="Config Builder" />,
  'config-reference': (
    <RecoverableLazyRoute loader={loadConfigSchemaReferencePage} routeLabel="Schema reference" />
  ),
  dashboard: <RecoverableLazyRoute loader={loadDashboardPage} routeLabel="Dashboard" />,
  evaluation: <RecoverableLazyRoute loader={loadEvaluationPage} routeLabel="Evaluation" />,
  insights: <RecoverableLazyRoute loader={loadInsightsPage} routeLabel="Insights" />,
  'insights-record': (
    <RecoverableLazyRoute loader={loadInsightsRecordPage} routeLabel="Insight record" />
  ),
  logs: <RecoverableLazyRoute loader={loadLogsPage} routeLabel="Logs" />,
  monitoring: <RecoverableLazyRoute loader={loadMonitoringPage} routeLabel="Monitoring" />,
  models: <RecoverableLazyRoute loader={loadModelHubPage} routeLabel="Model Hub" />,
  openclaw: <RecoverableLazyRoute loader={loadOpenClawPage} routeLabel="OpenClaw" />,
  playground: <RecoverableLazyRoute loader={loadPlaygroundPage} routeLabel="Playground" />,
  status: <RecoverableLazyRoute loader={loadStatusPage} routeLabel="Status" />,
  topology: <RecoverableLazyRoute loader={loadTopologyPage} routeLabel="Topology" />,
  tracing: <RecoverableLazyRoute loader={loadTracingPage} routeLabel="Tracing" />,
  users: <RecoverableLazyRoute loader={loadUsersPage} routeLabel="Users" />,
}

const renderShellContent = (
  route: Pick<ShellRouteDefinition, 'hideAccountControl' | 'hideHeaderOnMobile'>,
  element: React.ReactElement,
) => (
  <AppShellLayout
    hideHeaderOnMobile={route.hideHeaderOnMobile}
    hideAccountControl={route.hideAccountControl}
  >
    {element}
  </AppShellLayout>
)

const renderShellElement = (
  route: ShellRouteDefinition,
  settingsLoading: boolean,
  evaluationAvailable: boolean,
  evaluationUnavailableReason: string,
) => {
  const content = renderShellContent(route, shellPageElements[route.page])
  if (route.page !== 'evaluation') return content
  return (
    <EvaluationAvailabilityRoute
      available={evaluationAvailable}
      isLoading={settingsLoading}
      reason={evaluationUnavailableReason}
    >
      {content}
    </EvaluationAvailabilityRoute>
  )
}

export const renderAuthenticatedAppRoutes = ({
  canUseMLSetup,
  user,
  setupMode,
  settingsLoading,
  evaluationAvailable,
  evaluationUnavailableReason,
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
            renderShellElement(
              route,
              settingsLoading,
              evaluationAvailable,
              evaluationUnavailableReason,
            )
          ) : (
            <Navigate to="/dashboard" replace />
          )
        }
      />
    ))}
    <Route path="/config" element={<ConfigSectionRoute />} />
    <Route path="/config/:section" element={<ConfigSectionRoute />} />
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
    <Route path="/knowledge-bases/:view" element={<KnowledgeBaseRoute />} />
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
          )
        ) : (
          <Navigate to="/dashboard" replace />
        )
      }
    />
    <Route path="*" element={<Navigate to={fallbackRouteTarget(setupMode)} replace />} />
  </>
)
