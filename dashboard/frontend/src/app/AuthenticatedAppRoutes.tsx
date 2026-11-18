import React from 'react'
import { Navigate, Outlet, Route, useLocation, useParams } from 'react-router-dom'
import AppShellLayout from './AppShellLayout'
import {
  fallbackRouteTarget,
  legacyConfigSectionTarget,
  legacyRedirects,
  shellRouteDefinitions,
  type ShellRoutePage,
} from './routeManifest'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import EvaluationAvailabilityRoute from './EvaluationAvailabilityRoute'
import SettingsLayout from '../pages/SettingsLayout'
import NotFoundPage from '../pages/NotFoundPage'
import type { DashboardCapabilities } from '../contexts/dashboardCapabilities'
import type { ConfigSection } from '../components/ConfigNav'
import { canAccessDashboardPath, type PermissionUser } from '../utils/accessControl'
import {
  loadAgentRunPage,
  loadConfigPage,
  loadConfigSchemaReferencePage,
  loadDashboardPage,
  loadDecisionModelsWorkspace,
  loadEditorPage,
  loadEvaluationPage,
  loadGetStartedPage,
  loadInsightsPage,
  loadInsightsRecordPage,
  loadLogsPage,
  loadModelHubPage,
  loadMonitoringPage,
  loadPlaygroundPage,
  loadSetupWizardPage,
  loadStatusPage,
  loadTopologyPage,
  loadTracingPage,
  loadUsersPage,
} from './routeLoaders'

interface AuthenticatedAppRoutesProps {
  user: PermissionUser | null
  setupMode: boolean
  capabilities: DashboardCapabilities | null
  capabilitiesLoading: boolean
  capabilitiesError: string | null
  onRefreshAccess: () => void
}

const configSection = (section: ConfigSection, label: string) => (
  <RecoverableLazyRoute
    loader={loadConfigPage}
    routeLabel={label}
    componentProps={{ activeSection: section }}
  />
)

const shellPageElements: Record<ShellRoutePage, React.ReactElement> = {
  home: <RecoverableLazyRoute loader={loadDashboardPage} routeLabel="Home" />,
  'get-started': <RecoverableLazyRoute loader={loadGetStartedPage} routeLabel="Get started" />,
  playground: <RecoverableLazyRoute loader={loadPlaygroundPage} routeLabel="Playground" />,
  recipes: configSection('entrypoints-recipes', 'Recipes'),
  decisions: configSection('decisions', 'Decisions'),
  signals: configSection('signals', 'Signals'),
  projections: configSection('projections', 'Projections'),
  topology: <RecoverableLazyRoute loader={loadTopologyPage} routeLabel="Topology" />,
  editor: <RecoverableLazyRoute loader={loadEditorPage} routeLabel="Config editor" />,
  backends: configSection('models', 'Backend models'),
  'decision-models': (
    <RecoverableLazyRoute loader={loadDecisionModelsWorkspace} routeLabel="Decision models" />
  ),
  'model-hub': <RecoverableLazyRoute loader={loadModelHubPage} routeLabel="Model Hub" />,
  requests: <RecoverableLazyRoute loader={loadInsightsPage} routeLabel="Requests" />,
  'request-record': <RecoverableLazyRoute loader={loadInsightsRecordPage} routeLabel="Request" />,
  'agent-run': <RecoverableLazyRoute loader={loadAgentRunPage} routeLabel="Agent run" />,
  evaluations: <RecoverableLazyRoute loader={loadEvaluationPage} routeLabel="Evaluations" />,
  status: <RecoverableLazyRoute loader={loadStatusPage} routeLabel="Status" />,
  logs: <RecoverableLazyRoute loader={loadLogsPage} routeLabel="Logs" />,
  metrics: <RecoverableLazyRoute loader={loadMonitoringPage} routeLabel="Metrics" />,
  traces: <RecoverableLazyRoute loader={loadTracingPage} routeLabel="Traces" />,
  'router-settings': configSection('global-config', 'Router settings'),
  integrations: configSection('mcp', 'Integrations'),
  access: <RecoverableLazyRoute loader={loadUsersPage} routeLabel="Users & access" />,
  schema: (
    <RecoverableLazyRoute loader={loadConfigSchemaReferencePage} routeLabel="Schema reference" />
  ),
}

/** Redirects a pre-redesign path, keeping its query string and hash. */
function LegacyRedirect({ to }: { to: string }) {
  const location = useLocation()
  const [path, query] = to.split('?')
  const search = new URLSearchParams(query ?? '')
  new URLSearchParams(location.search).forEach((value, key) => search.set(key, value))
  const suffix = search.toString()
  return <Navigate to={`${path}${suffix ? `?${suffix}` : ''}${location.hash}`} replace />
}

function LegacyConfigSectionRedirect() {
  const { section } = useParams<{ section: string }>()
  return <LegacyRedirect to={legacyConfigSectionTarget(section)} />
}

function LegacyInsightRecordRedirect() {
  const { recordId } = useParams<{ recordId: string }>()
  return <LegacyRedirect to={`/observe/requests/${encodeURIComponent(recordId ?? '')}`} />
}

/** Keeps one shell mounted across navigation so its state and animations persist. */
function ShellOutlet() {
  return (
    <AppShellLayout>
      <Outlet />
    </AppShellLayout>
  )
}

function SettingsOutlet() {
  return (
    <SettingsLayout>
      <Outlet />
    </SettingsLayout>
  )
}

export const renderAuthenticatedAppRoutes = ({
  user,
  setupMode,
  capabilities,
  capabilitiesLoading,
  capabilitiesError,
  onRefreshAccess,
}: AuthenticatedAppRoutesProps): React.ReactElement => {
  const renderRoute = (route: (typeof shellRouteDefinitions)[number]) => {
    const allowed = canAccessDashboardPath(user, route.path)
    const page = shellPageElements[route.page]
    const element = !allowed ? (
      <Navigate to="/" replace />
    ) : route.page === 'evaluations' ? (
      <EvaluationAvailabilityRoute
        capability={capabilities?.evaluation}
        isLoading={capabilitiesLoading}
        error={capabilitiesError}
        onRefreshAccess={onRefreshAccess}
      >
        {page}
      </EvaluationAvailabilityRoute>
    ) : (
      page
    )
    return <Route key={route.path} path={route.path} element={element} />
  }
  return (
    <>
      <Route
        path="/setup"
        element={<RecoverableLazyRoute loader={loadSetupWizardPage} routeLabel="Setup" />}
      />
      <Route element={<ShellOutlet />}>
        {shellRouteDefinitions
          .filter((route) => !route.path.startsWith('/settings/'))
          .map(renderRoute)}
        <Route element={<SettingsOutlet />}>
          {shellRouteDefinitions
            .filter((route) => route.path.startsWith('/settings/'))
            .map(renderRoute)}
        </Route>
        {legacyRedirects.map((redirect) => (
          <Route
            key={redirect.from}
            path={redirect.from}
            element={<LegacyRedirect to={redirect.to} />}
          />
        ))}
        <Route path="/config/:section" element={<LegacyConfigSectionRedirect />} />
        <Route path="/insights/:recordId" element={<LegacyInsightRecordRedirect />} />
        <Route
          path="*"
          element={
            setupMode ? <Navigate to={fallbackRouteTarget(setupMode)} replace /> : <NotFoundPage />
          }
        />
      </Route>
    </>
  )
}
