import { lazy, type ComponentType, type LazyExoticComponent } from 'react'

export type RouteLoader = () => Promise<unknown>

export const loadLoginPage = () => import('../pages/LoginPage')
export const loadInviteAcceptPage = () => import('../pages/InviteAcceptPage')
export const loadSetupWizardPage = () => import('../pages/SetupWizardPage')
export const loadDashboardPage = () => import('../pages/DashboardPage')
export const loadGetStartedPage = () => import('../pages/GetStartedPage')
export const loadPlaygroundPage = () => import('../pages/PlaygroundPage')
export const loadConfigPage = () => import('../pages/ConfigPage')
export const loadTopologyPage = () => import('../pages/TopologyPage')
export const loadEditorPage = () => import('../pages/BuilderPage')
export const loadDecisionModelsWorkspace = () => import('../pages/DecisionModelsWorkspace')
export const loadModelHubPage = () => import('../pages/ModelHubPage')
export const loadInsightsPage = () => import('../pages/InsightsPage')
export const loadInsightsRecordPage = () => import('../pages/InsightsRecordPage')
export const loadAgentRunPage = () => import('../pages/InsightsRunPage')
export const loadEvaluationPage = () => import('../pages/EvaluationPage')
export const loadStatusPage = () => import('../pages/StatusPage')
export const loadLogsPage = () => import('../pages/LogsPage')
export const loadMonitoringPage = () => import('../pages/MonitoringPage')
export const loadTracingPage = () => import('../pages/TracingPage')
export const loadUsersPage = () => import('../pages/UsersPage')
export const loadConfigSchemaReferencePage = () => import('../pages/ConfigSchemaReferencePage')

const configPaths = ['/routing/recipes', '/routing/decisions', '/routing/signals', '/routing/projections', '/models/backends', '/settings/router', '/settings/integrations']

const routeLoaders: Array<{ matches: (pathname: string) => boolean; load: RouteLoader }> = [
  { matches: (pathname) => pathname === '/', load: loadDashboardPage },
  { matches: (pathname) => pathname.startsWith('/login'), load: loadLoginPage },
  { matches: (pathname) => pathname.startsWith('/invite/'), load: loadInviteAcceptPage },
  { matches: (pathname) => pathname.startsWith('/setup'), load: loadSetupWizardPage },
  { matches: (pathname) => pathname.startsWith('/get-started'), load: loadGetStartedPage },
  { matches: (pathname) => pathname.startsWith('/playground'), load: loadPlaygroundPage },
  { matches: (pathname) => configPaths.some((path) => pathname.startsWith(path)), load: loadConfigPage },
  { matches: (pathname) => pathname.startsWith('/routing/topology'), load: loadTopologyPage },
  { matches: (pathname) => pathname.startsWith('/routing/editor'), load: loadEditorPage },
  { matches: (pathname) => pathname.startsWith('/models/decision'), load: loadDecisionModelsWorkspace },
  { matches: (pathname) => pathname.startsWith('/models/hub'), load: loadModelHubPage },
  { matches: (pathname) => /^\/observe\/requests\/[^/]+/.test(pathname), load: loadInsightsRecordPage },
  { matches: (pathname) => pathname.startsWith('/observe/requests'), load: loadInsightsPage },
  { matches: (pathname) => pathname.startsWith('/observe/runs'), load: loadAgentRunPage },
  { matches: (pathname) => pathname.startsWith('/observe/evaluations'), load: loadEvaluationPage },
  { matches: (pathname) => pathname.startsWith('/observe/status'), load: loadStatusPage },
  { matches: (pathname) => pathname.startsWith('/observe/logs'), load: loadLogsPage },
  { matches: (pathname) => pathname.startsWith('/observe/metrics'), load: loadMonitoringPage },
  { matches: (pathname) => pathname.startsWith('/observe/traces'), load: loadTracingPage },
  { matches: (pathname) => pathname.startsWith('/settings/access'), load: loadUsersPage },
  { matches: (pathname) => pathname.startsWith('/settings/schema'), load: loadConfigSchemaReferencePage },
]

const routePreloads = new Map<RouteLoader, Promise<unknown>>()
const routePages = new WeakMap<RouteLoader, unknown>()

// A route page must keep one lazy() identity per loader. react-router v7
// navigates inside startTransition, so a suspended render is discarded; a
// lazy() created during render is rebuilt on every retry and never resolves.
export function lazyRoutePage<Props extends object>(
  loader: () => Promise<{ default: ComponentType<Props> }>,
): LazyExoticComponent<ComponentType<Props>> {
  const cached = routePages.get(loader)
  if (cached) return cached as LazyExoticComponent<ComponentType<Props>>

  const page = lazy(loader)
  routePages.set(loader, page)
  return page
}

export function resetDashboardRouteLoader(loader: RouteLoader): void {
  routePreloads.delete(loader)
  routePages.delete(loader)
}

export function preloadDashboardRoute(pathname: string): Promise<unknown> | undefined {
  const loader = routeLoaders.find((route) => route.matches(pathname))?.load
  if (!loader) return undefined

  const existing = routePreloads.get(loader)
  if (existing) return existing

  const preload = loader().catch(() => {
    routePreloads.delete(loader)
    // Intent preloading is opportunistic; do not turn a hover/focus fetch
    // failure into an unhandled rejection before the user actually navigates.
    return undefined
  })
  routePreloads.set(loader, preload)
  return preload
}
