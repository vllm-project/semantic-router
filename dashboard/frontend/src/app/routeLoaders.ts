export type RouteLoader = () => Promise<unknown>

export const loadLandingPage = () => import('../pages/LandingPage')
export const loadLoginPage = () => import('../pages/LoginPage')
export const loadInviteAcceptPage = () => import('../pages/InviteAcceptPage')
export const loadBuilderPage = () => import('../pages/BuilderPage')
export const loadConfigPage = () => import('../pages/ConfigPage')
export const loadDashboardPage = () => import('../pages/DashboardPage')
export const loadEvaluationPage = () => import('../pages/EvaluationPage')
export const loadInsightsPage = () => import('../pages/InsightsPage')
export const loadInsightsRecordPage = () => import('../pages/InsightsRecordPage')
export const loadKnowledgeMapPage = () => import('../pages/KnowledgeMapPage')
export const loadLogsPage = () => import('../pages/LogsPage')
export const loadMLSetupPage = () => import('../pages/MLSetupPage')
export const loadMonitoringPage = () => import('../pages/MonitoringPage')
export const loadOpenClawPage = () => import('../pages/OpenClawPage')
export const loadPlaygroundFullscreenPage = () => import('../pages/PlaygroundFullscreenPage')
export const loadPlaygroundPage = () => import('../pages/PlaygroundPage')
export const loadSetupWizardPage = () => import('../pages/SetupWizardPage')
export const loadStatusPage = () => import('../pages/StatusPage')
export const loadTaxonomyPage = () => import('../pages/TaxonomyPage')
export const loadTopologyPage = () => import('../pages/TopologyPage')
export const loadTracingPage = () => import('../pages/TracingPage')
export const loadUsersPage = () => import('../pages/UsersPage')

const routeLoaders: Array<{ matches: (pathname: string) => boolean; load: RouteLoader }> = [
  { matches: (pathname) => pathname === '/', load: loadLandingPage },
  { matches: (pathname) => pathname.startsWith('/login'), load: loadLoginPage },
  { matches: (pathname) => pathname.startsWith('/invite/'), load: loadInviteAcceptPage },
  { matches: (pathname) => pathname.startsWith('/setup'), load: loadSetupWizardPage },
  { matches: (pathname) => pathname.startsWith('/dashboard'), load: loadDashboardPage },
  {
    matches: (pathname) => pathname.startsWith('/playground/fullscreen'),
    load: loadPlaygroundFullscreenPage,
  },
  { matches: (pathname) => pathname.startsWith('/playground'), load: loadPlaygroundPage },
  { matches: (pathname) => pathname.startsWith('/builder'), load: loadBuilderPage },
  { matches: (pathname) => pathname.startsWith('/config'), load: loadConfigPage },
  {
    matches: (pathname) => /^\/knowledge-bases\/[^/]+\/map\/?$/.test(pathname),
    load: loadKnowledgeMapPage,
  },
  { matches: (pathname) => pathname.startsWith('/knowledge-bases'), load: loadTaxonomyPage },
  { matches: (pathname) => pathname.startsWith('/topology'), load: loadTopologyPage },
  { matches: (pathname) => pathname.startsWith('/openclaw'), load: loadOpenClawPage },
  { matches: (pathname) => /^\/insights\/[^/]+/.test(pathname), load: loadInsightsRecordPage },
  { matches: (pathname) => pathname.startsWith('/insights'), load: loadInsightsPage },
  { matches: (pathname) => pathname.startsWith('/evaluation'), load: loadEvaluationPage },
  { matches: (pathname) => pathname.startsWith('/ml-setup'), load: loadMLSetupPage },
  { matches: (pathname) => pathname.startsWith('/status'), load: loadStatusPage },
  { matches: (pathname) => pathname.startsWith('/logs'), load: loadLogsPage },
  { matches: (pathname) => pathname.startsWith('/monitoring'), load: loadMonitoringPage },
  { matches: (pathname) => pathname.startsWith('/tracing'), load: loadTracingPage },
  { matches: (pathname) => pathname.startsWith('/users'), load: loadUsersPage },
]

const routePreloads = new Map<RouteLoader, Promise<unknown>>()

export function resetDashboardRouteLoader(loader: RouteLoader): void {
  routePreloads.delete(loader)
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
