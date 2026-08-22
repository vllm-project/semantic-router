export type ShellRoutePage =
  | 'access-control'
  | 'builder'
  | 'clawos'
  | 'dashboard'
  | 'evaluation'
  | 'fleet-sim'
  | 'fleet-sim-fleets'
  | 'fleet-sim-runs'
  | 'fleet-sim-workloads'
  | 'insights'
  | 'insights-record'
  | 'logs'
  | 'monitoring'
  | 'playground'
  | 'plugins'
  | 'status'
  | 'topology'
  | 'tracing'

export interface ShellRouteDefinition {
  path: string
  page: ShellRoutePage
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
}

export interface RedirectRouteDefinition {
  path: string
  to: string
}

export const shellRouteDefinitions: readonly ShellRouteDefinition[] = [
  { path: '/access/:view', page: 'access-control' },
  { path: '/dashboard', page: 'dashboard' },
  { path: '/monitoring', page: 'monitoring' },
  {
    path: '/playground',
    page: 'playground',
    hideHeaderOnMobile: true,
    hideAccountControl: true,
  },
  { path: '/topology', page: 'topology' },
  { path: '/tracing', page: 'tracing' },
  { path: '/status', page: 'status' },
  { path: '/plugins', page: 'plugins' },
  { path: '/plugins/:plugin', page: 'plugins' },
  { path: '/logs', page: 'access-control' },
  { path: '/insights', page: 'insights' },
  { path: '/insights/:recordId', page: 'insights-record' },
  { path: '/evaluation', page: 'evaluation' },
  { path: '/fleet-sim', page: 'fleet-sim' },
  { path: '/fleet-sim/workloads', page: 'fleet-sim-workloads' },
  { path: '/fleet-sim/fleets', page: 'fleet-sim-fleets' },
  { path: '/fleet-sim/runs', page: 'fleet-sim-runs' },
  { path: '/builder', page: 'builder' },
  { path: '/clawos', page: 'clawos' },
]

export const redirectRouteDefinitions: readonly RedirectRouteDefinition[] = [
  { path: '/access', to: '/access/usage' },
  { path: '/access/overview', to: '/access/usage' },
  { path: '/access/statistics', to: '/access/usage' },
  { path: '/access/request-logs', to: '/logs' },
  { path: '/knowledge-bases', to: '/knowledge-bases/bases' },
  { path: '/taxonomy', to: '/knowledge-bases/bases' },
  { path: '/openclaw', to: '/clawos' },
  { path: '/users', to: '/access/users' },
  { path: '/response-cache', to: '/plugins/response-cache' },
  { path: '/context-compression', to: '/plugins/context-compression' },
]

export const fallbackRouteTarget = (setupMode: boolean): string =>
  setupMode ? '/setup' : '/dashboard'
