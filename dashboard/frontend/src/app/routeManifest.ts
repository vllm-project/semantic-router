export type ShellRoutePage =
  | 'access-control'
  | 'builder'
  | 'dashboard'
  | 'evaluation'
  | 'fleet-sim'
  | 'fleet-sim-fleets'
  | 'fleet-sim-runs'
  | 'fleet-sim-workloads'
  | 'insights'
  | 'insights-record'
  | 'monitoring'
  | 'openclaw'
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
  { path: '/openclaw', page: 'openclaw' },
]

export const redirectRouteDefinitions: readonly RedirectRouteDefinition[] = [
  { path: '/access', to: '/access/usage' },
]
