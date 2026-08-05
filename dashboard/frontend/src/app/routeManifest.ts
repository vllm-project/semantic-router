export type ShellRoutePage =
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
  | 'security'
  | 'status'
  | 'topology'
  | 'tracing'
  | 'users'

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
  { path: '/logs', page: 'logs' },
  { path: '/insights', page: 'insights' },
  { path: '/insights/:recordId', page: 'insights-record' },
  { path: '/evaluation', page: 'evaluation' },
  { path: '/fleet-sim', page: 'fleet-sim' },
  { path: '/fleet-sim/workloads', page: 'fleet-sim-workloads' },
  { path: '/fleet-sim/fleets', page: 'fleet-sim-fleets' },
  { path: '/fleet-sim/runs', page: 'fleet-sim-runs' },
  { path: '/builder', page: 'builder' },
  { path: '/clawos', page: 'clawos' },
  { path: '/users', page: 'users' },
  { path: '/security', page: 'security' },
]

export const redirectRouteDefinitions: readonly RedirectRouteDefinition[] = [
  { path: '/knowledge-bases', to: '/knowledge-bases/bases' },
  { path: '/taxonomy', to: '/knowledge-bases/bases' },
  { path: '/openclaw', to: '/clawos' },
]

export const fallbackRouteTarget = (setupMode: boolean): string => (
  setupMode ? '/setup' : '/dashboard'
)
