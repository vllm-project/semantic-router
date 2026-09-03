export type ShellRoutePage =
  | 'builder'
  | 'dashboard'
  | 'evaluation'
  | 'insights'
  | 'insights-record'
  | 'logs'
  | 'monitoring'
  | 'openclaw'
  | 'playground'
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
  { path: '/builder', page: 'builder' },
  { path: '/openclaw', page: 'openclaw' },
  { path: '/users', page: 'users' },
]

export const redirectRouteDefinitions: readonly RedirectRouteDefinition[] = [
  { path: '/knowledge-bases', to: '/knowledge-bases/bases' },
  { path: '/taxonomy', to: '/knowledge-bases/bases' },
]

export const fallbackRouteTarget = (setupMode: boolean): string =>
  setupMode ? '/setup' : '/dashboard'
