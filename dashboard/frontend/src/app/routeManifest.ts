export type ShellRoutePage =
  | 'home'
  | 'get-started'
  | 'playground'
  | 'recipes'
  | 'decisions'
  | 'signals'
  | 'projections'
  | 'topology'
  | 'editor'
  | 'backends'
  | 'decision-models'
  | 'model-hub'
  | 'requests'
  | 'request-record'
  | 'agent-run'
  | 'evaluations'
  | 'status'
  | 'logs'
  | 'metrics'
  | 'traces'
  | 'router-settings'
  | 'integrations'
  | 'access'
  | 'schema'

export interface ShellRouteDefinition {
  path: string
  page: ShellRoutePage
}

export const shellRouteDefinitions: readonly ShellRouteDefinition[] = [
  { path: '/', page: 'home' },
  { path: '/get-started', page: 'get-started' },
  { path: '/playground', page: 'playground' },
  { path: '/routing/recipes', page: 'recipes' },
  { path: '/routing/decisions', page: 'decisions' },
  { path: '/routing/signals', page: 'signals' },
  { path: '/routing/projections', page: 'projections' },
  { path: '/routing/topology', page: 'topology' },
  { path: '/routing/editor', page: 'editor' },
  { path: '/models/backends', page: 'backends' },
  { path: '/models/decision', page: 'decision-models' },
  { path: '/models/hub', page: 'model-hub' },
  { path: '/observe/requests', page: 'requests' },
  { path: '/observe/requests/:recordId', page: 'request-record' },
  { path: '/observe/runs', page: 'agent-run' },
  { path: '/observe/evaluations', page: 'evaluations' },
  { path: '/observe/status', page: 'status' },
  { path: '/observe/logs', page: 'logs' },
  { path: '/observe/metrics', page: 'metrics' },
  { path: '/observe/traces', page: 'traces' },
  { path: '/settings/router', page: 'router-settings' },
  { path: '/settings/integrations', page: 'integrations' },
  { path: '/settings/access', page: 'access' },
  { path: '/settings/schema', page: 'schema' },
]

/**
 * Every pre-redesign path keeps working. Query strings are carried over so deep
 * links such as `/topology?scope=…` or `/insights/run?session=…` still land.
 */
export const legacyRedirects: readonly { from: string; to: string }[] = [
  { from: '/dashboard', to: '/' },
  { from: '/config', to: '/settings/router' },
  { from: '/config/reference', to: '/settings/schema' },
  { from: '/topology', to: '/routing/topology' },
  { from: '/builder', to: '/routing/editor' },
  { from: '/models', to: '/models/hub' },
  { from: '/decision-model', to: '/models/decision' },
  { from: '/decision-model/playground', to: '/models/decision?tab=playground' },
  { from: '/decision-model/monitoring', to: '/models/decision?tab=monitoring' },
  { from: '/insights', to: '/observe/requests' },
  { from: '/insights/run', to: '/observe/runs' },
  { from: '/evaluation', to: '/observe/evaluations' },
  { from: '/status', to: '/observe/status' },
  { from: '/logs', to: '/observe/logs' },
  { from: '/monitoring', to: '/observe/metrics' },
  { from: '/tracing', to: '/observe/traces' },
  { from: '/users', to: '/settings/access' },
  { from: '/settings', to: '/settings/router' },
  { from: '/playground/fullscreen', to: '/playground' },
]

const configSectionTargets: Record<string, string> = {
  models: '/models/backends',
  endpoints: '/models/backends',
  signals: '/routing/signals',
  projections: '/routing/projections',
  decisions: '/routing/decisions',
  routes: '/routing/decisions',
  'entrypoints-recipes': '/routing/recipes',
  entrypoints: '/routing/recipes',
  recipes: '/routing/recipes',
  'global-config': '/settings/router',
  global: '/settings/router',
  'router-config': '/settings/router',
  mcp: '/settings/integrations',
  reference: '/settings/schema',
}

/** Target of a legacy `/config/:section` link, or the router settings when unknown. */
export function legacyConfigSectionTarget(section: string | undefined): string {
  return configSectionTargets[(section ?? '').toLowerCase()] ?? '/settings/router'
}

export const fallbackRouteTarget = (setupMode: boolean): string => (setupMode ? '/setup' : '/')
