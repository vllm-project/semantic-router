/** Config snapshot shapes used by the overview dashboard (loosely typed). */

export interface SignalConfig {
  name?: string
  type?: string
  [key: string]: unknown
}

export interface DecisionRule {
  name?: string
  description?: string
  priority?: number
  rules?: unknown[]
  modelRefs?: unknown[]
  plugins?: unknown[]
  routingScope?: string
  routingEntrypoints?: string[]
  [key: string]: unknown
}

export interface RouterConfig {
  signals?: Record<string, SignalConfig[]>
  decisions?: DecisionRule[]
  providers?: {
    models?: Array<{
      name?: string
      backend_refs?: Array<{ name?: string }>
      [key: string]: unknown
    }>
    [key: string]: unknown
  }
  routing?: {
    model_bindings?: Record<string, unknown>
    signals?: Record<string, SignalConfig[]>
    decisions?: DecisionRule[]
  }
  default_recipe?: { name: string; description?: string }
  entrypoints?: Array<{
    api?: 'chat' | 'systemone'
    model_names: string[]
    recipe: string
  }>
  recipes?: Array<{
    name: string
    description?: string
    routing: {
      model_bindings?: Record<string, unknown>
      signals?: Record<string, SignalConfig[]>
      projections?: Record<string, unknown>
      decisions?: DecisionRule[]
      strategy?: string
    }
  }>
  global?: Record<string, unknown>
  [key: string]: unknown
}
