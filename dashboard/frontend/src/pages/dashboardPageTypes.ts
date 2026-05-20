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
  [key: string]: unknown
}

export interface RouterConfig {
  signals?: Record<string, SignalConfig[]>
  decisions?: DecisionRule[]
  providers?: {
    defaults?: {
      default_model?: string
    }
    models?: Array<{
      name?: string
      backend_refs?: Array<{ name?: string }>
      endpoints?: Array<{ name?: string }>
      preferred_endpoints?: string[]
      [key: string]: unknown
    }>
    vllm_endpoints?: unknown[]
    [key: string]: unknown
  }
  routing?: {
    signals?: Record<string, SignalConfig[]>
    decisions?: DecisionRule[]
  }
  vllm_endpoints?: Array<{ name?: string }>
  plugins?: Record<string, unknown>
  global?: Record<string, unknown>
  [key: string]: unknown
}
