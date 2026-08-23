/** Router-managed catalog shapes used by the overview dashboard. */

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
