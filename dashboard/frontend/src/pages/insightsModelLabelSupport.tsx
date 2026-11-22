import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

import { ProviderLogo, cn } from '../ui'

import {
  buildAgentRuleIndex,
  buildCostBaselineModel,
  buildModelProviders,
  EMPTY_CONFIG_FACTS,
  InsightsConfigContext,
  modelProviderId,
  useInsightsConfig,
  type ConfigDocument,
  type InsightsConfigFacts,
} from './insightsConfigFacts'
import { fetchAbortableInsightsJSON } from './insightsPageRequestSupport'

// The label each configured model reads by, as the Router resolves it from
// the catalog it routes with: the card's display name, else the provider
// model id. Records and runs keep the configured name.
export type ModelLabels = Record<string, string>

interface ModelListResponse {
  object: string
  data: { name: string; label: string }[]
}

export const ModelLabelsContext = createContext<ModelLabels>({})

export function modelLabel(labels: ModelLabels, name?: string): string {
  return (name && labels[name]) || name || ''
}

// A model's label with its configured name, for text and tooltips.
export function modelText(labels: ModelLabels, name: string): string {
  const label = modelLabel(labels, name)
  return label === name ? name : `${label} (${name})`
}

export function useModelLabels(): ModelLabels {
  return useContext(ModelLabelsContext)
}

// Fetches the labels once for the page under it. Without them, as for a
// model the configuration no longer holds, a model reads by its name.
export function ModelLabelsProvider({ children }: { children: ReactNode }) {
  const [labels, setLabels] = useState<ModelLabels>({})
  useEffect(() => {
    const abort = new AbortController()
    fetchAbortableInsightsJSON<ModelListResponse>(
      '/api/router/api/v1/observability/replays/models',
      'model labels',
      abort.signal,
    )
      .then((list) =>
        setLabels(Object.fromEntries((list.data ?? []).map((model) => [model.name, model.label]))),
      )
      .catch(() => undefined)
    return () => abort.abort()
  }, [])
  return <ModelLabelsContext.Provider value={labels}>{children}</ModelLabelsContext.Provider>
}

// A model by its label, with the configured name beside it in small text,
// or only in the tooltip where space is tight.
export function ModelName({
  name,
  fallback = '-',
  compact = false,
}: {
  name?: string
  fallback?: string
  compact?: boolean
}) {
  const label = modelLabel(useModelLabels(), name)
  if (!name) return <>{fallback}</>
  if (label === name) return <>{name}</>
  return (
    <span
      className="inline-flex min-w-0 items-baseline gap-1.5 [&>small]:truncate [&>small]:font-mono [&>small]:text-[11px] [&>small]:font-normal [&>small]:text-subtle-foreground"
      title={`Configured as ${name}`}
    >
      {label}
      {compact ? null : <small>{name}</small>}
    </span>
  )
}

/** A model with its provider's logo when the model id or the configuration names one. */
export function ModelChip({
  name,
  fallback = '-',
  size = 14,
  className,
}: {
  name?: string
  fallback?: string
  size?: number
  className?: string
}) {
  const labels = useModelLabels()
  const { modelProviders } = useInsightsConfig()
  if (!name) return <span className="text-subtle-foreground">{fallback}</span>
  const provider = modelProviderId(name, modelProviders)
  const label = modelLabel(labels, name)
  return (
    <span
      className={cn('inline-flex min-w-0 items-center gap-1.5', className)}
      title={label === name ? name : `${label} · configured as ${name}`}
    >
      {provider ? <ProviderLogo provider={provider} size={size} /> : null}
      <span className="truncate">{label}</span>
    </span>
  )
}

export function InsightsConfigProvider({ children }: { children: ReactNode }) {
  const [facts, setFacts] = useState<InsightsConfigFacts>(EMPTY_CONFIG_FACTS)
  useEffect(() => {
    const abort = new AbortController()
    fetchAbortableInsightsJSON<ConfigDocument>(
      '/api/router/config/all',
      'router config',
      abort.signal,
    )
      .then((config) =>
        setFacts({
          agentRules: buildAgentRuleIndex(config),
          modelProviders: buildModelProviders(config),
          costBaselineModel: buildCostBaselineModel(config),
        }),
      )
      .catch(() => undefined)
    return () => abort.abort()
  }, [])
  return <InsightsConfigContext.Provider value={facts}>{children}</InsightsConfigContext.Provider>
}

/** Model labels and the configuration facts every Insights page reads. */
export function InsightsDataProviders({ children }: { children: ReactNode }) {
  return (
    <ModelLabelsProvider>
      <InsightsConfigProvider>{children}</InsightsConfigProvider>
    </ModelLabelsProvider>
  )
}
