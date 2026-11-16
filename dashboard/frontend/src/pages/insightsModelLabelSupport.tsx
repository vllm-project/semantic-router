import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

import styles from './InsightsRun.module.css'
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
    <span className={styles.modelLabel} title={`Configured as ${name}`}>
      {label}
      {compact ? null : <small>{name}</small>}
    </span>
  )
}
