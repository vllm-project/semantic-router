import React, { useMemo, useState } from 'react'

import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import { HubFilters, HubHero, ModelDetail, ModelList } from './ModelHubComponents'
import {
  modelHubPublishers,
  modelHubRows,
  modelHubStats,
  type ModelHubFilters,
} from './modelHubSupport'
import styles from './ModelHubPage.module.css'

const initialFilters: ModelHubFilters = {
  query: '',
  kind: 'physical',
  distribution: 'all',
  lifecycle: 'supported',
  publisher: 'all',
  sort: 'intelligence',
}

const ModelHubPage: React.FC = () => {
  const { catalog, error } = useBuiltInModelCatalog()
  const [filters, setFilters] = useState<ModelHubFilters>(initialFilters)
  const [selectedID, setSelectedID] = useState<string | null>(null)
  const stats = useMemo(() => modelHubStats(catalog), [catalog])
  const publishers = useMemo(() => modelHubPublishers(catalog), [catalog])
  const rows = useMemo(() => modelHubRows(catalog, filters), [catalog, filters])
  const selected = rows.find((row) => row.model.id === selectedID) ?? rows[0] ?? null
  const providers = useMemo(
    () => new Map(catalog.providers.map((provider) => [provider.id, provider])),
    [catalog],
  )
  const evaluations = useMemo(
    () => new Map(catalog.evaluations.map((evaluation) => [evaluation.id, evaluation])),
    [catalog],
  )
  const updateFilters = (patch: Partial<ModelHubFilters>): void => {
    setFilters((current) => ({ ...current, ...patch }))
  }

  return (
    <main className={styles.container} data-testid="model-hub-page">
      <HubHero stats={stats} />
      {error ? (
        <div className={styles.notice} role="status">
          Live catalog API unavailable; showing the identical bundled release snapshot. {error}
        </div>
      ) : null}

      <section className={styles.workspace} aria-label="Built-in models">
        <div className={styles.catalogPanel}>
          <HubFilters filters={filters} publishers={publishers} update={updateFilters} />
          <div className={styles.resultHeader}>
            <strong>{rows.length} models</strong>
            <span>Scores require at least 60% benchmark coverage.</span>
          </div>
          <ModelList rows={rows} selected={selected} select={setSelectedID} />
        </div>
        <ModelDetail
          row={selected}
          catalog={catalog}
          providers={providers}
          evaluations={evaluations}
        />
      </section>
    </main>
  )
}

export default ModelHubPage
