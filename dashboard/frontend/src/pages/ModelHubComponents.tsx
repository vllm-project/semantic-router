import React, { useState } from 'react'
import { Link } from 'react-router-dom'

import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogEvaluation,
} from '../types/modelCatalog'
import { resolveModelCatalogIcon } from './modelProviderIcons'
import {
  benchmarkName,
  formatContextWindow,
  formatIntelligence,
  type ModelHubDistributionFilter,
  type ModelHubFilters,
  type ModelHubKindFilter,
  type ModelHubLifecycleFilter,
  type ModelHubRow,
  type ModelHubSort,
  type ModelHubStats,
} from './modelHubSupport'
import styles from './ModelHubPage.module.css'

const distributionLabel: Record<BuiltInModelMetadata['distribution']['type'], string> = {
  open_weights: 'Open weights',
  proprietary_api: 'Proprietary API',
  router_recipe: 'Router recipe',
}

const ModelMark: React.FC<{ model: BuiltInModelMetadata; large?: boolean }> = ({
  model,
  large = false,
}) => {
  const icon = resolveModelCatalogIcon(model.presentation.logo)
  return (
    <span
      className={`${styles.modelMark} ${large ? styles.modelMarkLarge : ''} ${
        model.presentation.monochrome ? styles.modelMarkMonochrome : ''
      }`}
      aria-hidden="true"
    >
      {icon ? <img src={icon} alt="" /> : model.presentation.monogram}
    </span>
  )
}

const scorePercent = (value?: number | null): string =>
  typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'Missing'

const readable = (value: string): string => value.split('_').join(' ')

const subjectSummary = (subject: Record<string, unknown>): string =>
  Object.entries(subject)
    .filter(([key, value]) => key !== 'source_kind' && value !== null && value !== '')
    .map(([key, value]) => `${readable(key)}: ${String(value)}`)
    .join(' · ')

export const HubHero: React.FC<{ stats: ModelHubStats }> = ({ stats }) => (
  <header className={styles.hero}>
    <div>
      <span className={styles.eyebrow}>Built-in model catalog</span>
      <h1>Model Hub</h1>
      <p>
        Explore the canonical models, serving paths, capabilities, and sourced evaluation evidence
        built into this vLLM Semantic Router release.
      </p>
    </div>
    <Link className={styles.connectButton} to="/config/models">
      Add model
    </Link>
    <dl className={styles.stats}>
      {[
        ['Single models', stats.physicalModels],
        ['Virtual recipes', stats.virtualModels],
        ['Providers', stats.providers],
        ['Publishers', stats.publishers],
        ['Comparable scores', stats.scoredModels],
      ].map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  </header>
)

interface FilterSelectProps {
  label: string
  value: string
  onChange: (value: string) => void
  children: React.ReactNode
}

const FilterSelect: React.FC<FilterSelectProps> = ({ label, value, onChange, children }) => (
  <label>
    <span>{label}</span>
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      {children}
    </select>
  </label>
)

export const HubFilters: React.FC<{
  filters: ModelHubFilters
  publishers: string[]
  update: (patch: Partial<ModelHubFilters>) => void
}> = ({ filters, publishers, update }) => (
  <div className={styles.filters}>
    <label className={styles.searchField}>
      <span>Search models</span>
      <input
        type="search"
        value={filters.query}
        onChange={(event) => update({ query: event.target.value })}
        placeholder="Model, family, capability…"
      />
    </label>
    <FilterSelect
      label="Kind"
      value={filters.kind}
      onChange={(value) => update({ kind: value as ModelHubKindFilter })}
    >
      <option value="physical">Single models</option>
      <option value="virtual">Virtual recipes</option>
      <option value="all">All kinds</option>
    </FilterSelect>
    <FilterSelect
      label="Distribution"
      value={filters.distribution}
      onChange={(value) => update({ distribution: value as ModelHubDistributionFilter })}
    >
      <option value="all">All distributions</option>
      <option value="open_weights">Open weights</option>
      <option value="proprietary_api">Proprietary API</option>
      <option value="router_recipe">Router recipe</option>
    </FilterSelect>
    <FilterSelect
      label="Publisher"
      value={filters.publisher}
      onChange={(value) => update({ publisher: value })}
    >
      <option value="all">All publishers</option>
      {publishers.map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
    </FilterSelect>
    <FilterSelect
      label="Lifecycle"
      value={filters.lifecycle}
      onChange={(value) => update({ lifecycle: value as ModelHubLifecycleFilter })}
    >
      <option value="supported">Supported</option>
      <option value="active">Active</option>
      <option value="experimental">Experimental</option>
      <option value="deprecated">Deprecated</option>
      <option value="removed">Removed</option>
      <option value="all">All lifecycle states</option>
    </FilterSelect>
    <FilterSelect
      label="Sort"
      value={filters.sort}
      onChange={(value) => update({ sort: value as ModelHubSort })}
    >
      <option value="intelligence">Intelligence</option>
      <option value="name">Name</option>
    </FilterSelect>
  </div>
)

export const ModelList: React.FC<{
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
}> = ({ rows, selected, select }) => (
  <div className={styles.modelList} role="listbox" aria-label="Model catalog results">
    {rows.map((row) => {
      const active = row.model.id === selected?.model.id
      return (
        <button
          type="button"
          role="option"
          aria-selected={active}
          className={`${styles.modelRow} ${active ? styles.modelRowActive : ''}`}
          key={row.model.id}
          onClick={() => select(row.model.id)}
        >
          <span className={styles.modelIdentity}>
            <ModelMark model={row.model} />
            <span>
              <strong>{row.model.display_name}</strong>
              <small>{row.model.id}</small>
            </span>
          </span>
          <span className={styles.publisher}>{row.model.publisher}</span>
          <span className={styles.distribution}>
            {distributionLabel[row.model.distribution.type]}
          </span>
          <span className={styles.context}>
            {formatContextWindow(row.model.limits?.context_window_size)}
          </span>
          <span
            className={`${styles.score} ${
              row.intelligence?.status === 'available' ? styles.scoreAvailable : ''
            }`}
          >
            {formatIntelligence(row.intelligence)}
            {row.intelligence?.status === 'available' ? (
              <small>
                {row.intelligence.reasoning_effort} · {Math.round(row.intelligence.coverage * 100)}%
              </small>
            ) : null}
          </span>
        </button>
      )
    })}
    {rows.length === 0 ? (
      <div className={styles.emptyState}>
        <strong>No matching models</strong>
        <span>Clear one or more filters to restore the catalog.</span>
      </div>
    ) : null}
  </div>
)

const IntelligenceEvidence: React.FC<{
  row: ModelHubRow
  catalog: BuiltInModelCatalog
  evaluations: Map<string, CatalogEvaluation>
}> = ({ row, catalog, evaluations }) => {
  const [effort, setEffort] = useState(
    row.intelligence?.reasoning_effort ?? row.intelligenceByEffort[0]?.reasoning_effort ?? 'default',
  )
  const result =
    row.intelligenceByEffort.find((candidate) => candidate.reasoning_effort === effort) ??
    row.intelligence
  return (
  <section className={styles.detailSection}>
    <div className={styles.detailSectionTitle}>
      <h3>Intelligence evidence</h3>
      <strong>{formatIntelligence(result)}</strong>
    </div>
    {row.intelligenceByEffort.length > 1 ? (
      <label className={styles.effortSelect}>
        <span>Reasoning effort</span>
        <select value={effort} onChange={(event) => setEffort(event.target.value)}>
          {row.intelligenceByEffort.map((candidate) => (
            <option key={candidate.reasoning_effort} value={candidate.reasoning_effort}>
              {readable(candidate.reasoning_effort)}
            </option>
          ))}
        </select>
      </label>
    ) : null}
    <p>
      {result?.status === 'available'
        ? `${Math.round(result.coverage * 100)}% of the default index is backed by published evidence for this effort.`
        : 'No composite is shown until three of the five default benchmarks are available for this effort.'}
    </p>
    <div className={styles.benchmarkGrid}>
      {result?.components.map((component) => (
        <div
          key={
            component.index ??
            `${component.benchmark}#${component.metric}@${component.benchmark_profile}`
          }
        >
          <span>
            {component.metric
              ? benchmarkName(component.benchmark, component.metric, catalog)
              : (component.index ?? 'Index component')}
          </span>
          <strong>{scorePercent(component.value)}</strong>
        </div>
      ))}
    </div>
    {result?.provenance.length ? (
      <div className={styles.sources}>
        {result.provenance.map((recordID) => {
          const record = evaluations.get(recordID)
          return record ? (
            <div className={styles.evidenceRecord} key={recordID}>
              <span>
                {readable(record.evidence.provenance)} · {record.evidence.verification}
              </span>
              <small>{subjectSummary(record.subject)}</small>
              {record.evidence.source ? (
                <a href={record.evidence.source} target="_blank" rel="noreferrer">
                  Official evidence
                </a>
              ) : null}
            </div>
          ) : null
        })}
      </div>
    ) : null}
  </section>
  )
}

const ProviderList: React.FC<{
  row: ModelHubRow
}> = ({ row }) => (
  <section className={styles.detailSection}>
    <h3>Available through</h3>
    {row.providers.length ? (
      <ul className={styles.providerList}>
        {row.providers.map(({ provider, model }) => (
          <li key={`${provider.id}/${model.id}`}>
            <span>
              <strong>{provider.display_name}</strong>
              <code>{model.id}</code>
            </span>
            <small>{model.lifecycle}</small>
          </li>
        ))}
      </ul>
    ) : (
      <p>This virtual model is materialized from its built-in router recipe.</p>
    )}
  </section>
)

export const ModelDetail: React.FC<{
  row: ModelHubRow | null
  catalog: BuiltInModelCatalog
  evaluations: Map<string, CatalogEvaluation>
}> = ({ row, catalog, evaluations }) => (
  <aside className={styles.detailPanel} aria-live="polite">
    {row ? (
      <>
        <div className={styles.detailHeading}>
          <ModelMark model={row.model} large />
          <div>
            <span>{row.model.publisher}</span>
            <h2>{row.model.display_name}</h2>
            <code>{row.model.id}</code>
          </div>
        </div>
        <p className={styles.description}>{row.model.description}</p>
        <div className={styles.badges}>
          <span>{distributionLabel[row.model.distribution.type]}</span>
          <span>{row.model.lifecycle}</span>
          {row.model.parameter_size ? <span>{row.model.parameter_size}</span> : null}
        </div>
        <IntelligenceEvidence key={row.model.id} row={row} catalog={catalog} evaluations={evaluations} />
        <section className={styles.detailSection}>
          <h3>Capabilities</h3>
          <div className={styles.badges}>
            {row.model.capabilities.map((capability) => (
              <span key={capability}>{readable(capability)}</span>
            ))}
          </div>
        </section>
        <ProviderList row={row} />
        <a
          className={styles.sourceLink}
          href={row.model.distribution.source}
          target="_blank"
          rel="noreferrer"
        >
          Open official model source ↗
        </a>
      </>
    ) : (
      <div className={styles.emptyState}>Select a model to inspect its evidence.</div>
    )}
  </aside>
)
