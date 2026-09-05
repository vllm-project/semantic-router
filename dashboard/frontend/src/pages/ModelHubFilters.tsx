import React from 'react'

import type { CatalogProvider } from '../types/modelCatalog'
import {
  type ModelHubDistributionFilter,
  type ModelHubFilters,
  type ModelHubKindFilter,
  type ModelHubLifecycleFilter,
  type ModelHubSort,
  type ModelHubView,
  readableModelHubValue,
} from './modelHubSupport'
import styles from './ModelHubPage.module.css'

interface FilterSelectProps {
  label: string
  value: string
  onChange: (value: string) => void
  children: React.ReactNode
}

const FilterSelect: React.FC<FilterSelectProps> = ({ label, value, onChange, children }) => (
  <label className={styles.filterControl}>
    <span>{label}</span>
    <div className={styles.selectShell}>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {children}
      </select>
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="m4 6 4 4 4-4" />
      </svg>
    </div>
  </label>
)

const ViewIcon: React.FC<{ view: ModelHubView }> = ({ view }) => {
  if (view === 'cards') {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <rect x="2" y="2" width="5" height="5" rx="1" />
        <rect x="9" y="2" width="5" height="5" rx="1" />
        <rect x="2" y="9" width="5" height="5" rx="1" />
        <rect x="9" y="9" width="5" height="5" rx="1" />
      </svg>
    )
  }
  if (view === 'benchmarks') {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M2 13V8m4 5V3m4 10V6m4 7V1" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M2 3h12M2 8h12M2 13h12" />
    </svg>
  )
}

interface HubFiltersProps {
  filters: ModelHubFilters
  creators: string[]
  providers: CatalogProvider[]
  capabilities: string[]
  view: ModelHubView
  update: (patch: Partial<ModelHubFilters>) => void
  setView: (view: ModelHubView) => void
  reset: () => void
}

const activeModelHubFilterCount = (filters: ModelHubFilters): number =>
  [
    filters.query,
    filters.kind !== 'all',
    filters.distribution !== 'all',
    filters.lifecycle !== 'supported',
    filters.publisher !== 'all',
    filters.provider !== 'all',
    filters.capability !== 'all',
    filters.sort !== 'released',
  ].filter(Boolean).length

const HubFilterTopline: React.FC<{
  query: string
  view: ModelHubView
  update: HubFiltersProps['update']
  setView: HubFiltersProps['setView']
}> = ({ query, view, update, setView }) => (
  <div className={styles.filterTopline}>
    <label className={styles.searchField}>
      <span className={styles.srOnly}>Search models</span>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="8.5" cy="8.5" r="5.5" />
        <path d="m12.5 12.5 4 4" />
      </svg>
      <input
        type="search"
        value={query}
        onChange={(event) => update({ query: event.target.value })}
        placeholder="Search model, creator, family, capability…"
      />
      {query ? (
        <button type="button" onClick={() => update({ query: '' })} aria-label="Clear search">
          ×
        </button>
      ) : null}
    </label>
    <div className={styles.viewSwitch} role="group" aria-label="Catalog view">
      {(['table', 'cards', 'benchmarks'] as ModelHubView[]).map((candidate) => (
        <button
          key={candidate}
          type="button"
          className={view === candidate ? styles.viewButtonActive : ''}
          onClick={() => setView(candidate)}
          aria-pressed={view === candidate}
          title={`${readableModelHubValue(candidate)} view`}
        >
          <ViewIcon view={candidate} />
          <span>{readableModelHubValue(candidate)}</span>
        </button>
      ))}
    </div>
  </div>
)

const HubIdentityFilters: React.FC<{
  filters: ModelHubFilters
  creators: string[]
  providers: CatalogProvider[]
  capabilities: string[]
  update: HubFiltersProps['update']
}> = ({ filters, creators, providers, capabilities, update }) => (
  <>
    <FilterSelect
      label="Model type"
      value={filters.kind}
      onChange={(value) => update({ kind: value as ModelHubKindFilter })}
    >
      <option value="all">All models</option>
      <option value="physical">Single models</option>
      <option value="virtual">Virtual models</option>
    </FilterSelect>
    <FilterSelect
      label="Model creator"
      value={filters.publisher}
      onChange={(publisher) => update({ publisher })}
    >
      <option value="all">All creators</option>
      {creators.map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
    </FilterSelect>
    <FilterSelect
      label="Serving provider"
      value={filters.provider}
      onChange={(provider) => update({ provider })}
    >
      <option value="all">All serving providers</option>
      {providers.map((provider) => (
        <option key={provider.id} value={provider.id}>
          {provider.display_name}
        </option>
      ))}
    </FilterSelect>
    <FilterSelect
      label="Capability"
      value={filters.capability}
      onChange={(capability) => update({ capability })}
    >
      <option value="all">All capabilities</option>
      {capabilities.map((capability) => (
        <option key={capability} value={capability}>
          {readableModelHubValue(capability)}
        </option>
      ))}
    </FilterSelect>
  </>
)

const HubMetadataFilters: React.FC<{
  filters: ModelHubFilters
  activeCount: number
  update: HubFiltersProps['update']
  reset: HubFiltersProps['reset']
}> = ({ filters, activeCount, update, reset }) => (
  <>
    <FilterSelect
      label="Availability"
      value={filters.distribution}
      onChange={(distribution) =>
        update({ distribution: distribution as ModelHubDistributionFilter })
      }
    >
      <option value="all">All distributions</option>
      <option value="open_weights">Open weights</option>
      <option value="proprietary_api">Proprietary API</option>
      <option value="router_recipe">Router recipe</option>
    </FilterSelect>
    <FilterSelect
      label="Lifecycle"
      value={filters.lifecycle}
      onChange={(lifecycle) => update({ lifecycle: lifecycle as ModelHubLifecycleFilter })}
    >
      <option value="supported">Supported</option>
      <option value="active">Active</option>
      <option value="experimental">Experimental</option>
      <option value="deprecated">Deprecated</option>
      <option value="removed">Removed</option>
      <option value="all">All states</option>
    </FilterSelect>
    <FilterSelect
      label="Sort by"
      value={filters.sort}
      onChange={(sort) => update({ sort: sort as ModelHubSort })}
    >
      <option value="released">Newest release</option>
      <option value="context">Context window</option>
      <option value="providers">Provider coverage</option>
      <option value="name">Model name</option>
    </FilterSelect>
    <button
      className={styles.resetButton}
      type="button"
      onClick={reset}
      disabled={activeCount === 0}
    >
      Reset {activeCount ? `(${activeCount})` : ''}
    </button>
  </>
)

export const HubFilters: React.FC<HubFiltersProps> = ({
  filters,
  creators,
  providers,
  capabilities,
  view,
  update,
  setView,
  reset,
}) => {
  const activeCount = activeModelHubFilterCount(filters)

  return (
    <div className={styles.filterArea}>
      <HubFilterTopline query={filters.query} view={view} update={update} setView={setView} />
      <div className={styles.filters}>
        <HubIdentityFilters
          filters={filters}
          creators={creators}
          providers={providers}
          capabilities={capabilities}
          update={update}
        />
        <HubMetadataFilters
          filters={filters}
          activeCount={activeCount}
          update={update}
          reset={reset}
        />
      </div>
    </div>
  )
}
