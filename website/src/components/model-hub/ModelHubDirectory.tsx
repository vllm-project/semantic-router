import React from 'react'

import type {
  CatalogEvaluation,
  CatalogModel,
  CatalogModelBinding,
  CatalogProvider,
  Distribution,
  ModelView,
} from '../../data/modelHubCatalogTypes'
import {
  modelHubActiveFilterCount,
  type ModelHubDirectoryFilters,
} from '../../data/modelHubDirectorySupport'
import { CatalogMark } from './ModelHubMark'
import {
  Badge,
  EmptyState,
  formatDate,
  modelContextLabel,
  Pagination,
  SelectControl,
  srOnlyClass,
} from './ModelHubPrimitives'
import styles from './modelHubDirectory.module.css'

type ProviderBindings = Map<
  string,
  Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
>

const distributionLabel: Record<Distribution, string> = {
  open_weights: 'Open weights',
  proprietary_api: 'Proprietary API',
  router_recipe: 'Router recipe',
}

const candidatePool = (model: CatalogModel): string[] =>
  Array.from(new Set(model.roles?.flatMap(role => role.recommended_pool) ?? []))

function ModelListRow({
  model,
  evaluationCount,
  providerCount,
  onSelect,
}: {
  model: CatalogModel
  evaluationCount: number
  providerCount: number
  onSelect: () => void
}) {
  const pool = candidatePool(model)
  return (
    <article className={styles.modelRow} role="listitem">
      <button type="button" onClick={onSelect} aria-label={`Open ${model.display_name} details`}>
        <span className={styles.modelIdentity}>
          <CatalogMark presentation={model.presentation} />
          <span>
            <span className={styles.modelTitle}>
              <strong title={model.display_name}>{model.display_name}</strong>
              <i data-status={model.lifecycle}>{model.lifecycle}</i>
            </span>
            <span className={styles.modelId} title={model.id}>{model.id}</span>
            <small>{model.description}</small>
          </span>
        </span>
        <span className={styles.modelFacts}>
          <span>
            <small>Creator</small>
            <strong title={model.publisher}>{model.publisher}</strong>
          </span>
          <span>
            <small>Context</small>
            <strong>{modelContextLabel(model)}</strong>
          </span>
          <span>
            <small>{model.kind === 'virtual' ? 'Roles' : 'Providers'}</small>
            <strong>{model.kind === 'virtual' ? model.roles?.length || '—' : providerCount || '—'}</strong>
          </span>
          <span>
            <small>Results</small>
            <strong>{evaluationCount || '—'}</strong>
          </span>
        </span>
        <span className={styles.modelFooter}>
          <Badge value={model.distribution.type} label={distributionLabel[model.distribution.type]} />
          {model.capabilities.slice(0, 2).map(capability => (
            <span className={styles.capability} key={capability}>
              {capability.replace(/_/g, ' ')}
            </span>
          ))}
          {model.kind === 'virtual'
            ? (
                <span className={styles.poolPreview}>
                  <small>Pool</small>
                  {pool.slice(0, 3).map(candidate => (
                    <code key={candidate}>{candidate.split('/').slice(-1)[0]}</code>
                  ))}
                  {pool.length > 3
                    ? (
                        <i>
                          +
                          {pool.length - 3}
                        </i>
                      )
                    : null}
                </span>
              )
            : null}
          <span className={styles.release}>{formatDate(model.released_at)}</span>
          <b aria-hidden="true">→</b>
        </span>
      </button>
    </article>
  )
}

function ModelTable({
  models,
  providersByModel,
  evaluationsByModel,
  onSelect,
}: {
  models: CatalogModel[]
  providersByModel: ProviderBindings
  evaluationsByModel: Map<string, CatalogEvaluation[]>
  onSelect: (id: string) => void
}) {
  return (
    <div className={styles.tableFrame}>
      <table>
        <colgroup>
          <col className={styles.modelColumn} />
          <col className={styles.distributionColumn} />
          <col className={styles.contextColumn} />
          <col className={styles.providersColumn} />
          <col className={styles.resultsColumn} />
          <col className={styles.releasedColumn} />
        </colgroup>
        <thead>
          <tr>
            <th>Model</th>
            <th>Distribution</th>
            <th>Context</th>
            <th>Providers</th>
            <th>Results</th>
            <th>Released</th>
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr key={model.id}>
              <td>
                <button
                  type="button"
                  className={styles.tableModel}
                  aria-label={`View details for ${model.display_name}`}
                  onClick={() => onSelect(model.id)}
                >
                  <CatalogMark presentation={model.presentation} />
                  <span>
                    <strong>{model.display_name}</strong>
                    <small>
                      {model.publisher}
                      {' '}
                      ·
                      {' '}
                      {model.id}
                    </small>
                  </span>
                </button>
              </td>
              <td><Badge value={model.distribution.type} label={distributionLabel[model.distribution.type]} /></td>
              <td>{modelContextLabel(model)}</td>
              <td>{model.kind === 'virtual' ? `${model.roles?.length ?? 0} roles` : providersByModel.get(model.id)?.length ?? '—'}</td>
              <td>{evaluationsByModel.get(model.id)?.length ?? '—'}</td>
              <td>{formatDate(model.released_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ViewToggle({ value, onChange }: { value: ModelView, onChange: (value: ModelView) => void }) {
  return (
    <div className={styles.viewToggle} role="group" aria-label="Model view">
      {(['list', 'table'] as ModelView[]).map(view => (
        <button
          key={view}
          type="button"
          aria-pressed={value === view}
          onClick={() => onChange(view)}
        >
          {view === 'list' ? '☷' : '▦'}
          {' '}
          <span>{view}</span>
        </button>
      ))}
    </div>
  )
}

export function ModelHubDirectory({
  models,
  pageModels,
  publishers,
  providers,
  capabilities,
  providersByModel,
  evaluationsByModel,
  filters,
  updateFilters,
  resetFilters,
  view,
  setView,
  page,
  pageCount,
  pageSize,
  setPage,
  selectModel,
}: {
  models: CatalogModel[]
  pageModels: CatalogModel[]
  publishers: string[]
  providers: CatalogProvider[]
  capabilities: string[]
  providersByModel: ProviderBindings
  evaluationsByModel: Map<string, CatalogEvaluation[]>
  filters: ModelHubDirectoryFilters
  updateFilters: (patch: Partial<ModelHubDirectoryFilters>) => void
  resetFilters: () => void
  view: ModelView
  setView: (view: ModelView) => void
  page: number
  pageCount: number
  pageSize: number
  setPage: (page: number) => void
  selectModel: (id: string) => void
}) {
  const activeFilters = modelHubActiveFilterCount(filters)

  return (
    <div className={styles.directory}>
      <div className={styles.toolbar}>
        <label className={styles.search}>
          <span className={srOnlyClass}>Search models</span>
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="8.5" cy="8.5" r="5.5" />
            <path d="m12.5 12.5 4 4" />
          </svg>
          <input type="search" value={filters.search} onChange={event => updateFilters({ search: event.target.value })} placeholder="Search models" />
        </label>
        <div className={styles.typeTabs} role="group" aria-label="Model type">
          {([['all', 'All'], ['physical', 'Single'], ['virtual', 'Virtual']] as const).map(([kind, label]) => (
            <button key={kind} type="button" aria-pressed={filters.kind === kind} onClick={() => updateFilters({ kind })}>{label}</button>
          ))}
        </div>
        <SelectControl
          label="Sort"
          value={filters.sort}
          options={[
            ['newest', 'Newest'], ['name', 'Name A–Z'], ['context', 'Context'],
          ]}
          onChange={sort => updateFilters({ sort: sort as ModelHubDirectoryFilters['sort'] })}
        />
        <ViewToggle value={view} onChange={setView} />
      </div>
      <div className={styles.filterRow} aria-label="Model filters">
        <SelectControl
          label="Distribution"
          value={filters.distribution}
          options={[
            ['all', 'All distributions'], ['open_weights', 'Open weights'], ['proprietary_api', 'Proprietary API'], ['router_recipe', 'Router recipe'],
          ]}
          onChange={distribution => updateFilters({ distribution: distribution as ModelHubDirectoryFilters['distribution'] })}
        />
        <SelectControl
          label="Creator"
          value={filters.publisher}
          options={[
            ['all', 'All creators'], ...publishers.map(publisher => [publisher, publisher] as [string, string]),
          ]}
          onChange={publisher => updateFilters({ publisher })}
        />
        <SelectControl
          label="Provider"
          value={filters.provider}
          options={[
            ['all', 'All providers'], ...providers.map(provider => [provider.id, provider.display_name] as [string, string]),
          ]}
          onChange={provider => updateFilters({ provider })}
        />
        <SelectControl
          label="Capability"
          value={filters.capability}
          options={[
            ['all', 'All capabilities'], ...capabilities.map(capability => [capability, capability.replace(/_/g, ' ')] as [string, string]),
          ]}
          onChange={capability => updateFilters({ capability })}
        />
        <SelectControl
          label="Lifecycle"
          value={filters.lifecycle}
          options={[
            ['supported', 'Supported'], ['active', 'Active'], ['experimental', 'Experimental'], ['deprecated', 'Deprecated'], ['removed', 'Removed'], ['all', 'All states'],
          ]}
          onChange={lifecycle => updateFilters({ lifecycle: lifecycle as ModelHubDirectoryFilters['lifecycle'] })}
        />
        <button
          type="button"
          className={styles.resetFilters}
          onClick={resetFilters}
          disabled={!activeFilters}
        >
          Reset
        </button>
      </div>
      <div className={styles.resultCount}>
        <strong>{models.length}</strong>
        {' '}
        models
      </div>
      {view === 'list'
        ? (
            <div className={styles.modelList} role="list" aria-label="Model catalog results">
              {pageModels.map(model => (
                <ModelListRow key={model.id} model={model} evaluationCount={evaluationsByModel.get(model.id)?.length ?? 0} providerCount={providersByModel.get(model.id)?.length ?? 0} onSelect={() => selectModel(model.id)} />
              ))}
            </div>
          )
        : (
            <ModelTable models={pageModels} providersByModel={providersByModel} evaluationsByModel={evaluationsByModel} onSelect={selectModel} />
          )}
      {!models.length ? <EmptyState title="No models found" body="Try a broader search or fewer filters." /> : null}
      <Pagination page={page} pageCount={pageCount} total={models.length} pageSize={pageSize} label="models" onChange={setPage} />
    </div>
  )
}
