import React from 'react'
import { Link } from 'react-router-dom'

import type { BuiltInModelMetadata, CatalogProvider } from '../types/modelCatalog'
import { resolveModelCatalogIcon } from './modelProviderIcons'
import { type ModelHubPagination, type ModelHubStats } from './modelHubSupport'
import styles from './ModelHubPage.module.css'

const CatalogMark: React.FC<{
  presentation: BuiltInModelMetadata['presentation']
  large?: boolean
}> = ({ presentation, large = false }) => {
  const icon = resolveModelCatalogIcon(presentation.logo)
  return (
    <span
      className={`${styles.modelMark} ${large ? styles.modelMarkLarge : ''} ${
        presentation.monochrome ? styles.modelMarkMonochrome : ''
      }`}
      aria-hidden="true"
    >
      {icon ? <img src={icon} alt="" /> : presentation.monogram}
    </span>
  )
}

export const ModelMark: React.FC<{ model: BuiltInModelMetadata; large?: boolean }> = ({
  model,
  large = false,
}) => <CatalogMark presentation={model.presentation} large={large} />

export const ProviderMark: React.FC<{ provider: CatalogProvider }> = ({ provider }) => (
  <CatalogMark presentation={provider.presentation} />
)

const Stat: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div className={styles.statFact}>
    <dd>{value.toLocaleString()}</dd>
    <dt>{label}</dt>
  </div>
)

export const HubHero: React.FC<{ stats: ModelHubStats }> = ({ stats }) => (
  <header className={styles.hero}>
    <div className={styles.heroCopy}>
      <h1>Model Hub</h1>
      <p>Models, ready to route.</p>
    </div>
    <dl className={styles.stats}>
      <Stat label="models" value={stats.models} />
      <Stat label="creators" value={stats.creators} />
      <Stat label="mapped providers" value={stats.mappedProviders} />
      <Stat label="evaluations" value={stats.evaluations} />
    </dl>
    <Link className={styles.primaryAction} to="/config/models">
      Add model
      <span aria-hidden="true">→</span>
    </Link>
  </header>
)

const pageWindow = (page: number, totalPages: number): Array<number | 'gap'> => {
  if (totalPages <= 7) return Array.from({ length: totalPages }, (_, index) => index + 1)
  const values = [...new Set([1, totalPages, page - 1, page, page + 1])]
    .filter((candidate) => candidate > 0 && candidate <= totalPages)
    .sort((left, right) => left - right)
  const output: Array<number | 'gap'> = []
  values.forEach((value, index) => {
    if (index > 0 && value - values[index - 1] > 1) output.push('gap')
    output.push(value)
  })
  return output
}

export const HubPagination: React.FC<{
  pagination: ModelHubPagination<unknown>
  setPage: (page: number) => void
  setPageSize: (pageSize: number) => void
}> = ({ pagination, setPage, setPageSize }) => (
  <nav className={styles.pagination} aria-label="Model catalog pagination">
    <span>
      {pagination.start}–{pagination.end} of {pagination.totalItems.toLocaleString()}
    </span>
    <div className={styles.pageButtons}>
      <button
        type="button"
        onClick={() => setPage(pagination.page - 1)}
        disabled={pagination.page === 1}
        aria-label="Previous page"
      >
        ←
      </button>
      {pageWindow(pagination.page, pagination.totalPages).map((item, index) =>
        item === 'gap' ? (
          <span key={`gap-${index}`} aria-hidden="true">
            …
          </span>
        ) : (
          <button
            type="button"
            key={item}
            className={item === pagination.page ? styles.pageButtonActive : ''}
            aria-current={item === pagination.page ? 'page' : undefined}
            onClick={() => setPage(item)}
          >
            {item}
          </button>
        ),
      )}
      <button
        type="button"
        onClick={() => setPage(pagination.page + 1)}
        disabled={pagination.page === pagination.totalPages}
        aria-label="Next page"
      >
        →
      </button>
    </div>
    <label className={styles.pageSize}>
      <span>Per page</span>
      <select value={pagination.pageSize} onChange={(event) => setPageSize(+event.target.value)}>
        {[10, 20, 50].map((size) => (
          <option value={size} key={size}>
            {size}
          </option>
        ))}
      </select>
    </label>
  </nav>
)
