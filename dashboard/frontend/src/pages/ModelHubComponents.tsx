import React from 'react'
import { Link } from 'react-router-dom'

import type { BuiltInModelMetadata } from '../types/modelCatalog'
import { resolveModelCatalogIcon } from './modelProviderIcons'
import { type ModelHubPagination, type ModelHubStats } from './modelHubSupport'
import styles from './ModelHubPage.module.css'

export const ModelMark: React.FC<{ model: BuiltInModelMetadata; large?: boolean }> = ({
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

const Stat: React.FC<{ label: string; value: number; hint: string }> = ({ label, value, hint }) => (
  <div className={styles.statCard}>
    <dt>{label}</dt>
    <dd>{value.toLocaleString()}</dd>
    <span>{hint}</span>
  </div>
)

export const HubHero: React.FC<{ stats: ModelHubStats }> = ({ stats }) => (
  <header className={styles.hero}>
    <div className={styles.heroCopy}>
      <span className={styles.eyebrow}>
        <i /> Built into this release
      </span>
      <h1>Find the right model for every route.</h1>
      <p>
        Explore {stats.physicalModels} single models from {stats.creators} mainstream creators,
        centered on recent generations and representative lines. Runtime access, reasoning, and
        sourced benchmark evidence come from the same catalog.
      </p>
      <div className={styles.heroActions}>
        <Link className={styles.primaryAction} to="/config/models">
          Add a model
          <span aria-hidden="true">→</span>
        </Link>
        <a className={styles.secondaryAction} href="#catalog-explorer">
          Explore catalog
        </a>
      </div>
    </div>
    <div className={styles.heroSignal} aria-hidden="true">
      <div className={styles.signalOrb}>
        <span>{stats.models}</span>
        <small>built-in models</small>
      </div>
      <div className={styles.signalTrack} />
      <div className={styles.signalTrackAlt} />
    </div>
    <dl className={styles.stats}>
      <Stat label="Single models" value={stats.physicalModels} hint="Canonical checkpoints" />
      <Stat label="Virtual models" value={stats.virtualModels} hint="Composable recipes" />
      <Stat
        label="Mapped providers"
        value={stats.mappedProviders}
        hint={`${stats.providerContracts} runtime contracts in Add Model`}
      />
      <Stat label="Model creators" value={stats.creators} hint="Mainstream model companies" />
      <Stat
        label="Evaluated models"
        value={stats.evaluatedModels}
        hint={`${stats.evaluations} records`}
      />
    </dl>
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
        {[12, 24, 48, 96].map((size) => (
          <option value={size} key={size}>
            {size}
          </option>
        ))}
      </select>
    </label>
  </nav>
)
