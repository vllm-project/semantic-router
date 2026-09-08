import React from 'react'

import type { CatalogSnapshot } from '../../data/modelHubCatalogTypes'
import { modelHubArenaData } from '../../data/modelHubArenaSupport'
import type { ModelHubArenaScope } from '../../data/modelHubUrlState'
import { CatalogMark } from './ModelHubMark'
import styles from './modelHubArena.module.css'

const scopeOptions: Array<[ModelHubArenaScope, string]> = [
  ['all', 'All models'],
  ['open', 'Open weights'],
  ['virtual', 'Virtual models'],
]

export function ModelHubArena({
  catalog,
  scope,
  setScope,
  selectModel,
}: {
  catalog: CatalogSnapshot
  scope: ModelHubArenaScope
  setScope: (scope: ModelHubArenaScope) => void
  selectModel: (id: string) => void
}) {
  const arena = modelHubArenaData(catalog, scope)
  if (!arena) return null
  const benchmarkNames = new Map(
    catalog.benchmarks.map(benchmark => [benchmark.id, benchmark.display_name]),
  )
  const pendingPreview = arena.awaitingEvidence.slice(0, 12)

  return (
    <div className={styles.arena}>
      <div className={styles.intro}>
        <div>
          <strong>{arena.index.display_name}</strong>
          <p>{arena.index.description}</p>
        </div>
        {arena.index.methodology
          ? <a href={arena.index.methodology}>Methodology</a>
          : null}
      </div>

      <div className={styles.components} aria-label="Index components">
        {arena.index.components.map(component => (
          <span key={`${component.benchmark}:${component.metric}`}>
            {benchmarkNames.get(component.benchmark ?? '') ?? component.benchmark}
            <b>{`${Math.round(component.weight * 100)}%`}</b>
          </span>
        ))}
      </div>

      <div className={styles.toolbar}>
        <div className={styles.scopeTabs} role="group" aria-label="Arena model scope">
          {scopeOptions.map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={scope === value}
              onClick={() => setScope(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <span>
          <strong>{arena.ranked.length}</strong>
          {' '}
          ranked ·
          {' '}
          <strong>{arena.awaitingEvidence.length}</strong>
          {' '}
          awaiting complete evidence
        </span>
      </div>

      {arena.ranked.length
        ? (
            <div className={styles.tableFrame}>
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Creator</th>
                    <th>Score</th>
                    <th>Coverage</th>
                    <th>Effort</th>
                  </tr>
                </thead>
                <tbody>
                  {arena.ranked.map(row => (
                    <tr key={row.model.id}>
                      <td className={styles.rank}>{`#${row.rank}`}</td>
                      <td>
                        <button
                          type="button"
                          className={styles.modelButton}
                          onClick={() => selectModel(row.model.id)}
                        >
                          <CatalogMark presentation={row.model.presentation} />
                          <span>
                            <strong>{row.model.display_name}</strong>
                            <small>{row.model.id}</small>
                          </span>
                        </button>
                      </td>
                      <td>{row.model.publisher}</td>
                      <td className={styles.score}>{row.result.score?.toFixed(1)}</td>
                      <td>{`${Math.round(row.result.coverage * 100)}%`}</td>
                      <td>{row.result.reasoning_effort}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        : (
            <div className={styles.emptyRank}>No model in this view has all five core results yet.</div>
          )}

      {pendingPreview.length
        ? (
            <div className={styles.coverageQueue}>
              <header>
                <h3>Coverage queue</h3>
                <span>Partial evidence is visible but never converted into a score.</span>
              </header>
              <div className={styles.coverageRows}>
                {pendingPreview.map(row => (
                  <button
                    key={row.model.id}
                    type="button"
                    onClick={() => selectModel(row.model.id)}
                  >
                    <span className={styles.pendingIdentity}>
                      <CatalogMark presentation={row.model.presentation} />
                      <span>
                        <strong>{row.model.display_name}</strong>
                        <small>{row.model.publisher}</small>
                      </span>
                    </span>
                    <span className={styles.coverageMeter}>
                      <i><b style={{ width: `${row.result.coverage * 100}%` }} /></i>
                      <strong>{`${Math.round(row.result.coverage * 100)}%`}</strong>
                    </span>
                    <small className={styles.missing}>
                      {row.missingBenchmarks.length
                        ? `Missing ${row.missingBenchmarks.join(', ')}`
                        : 'No core evidence yet'}
                    </small>
                  </button>
                ))}
              </div>
              {arena.awaitingEvidence.length > pendingPreview.length
                ? (
                    <p className={styles.remaining}>
                      +
                      {arena.awaitingEvidence.length - pendingPreview.length}
                      {' '}
                      more models in the catalog
                    </p>
                  )
                : null}
            </div>
          )
        : null}
    </div>
  )
}
