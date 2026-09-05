import React from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  formatContextWindow,
  modelHubDistributionLabel as distributionLabel,
  readableModelHubValue as readable,
  type ModelHubRow,
} from './modelHubSupport'
import { HubPagination, ModelMark } from './ModelHubComponents'
import { ModelHubBenchmarkChart, ModelHubBenchmarkControls } from './ModelHubBenchmarkViews'
import { OpenModelButton } from './ModelHubOpenModelButton'
import { useModelHubBenchmarkController } from './modelHubBenchmarkController'
import styles from './ModelHubPage.module.css'

const EvaluationSummary: React.FC<{ row: ModelHubRow }> = ({ row }) => (
  <span className={styles.evaluationSummary}>
    <small>Benchmarks</small>
    <strong>{row.benchmarkCount || '—'}</strong>
    <small>{row.evaluationCount ? `${row.evaluationCount} records` : 'Evidence pending'}</small>
  </span>
)

export const ModelTable: React.FC<{
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
}> = ({ rows, selected, select }) => (
  <div className={styles.tableScroller}>
    <p className={styles.tableScrollHint} id="model-hub-table-scroll-hint">
      Swipe horizontally to compare every column <span aria-hidden="true">→</span>
    </p>
    <table
      className={styles.modelTable}
      aria-label="Model catalog results"
      aria-describedby="model-hub-table-scroll-hint"
    >
      <thead>
        <tr className={styles.tableHeader}>
          <th scope="col">Model</th>
          <th scope="col">Distribution</th>
          <th scope="col">Context</th>
          <th scope="col">Providers</th>
          <th scope="col">Benchmarks</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr
            key={row.model.id}
            className={`${styles.tableRow} ${selected?.model.id === row.model.id ? styles.selected : ''}`}
            onClick={() => select(row.model.id)}
          >
            <td>
              <OpenModelButton
                row={row}
                selected={selected}
                select={select}
                className={styles.tableModelButton}
              >
                <span className={styles.modelIdentity}>
                  <ModelMark model={row.model} />
                  <span>
                    <strong>{row.model.display_name}</strong>
                    <small>{row.model.publisher}</small>
                  </span>
                </span>
              </OpenModelButton>
            </td>
            <td className={styles.tableMeta}>{distributionLabel[row.model.distribution.type]}</td>
            <td className={styles.tableMeta}>
              {formatContextWindow(row.model.limits?.context_window_size)}
            </td>
            <td>
              <span className={styles.providerCount}>
                <strong>{row.providers.length || '—'}</strong>
                <small>{row.model.kind === 'virtual' ? 'recipe' : 'paths'}</small>
              </span>
            </td>
            <td>
              <EvaluationSummary row={row} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)

export const ModelCards: React.FC<{
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
}> = ({ rows, selected, select }) => (
  <div className={styles.cardGrid} role="list" aria-label="Model catalog results">
    {rows.map((row) => (
      <article className={styles.cardListItem} role="listitem" key={row.model.id}>
        <OpenModelButton row={row} selected={selected} select={select} className={styles.modelCard}>
          <span className={styles.cardTopline}>
            <ModelMark model={row.model} large />
            <span className={styles.lifecycleDot} data-lifecycle={row.model.lifecycle}>
              {row.model.lifecycle}
            </span>
          </span>
          <span className={styles.cardTitle}>
            <strong>{row.model.display_name}</strong>
            <small>{row.model.publisher}</small>
          </span>
          <span className={styles.cardDescription}>{row.model.description}</span>
          <span className={styles.cardTags}>
            {row.model.capabilities.slice(0, 3).map((capability) => (
              <i key={capability}>{readable(capability)}</i>
            ))}
          </span>
          <span className={styles.cardMetrics}>
            <span>
              <small>Context</small>
              <strong>{formatContextWindow(row.model.limits?.context_window_size)}</strong>
            </span>
            <span>
              <small>Providers</small>
              <strong>{row.providers.length || 'Recipe'}</strong>
            </span>
            <EvaluationSummary row={row} />
          </span>
        </OpenModelButton>
      </article>
    ))}
  </div>
)

export const BenchmarkExplorer: React.FC<{
  catalog: BuiltInModelCatalog
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
  compact?: boolean
}> = ({ catalog, rows, selected, select, compact = false }) => {
  const controller = useModelHubBenchmarkController(catalog, rows, compact)

  return (
    <section className={styles.benchmarkExplorer} aria-label="Benchmark explorer">
      <div className={styles.benchmarkHeading}>
        <div>
          <span>Benchmark explorer</span>
          <h2>Compare like with like</h2>
          <p>
            Choose one benchmark setup. Every bar is a published model and evaluation-condition
            result for that exact metric and profile.
          </p>
        </div>
        {controller.benchmark?.source ? (
          <a href={controller.benchmark.source} target="_blank" rel="noreferrer">
            Benchmark source ↗
          </a>
        ) : null}
      </div>
      <ModelHubBenchmarkControls catalog={catalog} controller={controller} />
      <div className={styles.chartMeta}>
        <span>{controller.points.length} comparable results</span>
        <span>
          {controller.metric?.direction === 'lower_is_better'
            ? 'Lower is better'
            : 'Higher is better'}
        </span>
      </div>
      {controller.pagination.items.length ? (
        <ModelHubBenchmarkChart
          rows={rows}
          selected={selected}
          select={select}
          controller={controller}
        />
      ) : (
        <EmptyResults
          title="No available scores"
          body="The current catalog filters do not contain an available result for this benchmark setup."
        />
      )}
      <HubPagination
        pagination={controller.pagination}
        setPage={controller.setPage}
        setPageSize={controller.setPageSize}
      />
    </section>
  )
}

export const EmptyResults: React.FC<{ title?: string; body?: string }> = ({
  title = 'No matching models',
  body = 'Adjust or reset the filters to restore the catalog.',
}) => (
  <div className={styles.emptyState}>
    <span aria-hidden="true">⌁</span>
    <strong>{title}</strong>
    <small>{body}</small>
  </div>
)
