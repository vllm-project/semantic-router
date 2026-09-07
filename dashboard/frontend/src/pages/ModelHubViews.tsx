import React from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  modelHubContextLabel,
  modelHubDistributionLabel as distributionLabel,
  modelHubRowElementID,
  readableModelHubValue as readable,
  type ModelHubRow,
} from './modelHubSupport'
import { ModelMark } from './ModelHubComponents'
import { ModelHubBenchmarkChart } from './ModelHubBenchmarkViews'
import { OpenModelButton } from './ModelHubOpenModelButton'
import { useModelHubBenchmarkController } from './modelHubBenchmarkController'
import styles from './ModelHubViews.module.css'

const EvaluationSummary: React.FC<{ row: ModelHubRow }> = ({ row }) => (
  <span className={styles.evaluationSummary}>
    <small>Results</small>
    <strong>{row.evaluationCount || '—'}</strong>
    <small>{row.benchmarkCount ? `${row.benchmarkCount} benchmarks` : 'Pending'}</small>
  </span>
)

const virtualPool = (row: ModelHubRow): string[] => [
  ...new Set(row.model.roles?.flatMap((role) => role.recommended_pool) ?? []),
]

const VirtualPoolPreview: React.FC<{ row: ModelHubRow }> = ({ row }) => {
  const candidates = virtualPool(row)
  if (row.model.kind !== 'virtual') return null
  return (
    <span className={styles.listPool}>
      <small>Pool</small>
      <span>
        {candidates.slice(0, 3).map((candidate) => (
          <code key={candidate}>{candidate.split('/').slice(-1)[0]}</code>
        ))}
        {candidates.length > 3 ? <i>+{candidates.length - 3}</i> : null}
      </span>
    </span>
  )
}

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
          <th scope="col">Results</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr
            key={row.model.id}
            id={modelHubRowElementID(row.model.id)}
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
            <td className={styles.tableMeta}>{modelHubContextLabel(row.model)}</td>
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

export const ModelList: React.FC<{
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
}> = ({ rows, selected, select }) => (
  <div className={styles.modelList} role="list" aria-label="Model catalog results">
    {rows.map((row) => (
      <article
        className={styles.modelListItem}
        role="listitem"
        key={row.model.id}
        id={modelHubRowElementID(row.model.id)}
      >
        <OpenModelButton row={row} selected={selected} select={select} className={styles.modelRow}>
          <span className={styles.listIdentity}>
            <ModelMark model={row.model} />
            <span>
              <span className={styles.listTitle}>
                <strong title={row.model.display_name}>{row.model.display_name}</strong>
                <i className={styles.lifecycleDot} data-lifecycle={row.model.lifecycle}>
                  {row.model.lifecycle}
                </i>
              </span>
              <code>{row.model.id}</code>
              <small>{row.model.description}</small>
            </span>
          </span>
          <span className={styles.listFacts}>
            <span>
              <small>Creator</small>
              <strong title={row.model.publisher}>{row.model.publisher}</strong>
            </span>
            <span>
              <small>Context</small>
              <strong>{modelHubContextLabel(row.model)}</strong>
            </span>
            <span>
              <small>{row.model.kind === 'virtual' ? 'Roles' : 'Providers'}</small>
              <strong>
                {row.model.kind === 'virtual'
                  ? row.model.roles?.length || '—'
                  : row.providers.length || '—'}
              </strong>
            </span>
            <EvaluationSummary row={row} />
          </span>
          <span className={styles.listFooter}>
            <span className={styles.cardTags}>
              <i>{distributionLabel[row.model.distribution.type]}</i>
              {row.model.capabilities.slice(0, 2).map((capability) => (
                <i key={capability}>{readable(capability)}</i>
              ))}
            </span>
            <VirtualPoolPreview row={row} />
            <span className={styles.inspectHint}>
              Inspect <b aria-hidden="true">→</b>
            </span>
          </span>
        </OpenModelButton>
      </article>
    ))}
  </div>
)

export const BenchmarkExplorer: React.FC<{
  catalog: BuiltInModelCatalog
  rows: ModelHubRow[]
  openModel: (id: string) => void
}> = ({ catalog, rows, openModel }) => {
  const controller = useModelHubBenchmarkController(catalog, rows)

  return (
    <section className={styles.benchmarkExplorer} aria-label="Benchmark explorer">
      <div className={styles.benchmarkHeading}>
        <div>
          <h2>Benchmarks</h2>
          <span>{controller.charts.length} comparable sets</span>
        </div>
        <small>All published results</small>
      </div>
      {controller.charts.length ? (
        <div className={styles.benchmarkStack}>
          {controller.charts.map((chart) => (
            <article className={styles.benchmarkPanel} key={chart.key}>
              <header className={styles.benchmarkPanelHeading}>
                <div>
                  <h3>{chart.benchmark?.display_name ?? chart.selection.benchmark}</h3>
                  <span>
                    {readable(chart.selection.profile)} · {readable(chart.selection.metric)}
                  </span>
                </div>
                {chart.benchmark?.source ? (
                  <a href={chart.benchmark.source} target="_blank" rel="noreferrer">
                    Source ↗
                  </a>
                ) : null}
              </header>
              <div className={styles.chartMeta}>
                <span>{chart.points.length} results</span>
                <span>
                  {chart.metric?.direction === 'lower_is_better'
                    ? 'Lower is better'
                    : 'Higher is better'}
                </span>
              </div>
              <ModelHubBenchmarkChart rows={rows} openModel={openModel} chart={chart} />
            </article>
          ))}
        </div>
      ) : (
        <EmptyResults title="No available scores" body="Change the model filters." />
      )}
    </section>
  )
}

export const EmptyResults: React.FC<{ title?: string; body?: string }> = ({
  title = 'No matching models',
  body = 'Try another filter.',
}) => (
  <div className={styles.emptyState}>
    <span aria-hidden="true">⌁</span>
    <strong>{title}</strong>
    <small>{body}</small>
  </div>
)
