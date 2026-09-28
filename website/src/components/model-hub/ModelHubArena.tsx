import React from 'react'

import type { CatalogSnapshot } from '../../data/modelHubCatalogTypes'
import {
  modelHubArenaData,
  type ModelHubArenaSurface,
} from '../../data/modelHubArenaSupport'
import type {
  ModelHubArenaLayer,
  ModelHubArenaScope,
} from '../../data/modelHubUrlState'
import { CatalogMark } from './ModelHubMark'
import styles from './modelHubArena.module.css'

const scopeOptions: Array<[ModelHubArenaScope, string]> = [
  ['all', 'All models'],
  ['open', 'Open weights'],
  ['virtual', 'Virtual models'],
]

const layerOptions: Array<[ModelHubArenaLayer, string, string]> = [
  ['overall', 'Overall', 'Complete Intelligence 1.0'],
  ['capabilities', 'Capabilities', 'General, reasoning, coding, and agentic'],
  ['benchmarks', 'Benchmarks', 'The six versioned core evaluations'],
]

function selectedSurface(
  layer: ModelHubArenaLayer,
  overall: ModelHubArenaSurface,
  capabilities: ModelHubArenaSurface[],
  benchmarks: ModelHubArenaSurface[],
  capability: string,
  benchmark: string,
): ModelHubArenaSurface {
  if (layer === 'capabilities') {
    return capabilities.find(surface => surface.id === capability) ?? capabilities[0] ?? overall
  }
  if (layer === 'benchmarks') {
    return benchmarks.find(surface => surface.id === benchmark) ?? benchmarks[0] ?? overall
  }
  return overall
}

function SurfaceSelector({
  label,
  surfaces,
  selected,
  select,
}: {
  label: string
  surfaces: ModelHubArenaSurface[]
  selected: string
  select: (id: string) => void
}) {
  return (
    <div className={styles.surfaceSelector} role="group" aria-label={label}>
      {surfaces.map(surface => (
        <button
          key={surface.id}
          type="button"
          aria-pressed={selected === surface.id}
          onClick={() => select(surface.id)}
        >
          <span>{surface.displayName}</span>
          <small>{surface.rows.length}</small>
        </button>
      ))}
    </div>
  )
}

function RankChart({
  surface,
  selectModel,
}: {
  surface: ModelHubArenaSurface
  selectModel: (id: string) => void
}) {
  const rows = surface.rows.slice(0, 12)
  const maximum = Math.max(...rows.map(row => row.score), 1)
  return (
    <div className={styles.chart} aria-label={`${surface.displayName} top model scores`}>
      {rows.map(row => (
        <button
          key={row.model.id}
          type="button"
          className={styles.chartRow}
          onClick={() => selectModel(row.model.id)}
        >
          <span className={styles.chartRank}>{row.rank}</span>
          <span className={styles.chartModel} title={row.model.display_name}>
            <CatalogMark presentation={row.model.presentation} />
            <strong>{row.model.display_name}</strong>
          </span>
          <i aria-hidden="true">
            <b style={{ width: `${(row.score / maximum) * 100}%` }} />
          </i>
          <span className={styles.chartScore}>{row.score.toFixed(1)}</span>
        </button>
      ))}
    </div>
  )
}

function RankTable({
  surface,
  selectModel,
}: {
  surface: ModelHubArenaSurface
  selectModel: (id: string) => void
}) {
  return (
    <div className={styles.tableFrame}>
      <table>
        <colgroup>
          <col className={styles.rankColumn} />
          <col className={styles.modelColumn} />
          <col className={styles.creatorColumn} />
          <col className={styles.scoreColumn} />
          <col className={styles.effortColumn} />
        </colgroup>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Model</th>
            <th>Creator</th>
            <th>Score</th>
            <th>Effort</th>
          </tr>
        </thead>
        <tbody>
          {surface.rows.map(row => (
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
              <td>
                <span className={styles.score}>
                  <strong>{row.score.toFixed(1)}</strong>
                  <i aria-hidden="true"><b style={{ width: `${row.score}%` }} /></i>
                </span>
              </td>
              <td><span className={styles.effort}>{row.reasoningEffort}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ModelHubArena({
  catalog,
  scope,
  setScope,
  layer,
  setLayer,
  capability,
  setCapability,
  benchmark,
  setBenchmark,
  selectModel,
}: {
  catalog: CatalogSnapshot
  scope: ModelHubArenaScope
  setScope: (scope: ModelHubArenaScope) => void
  layer: ModelHubArenaLayer
  setLayer: (layer: ModelHubArenaLayer) => void
  capability: string
  setCapability: (id: string) => void
  benchmark: string
  setBenchmark: (id: string) => void
  selectModel: (id: string) => void
}) {
  const arena = modelHubArenaData(catalog, scope)
  if (!arena) return null
  const surface = selectedSurface(
    layer,
    arena.overall,
    arena.capabilities,
    arena.benchmarks,
    capability,
    benchmark,
  )

  return (
    <div className={styles.arena}>
      <div className={styles.intro}>
        <div>
          <small>Open Intelligence · 1.0</small>
          <h3>{arena.index.display_name}</h3>
          <p>One evidence graph for standalone models, virtual models, and routing.</p>
        </div>
      </div>

      <div className={styles.layerTabs} role="tablist" aria-label="Arena ranking layer">
        {layerOptions.map(([value, label, description]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={layer === value}
            onClick={() => setLayer(value)}
          >
            <strong>{label}</strong>
            <small>{description}</small>
          </button>
        ))}
      </div>

      <div className={styles.controls}>
        {layer === 'capabilities'
          ? (
              <SurfaceSelector
                label="Capability ranking"
                surfaces={arena.capabilities}
                selected={surface.id}
                select={setCapability}
              />
            )
          : null}
        {layer === 'benchmarks'
          ? (
              <SurfaceSelector
                label="Benchmark ranking"
                surfaces={arena.benchmarks}
                selected={surface.id}
                select={setBenchmark}
              />
            )
          : null}
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
      </div>

      <div className={styles.surfaceHeading}>
        <div>
          <small>{layer === 'overall' ? 'Overall rank' : layer === 'capabilities' ? 'Capability rank' : 'Benchmark rank'}</small>
          <h3>{surface.displayName}</h3>
          <p>{surface.description}</p>
          <div className={styles.surfaceMeta}>
            <code>{surface.id}</code>
            {surface.profiles?.length
              ? (
                  <span>
                    Profile:
                    {' '}
                    {surface.profiles.join(' / ')}
                  </span>
                )
              : null}
            {surface.metric
              ? (
                  <span>
                    Metric:
                    {' '}
                    {surface.metric.replace(/_/g, ' ')}
                  </span>
                )
              : null}
            {surface.source ? <a href={surface.source}>Source</a> : null}
          </div>
        </div>
        <span>
          <strong>{surface.rows.length}</strong>
          {' '}
          ranked models
        </span>
      </div>

      {surface.rows.length
        ? (
            <>
              <RankChart surface={surface} selectModel={selectModel} />
              <RankTable surface={surface} selectModel={selectModel} />
            </>
          )
        : (
            <div className={styles.emptyRank}>
              No model in this scope has an available result for
              {' '}
              {surface.displayName}
              .
            </div>
          )}
    </div>
  )
}
