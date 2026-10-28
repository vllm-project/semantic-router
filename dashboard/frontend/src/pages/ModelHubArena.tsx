import React from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { ModelMark } from './ModelHubComponents'
import {
  modelHubArenaData,
  type ModelHubArenaLayer,
  type ModelHubArenaRoute,
  type ModelHubArenaScope,
  type ModelHubArenaSurface,
} from './modelHubArenaSupport'
import styles from './ModelHubArena.module.css'

const layers: Array<[ModelHubArenaLayer, string, string]> = [
  ['overall', 'Overall', 'Complete Intelligence 1.0'],
  ['capabilities', 'Capabilities', 'General, reasoning, coding, agentic'],
  ['benchmarks', 'Benchmarks', 'Six core evaluations'],
]

const scopes: Array<[ModelHubArenaScope, string]> = [
  ['all', 'All models'],
  ['open', 'Open weights'],
  ['virtual', 'Virtual models'],
]

const activeSurface = (
  route: ModelHubArenaRoute,
  overall: ModelHubArenaSurface,
  capabilities: ModelHubArenaSurface[],
  benchmarks: ModelHubArenaSurface[],
): ModelHubArenaSurface => {
  if (route.layer === 'capabilities') {
    return (
      capabilities.find((surface) => surface.id === route.capability) ?? capabilities[0] ?? overall
    )
  }
  if (route.layer === 'benchmarks') {
    return benchmarks.find((surface) => surface.id === route.benchmark) ?? benchmarks[0] ?? overall
  }
  return overall
}

const SurfaceChoices: React.FC<{
  label: string
  surfaces: ModelHubArenaSurface[]
  active: string
  select: (id: string) => void
}> = ({ label, surfaces, active, select }) => (
  <div className={styles.surfaceChoices} role="group" aria-label={label}>
    {surfaces.map((surface) => (
      <button
        key={surface.id}
        type="button"
        aria-pressed={active === surface.id}
        onClick={() => select(surface.id)}
      >
        {surface.displayName}
        <small>{surface.rows.length}</small>
      </button>
    ))}
  </div>
)

const RankChart: React.FC<{
  surface: ModelHubArenaSurface
  openModel: (id: string) => void
}> = ({ surface, openModel }) => {
  const rows = surface.rows.slice(0, 12)
  const maximum = Math.max(...rows.map((row) => row.score), 1)
  return (
    <div className={styles.chart} aria-label={`${surface.displayName} top scores`}>
      {rows.map((row) => (
        <button key={row.model.id} type="button" onClick={() => openModel(row.model.id)}>
          <span>{row.rank}</span>
          <span className={styles.chartModel}>
            <ModelMark model={row.model} />
            <strong title={row.model.display_name}>{row.model.display_name}</strong>
          </span>
          <i aria-hidden="true">
            <b style={{ width: `${(row.score / maximum) * 100}%` }} />
          </i>
          <strong className={styles.chartScore}>{row.score.toFixed(1)}</strong>
        </button>
      ))}
    </div>
  )
}

const RankTable: React.FC<{
  surface: ModelHubArenaSurface
  openModel: (id: string) => void
}> = ({ surface, openModel }) => (
  <div className={styles.tableFrame}>
    <table>
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
        {surface.rows.map((row) => (
          <tr key={row.model.id}>
            <td className={styles.rank}>{`#${row.rank}`}</td>
            <td>
              <button
                type="button"
                className={styles.modelButton}
                onClick={() => openModel(row.model.id)}
              >
                <ModelMark model={row.model} />
                <span>
                  <strong>{row.model.display_name}</strong>
                  <small>{row.model.id}</small>
                </span>
              </button>
            </td>
            <td>{row.model.publisher}</td>
            <td>
              <span className={styles.tableScore}>
                <strong>{row.score.toFixed(1)}</strong>
                <i aria-hidden="true">
                  <b style={{ width: `${row.score}%` }} />
                </i>
              </span>
            </td>
            <td>
              <span className={styles.effort}>{row.reasoningEffort}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)

export const ModelHubArena: React.FC<{
  catalog: BuiltInModelCatalog
  route: ModelHubArenaRoute
  setRoute: (patch: Partial<ModelHubArenaRoute>) => void
  openModel: (id: string) => void
}> = ({ catalog, route, setRoute, openModel }) => {
  const arena = modelHubArenaData(catalog, route.scope)
  if (!arena) return null
  const surface = activeSurface(route, arena.overall, arena.capabilities, arena.benchmarks)

  return (
    <section className={styles.arena} aria-labelledby="model-arena-heading">
      <header className={styles.intro}>
        <div>
          <small>Open Intelligence · 1.0</small>
          <h2 id="model-arena-heading">Model Arena</h2>
          <p>One evidence graph for standalone models, virtual models, and routing.</p>
        </div>
      </header>

      <div className={styles.layerTabs} role="tablist" aria-label="Arena ranking layer">
        {layers.map(([value, label, description]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={route.layer === value}
            onClick={() => setRoute({ layer: value })}
          >
            <strong>{label}</strong>
            <small>{description}</small>
          </button>
        ))}
      </div>

      <div className={styles.controls}>
        {route.layer === 'capabilities' ? (
          <SurfaceChoices
            label="Capability ranking"
            surfaces={arena.capabilities}
            active={surface.id}
            select={(capability) => setRoute({ capability, layer: 'capabilities' })}
          />
        ) : null}
        {route.layer === 'benchmarks' ? (
          <SurfaceChoices
            label="Benchmark ranking"
            surfaces={arena.benchmarks}
            active={surface.id}
            select={(benchmark) => setRoute({ benchmark, layer: 'benchmarks' })}
          />
        ) : null}
        <div className={styles.scopeTabs} role="group" aria-label="Arena model scope">
          {scopes.map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={route.scope === value}
              onClick={() => setRoute({ scope: value })}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.surfaceHeading}>
        <div>
          <small>
            {route.layer === 'overall'
              ? 'Overall rank'
              : route.layer === 'capabilities'
                ? 'Capability rank'
                : 'Benchmark rank'}
          </small>
          <h3>{surface.displayName}</h3>
          <p>{surface.description}</p>
          <div className={styles.surfaceMeta}>
            <code>{surface.id}</code>
            {surface.profiles?.length ? <span>Profile: {surface.profiles.join(' / ')}</span> : null}
            {surface.metric ? <span>Metric: {surface.metric.replace(/_/g, ' ')}</span> : null}
            {surface.source ? (
              <a href={surface.source} target="_blank" rel="noreferrer">
                Source
              </a>
            ) : null}
          </div>
        </div>
        <span>
          <strong>{surface.rows.length}</strong> ranked models
        </span>
      </div>

      {surface.rows.length ? (
        <>
          <RankChart surface={surface} openModel={openModel} />
          <RankTable surface={surface} openModel={openModel} />
        </>
      ) : (
        <p className={styles.empty}>No model in this scope has an available result.</p>
      )}
    </section>
  )
}
