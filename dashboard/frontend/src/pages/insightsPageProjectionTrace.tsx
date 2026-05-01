import { Fragment } from 'react'

import CollapsibleSection from '../components/CollapsibleSection'
import type { ViewField } from '../components/ViewPanel'
import type {
  InsightsRecord,
  ProjectionTrace,
  ProjectionTraceMapping,
  ProjectionTracePartition,
} from './insightsPageTypes'
import styles from './InsightsPage.module.css'

function fmtFixed(n: number | undefined | null, digits = 4): string {
  if (n === undefined || n === null || Number.isNaN(n)) {
    return '—'
  }
  return n.toFixed(digits)
}

export function buildProjectionTraceFields(record: InsightsRecord): ViewField[] {
  if (!record.projection_trace) {
    return []
  }
  const trace = record.projection_trace
  return [
    {
      label: 'Structured diagnostics',
      value: (
        <div className={styles.projectionTraceWrap}>
          {renderProjectionTraceTables(trace)}
          <CollapsibleSection
            id={`projection-trace-raw-${record.id}`}
            title={`Raw JSON (schema ${trace.schema_version || '?'})`}
            defaultExpanded={false}
            content={
              <pre className={styles.bodyPreview}>{JSON.stringify(trace, null, 2)}</pre>
            }
          />
        </div>
      ),
      fullWidth: true,
    },
  ]
}

function renderPartitionBlock(p: ProjectionTracePartition, idx: number) {
  const key = `${p.group_name}-${p.signal_type}-${idx}`
  return (
    <div key={key} className={styles.projectionTraceBlock}>
      <table className={styles.projectionTraceTable}>
        <thead>
          <tr>
            <th>Group</th>
            <th>Signal</th>
            <th>Winner</th>
            <th>Margin</th>
            <th>Semantics</th>
            <th>Winner score</th>
            <th>Raw winner</th>
            <th>Temp</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{p.group_name}</td>
            <td>{p.signal_type}</td>
            <td>{p.default_used && p.winner ? `${p.winner} (default)` : p.winner || '—'}</td>
            <td>{fmtFixed(p.margin)}</td>
            <td>{p.semantics || '—'}</td>
            <td>{fmtFixed(p.winner_score)}</td>
            <td>{fmtFixed(p.raw_winner_score)}</td>
            <td>{p.temperature != null && !Number.isNaN(p.temperature) ? fmtFixed(p.temperature, 3) : '—'}</td>
          </tr>
        </tbody>
      </table>
      {p.contenders && p.contenders.length > 0 ? (
        <>
          <div className={styles.projectionTraceNestedLabel}>Contenders</div>
          <table className={styles.projectionTraceNestedTable}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Raw score</th>
                <th>Normalized</th>
              </tr>
            </thead>
            <tbody>
              {p.contenders.map((c) => (
                <tr key={`${key}-${c.name}`}>
                  <td>{c.name}</td>
                  <td>{fmtFixed(c.raw_score)}</td>
                  <td>{c.normalized_score != null ? fmtFixed(c.normalized_score) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}
    </div>
  )
}

function renderMappingBlock(m: ProjectionTraceMapping, idx: number) {
  const key = `${m.mapping_name}-${idx}`
  const hasOutputs = m.outputs && m.outputs.length > 0
  return (
    <Fragment key={key}>
      <tr>
        <td>{m.mapping_name}</td>
        <td>{m.source_score}</td>
        <td>{fmtFixed(m.score_value)}</td>
        <td>{m.selected_output || '—'}</td>
        <td>{fmtFixed(m.confidence)}</td>
        <td>{fmtFixed(m.boundary_distance)}</td>
      </tr>
      {hasOutputs ? (
        <tr>
          <td colSpan={6} className={styles.projectionTraceNestedCell}>
            <div className={styles.projectionTraceNestedLabel}>Threshold steps</div>
            <table className={styles.projectionTraceNestedTable}>
              <thead>
                <tr>
                  <th>Output</th>
                  <th>Matched</th>
                  <th>Boundary distance</th>
                </tr>
              </thead>
              <tbody>
                {m.outputs!.map((o) => (
                  <tr key={`${key}-${o.name}`}>
                    <td>{o.name}</td>
                    <td>
                      <span className={o.matched ? styles.projectionTraceMatchYes : styles.projectionTraceMatchNo}>
                        {o.matched ? 'yes' : 'no'}
                      </span>
                    </td>
                    <td>{fmtFixed(o.boundary_distance)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </td>
        </tr>
      ) : null}
    </Fragment>
  )
}

function renderProjectionTraceTables(trace: ProjectionTrace) {
  return (
    <>
      {trace.partitions && trace.partitions.length > 0 ? (
        <div>
          <div className={styles.projectionTraceSectionTitle}>Partitions</div>
          {trace.partitions.map((p, idx) => renderPartitionBlock(p, idx))}
        </div>
      ) : null}
      {trace.scores && trace.scores.length > 0 ? (
        <div>
          <div className={styles.projectionTraceSectionTitle}>Score breakdowns</div>
          {trace.scores.map((s) => (
            <div key={s.name}>
              <div className={styles.projectionTraceScoreName}>
                {s.name} (total {s.total.toFixed(4)})
              </div>
              {s.inputs && s.inputs.length > 0 ? (
                <table className={styles.projectionTraceTable}>
                  <thead>
                    <tr>
                      <th>Input</th>
                      <th>Weight</th>
                      <th>Value</th>
                      <th>Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.inputs.map((inp, idx) => (
                      <tr key={`${s.name}-${inp.type}-${inp.name || ''}-${idx}`}>
                        <td>
                          {inp.type}
                          {inp.name ? `:${inp.name}` : ''}
                        </td>
                        <td>{inp.weight}</td>
                        <td>{inp.value.toFixed(4)}</td>
                        <td>{inp.contribution.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {trace.mappings && trace.mappings.length > 0 ? (
        <div>
          <div className={styles.projectionTraceSectionTitle}>Mappings</div>
          <table className={styles.projectionTraceTable}>
            <thead>
              <tr>
                <th>Mapping</th>
                <th>Source score</th>
                <th>Value</th>
                <th>Selected</th>
                <th>Confidence</th>
                <th>Boundary</th>
              </tr>
            </thead>
            <tbody>{trace.mappings.map((m, idx) => renderMappingBlock(m, idx))}</tbody>
          </table>
        </div>
      ) : null}
    </>
  )
}
