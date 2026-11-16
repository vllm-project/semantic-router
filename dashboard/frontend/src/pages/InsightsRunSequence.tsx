import type { CSSProperties } from 'react'

import styles from './InsightsRun.module.css'
import { Markers, RunRowDetails } from './InsightsRunDiagram'
import { modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import {
  formatRowCost,
  sideKinds,
  toolLine,
  toolStatusText,
  type RunArrowRow,
  type RunCallsRow,
  type RunColumn,
  type RunRow,
  type RunSequence as Sequence,
  type RunSideRow,
  type RunTurnRow,
} from './insightsRunSequence'
import { countText, formatRunCost, formatRunDuration, modelHue } from './insightsRunSupport'
import type { RunCall, RunSummary } from './insightsRunTypes'

// One color per agent column, for its name and the arrows to and from it.
const AGENT_HUES = ['#d7dae0', '#7aa7e8', '#e0a84f', '#e8798f', '#5cc8b8', '#a395e8']

function agentHue(column: number): string {
  return AGENT_HUES[column % AGENT_HUES.length]
}

interface RunSequenceProps {
  run: RunSummary
  sequence: Sequence
  // The row whose details show under it, and the call picked in it.
  expanded?: string
  call?: RunCall
  narrow?: boolean
  onToggle: (row: RunCallsRow) => void
  onSelectCall: (call: RunCall) => void
}

function ModelChip({ run, model }: { run: RunSummary; model: string }) {
  const labels = useModelLabels()
  return (
    <span
      className={styles.chip}
      style={{ '--chip': modelHue(run.models, model) } as CSSProperties}
      title={model ? modelText(labels, model) : 'no model'}
    >
      {modelLabel(labels, model) || 'no model'}
    </span>
  )
}

// The lifelines a row crosses: one per agent, and the side column's.
function Lifelines({ columns }: { columns: RunColumn[] }) {
  return (
    <>
      {columns.map((entry, index) => (
        <span
          key={entry.agent.label}
          className={styles.lifeline}
          style={{ gridColumn: index + 1 }}
        />
      ))}
      <span
        className={styles.lifeline}
        data-side="true"
        style={{ gridColumn: columns.length + 1 }}
      />
    </>
  )
}

function SequenceHeader({
  run,
  columns,
  sides,
}: {
  run: RunSummary
  columns: RunColumn[]
  sides: number
}) {
  return (
    <div className={styles.sequenceHeader}>
      {columns.map(({ agent, depth }, index) => (
        <div
          key={agent.label}
          className={styles.agentHead}
          style={{ '--depth': depth, '--agent': agentHue(index) } as CSSProperties}
        >
          <strong>{depth > 0 ? `↳ ${agent.label}` : agent.label}</strong>
          <span>
            {[agent.kind, countText(agent.calls, 'call'), formatRunCost(agent.cost)]
              .filter(Boolean)
              .join(' · ')}
          </span>
          <span className={styles.agentModels}>
            {(agent.models ?? []).map((model) => (
              <ModelChip key={model} run={run} model={model} />
            ))}
          </span>
          {agent.offered_tools?.length ? (
            <details className={styles.offered}>
              <summary>{countText(agent.offered_tools.length, 'tool')} offered</summary>
              <span>{agent.offered_tools.join(', ')}</span>
            </details>
          ) : null}
        </div>
      ))}
      <div className={styles.agentHead} data-side="true">
        <strong>side</strong>
        <span>{countText(sides, 'side request')}</span>
      </div>
    </div>
  )
}

// The tool calls of a row's turns: a line each under one call, in the order
// its response asked for them, a count for several calls, listed in full when
// the row opens.
function ToolChips({ calls }: { calls: RunCall[] }) {
  const tools = calls.flatMap((call) => call.tool_calls ?? [])
  if (tools.length === 0) return null
  if (calls.length > 1) {
    const failed = tools.filter((tool) => tool.status === 'failed').length
    return (
      <span className={styles.toolCount} data-failed={failed > 0}>
        {countText(tools.length, 'tool call')}
        {failed ? `, ${failed} failed` : ''}
      </span>
    )
  }
  return (
    <span className={styles.tools}>
      {tools.slice(0, 8).map((tool, index) => (
        <span
          key={tool.id || index}
          className={styles.tool}
          data-status={tool.status}
          title={`${tool.name}${tool.arguments ? ` ${tool.arguments}` : ''}: ${toolStatusText(tool.status)}`}
        >
          {toolLine(tool)}
        </span>
      ))}
      {tools.length > 8 ? <span>+{tools.length - 8} more</span> : null}
    </span>
  )
}

// Where a turn offered other tools than its agent's previous one, as when
// Claude Code's ToolSearch loads deferred tools.
function OfferedChange({ calls }: { calls: RunCall[] }) {
  const added = calls.flatMap((call) => call.offered_tools_added ?? [])
  const removed = calls.flatMap((call) => call.offered_tools_removed ?? [])
  if (added.length + removed.length === 0) return null
  return (
    <span className={styles.offeredChange}>
      {added.length ? (
        <span title={added.join(', ')}>+{countText(added.length, 'tool')} offered</span>
      ) : null}
      {removed.length ? (
        <span title={removed.join(', ')}>−{countText(removed.length, 'tool')} offered</span>
      ) : null}
    </span>
  )
}

function TurnCard({ run, row }: { run: RunSummary; row: RunTurnRow }) {
  const cost = formatRowCost(row.calls)
  return (
    <>
      <span className={styles.cardTitle}>
        <span className={styles.agentTag}>{row.agent}</span>
        <ModelChip run={run} model={row.model} />
        <span>
          {[
            row.calls.length > 1 ? countText(row.calls.length, 'call') : '',
            formatRunDuration(row.end - row.begin),
          ]
            .filter(Boolean)
            .join(' · ')}
        </span>
      </span>
      <span className={styles.cardMeta}>
        <span>{[row.decision || 'no decision', cost].filter(Boolean).join(' · ')}</span>
        <Markers markers={row.markers} />
      </span>
      <OfferedChange calls={row.calls} />
      <ToolChips calls={row.calls} />
    </>
  )
}

function SideCard({ run, row }: { run: RunSummary; row: RunSideRow }) {
  const agents = [...new Set(row.calls.map((call) => call.agent))].join(', ')
  return (
    <>
      <span className={styles.cardTitle}>
        <span className={styles.agentTag}>side · {agents}</span>
        <span>
          {row.calls.length > 1 ? countText(row.calls.length, 'side request') : 'side request'}
        </span>
      </span>
      <span className={styles.cardMeta}>
        <span>{sideKinds(row.calls)}</span>
        {[...new Set(row.calls.map((call) => call.model ?? ''))].map((model) => (
          <ModelChip key={model} run={run} model={model} />
        ))}
        <Markers markers={row.markers} />
      </span>
    </>
  )
}

function CallsRow({
  row,
  columns,
  ...props
}: RunSequenceProps & { row: RunCallsRow; columns: RunColumn[] }) {
  const open = props.expanded === row.key
  const side = row.kind === 'side'
  const agent = side ? row.calls[0].agent : row.agent
  const depth = columns.find((entry) => entry.agent.label === agent)?.depth ?? 0
  return (
    <div
      id={`run-row-${row.key}`}
      className={styles.row}
      data-kind={row.kind}
      role="listitem"
      style={{ '--depth': depth } as CSSProperties}
    >
      {props.narrow ? null : <Lifelines columns={columns} />}
      <button
        type="button"
        className={styles.card}
        data-side={side}
        aria-expanded={open}
        style={{ gridColumn: side ? columns.length + 1 : row.column + 1 }}
        onClick={() => props.onToggle(row)}
      >
        {side ? <SideCard run={props.run} row={row} /> : <TurnCard run={props.run} row={row} />}
      </button>
      {open ? (
        <div className={styles.rowDetails}>
          <RunRowDetails calls={row.calls} call={props.call} onSelectCall={props.onSelectCall} />
        </div>
      ) : null}
    </div>
  )
}

function arrowText(row: RunArrowRow, narrow?: boolean): string {
  if (!narrow) return row.label
  if (row.kind === 'return') return `↩ ${row.agent} ${row.label}`
  return `↳ ${row.parent} spawns ${row.agent}${row.label.startsWith('spawns ') ? '' : `: ${row.label}`}`
}

// A spawn, solid, from the parent's call to the subagent's first, or a
// return, dashed, from the subagent's last call back to the parent's call
// that took its result, in the subagent's color.
function ArrowRow({
  row,
  columns,
  narrow,
}: {
  row: RunArrowRow
  columns: RunColumn[]
  narrow?: boolean
}) {
  const child = row.kind === 'spawn' ? row.to : row.from
  const left = Math.min(row.from, row.to)
  const right = Math.max(row.from, row.to)
  const title =
    row.kind === 'spawn'
      ? `${row.parent} spawned ${row.agent}: ${row.label}`
      : `${row.agent} returned to ${row.parent}: ${row.label}`
  return (
    <div
      className={styles.row}
      data-kind={row.kind}
      role="listitem"
      style={{ '--depth': columns[child]?.depth ?? 1 } as CSSProperties}
    >
      {narrow ? null : <Lifelines columns={columns} />}
      <span
        className={styles.arrow}
        data-kind={row.kind}
        data-direction={row.to > row.from ? 'right' : 'left'}
        data-failed={Boolean(row.failed)}
        data-call={row.call}
        title={title}
        style={
          {
            gridColumn: `${left + 1} / ${right + 2}`,
            '--span': right - left + 1,
            '--arrow': agentHue(child),
          } as CSSProperties
        }
      >
        <span>{arrowText(row, narrow)}</span>
      </span>
    </div>
  )
}

function IdleRow({
  row,
  columns,
  narrow,
}: {
  row: RunRow
  columns: RunColumn[]
  narrow?: boolean
}) {
  return (
    <div className={styles.row} data-kind="idle" role="listitem">
      {narrow ? null : <Lifelines columns={columns} />}
      <span className={styles.idleLabel}>· {formatRunDuration(row.end - row.begin)} idle</span>
    </div>
  )
}

// A run as a sequence diagram: a lifeline per agent, its calls top to bottom
// in time order with the tools each turn called, side requests in a column of
// their own, an arrow where a subagent starts and one where its result comes
// back, and a divider where the run sat idle. A row opens its calls' details
// under it.
export function RunSequence(props: RunSequenceProps) {
  const { run, sequence, narrow } = props
  const columns = sequence.columns
  const sides = sequence.rows.reduce(
    (count, row) => count + (row.kind === 'side' ? row.calls.length : 0),
    0,
  )
  // The side column shows only when the run made side requests.
  const template = `repeat(${columns.length}, minmax(0, 1fr))${sides ? ' minmax(6.5rem, 0.55fr)' : ''}`
  return (
    <div
      className={styles.sequence}
      data-narrow={Boolean(narrow)}
      data-sides={sides > 0}
      role="list"
      aria-label="Run sequence"
      style={{ '--sequence-columns': template } as CSSProperties}
    >
      <SequenceHeader run={run} columns={columns} sides={sides} />
      {sequence.rows.map((row) => {
        if (row.kind === 'turns' || row.kind === 'side') {
          return <CallsRow key={row.key} {...props} row={row} columns={columns} />
        }
        if (row.kind === 'idle') {
          return <IdleRow key={row.key} row={row} columns={columns} narrow={narrow} />
        }
        return <ArrowRow key={row.key} row={row} columns={columns} narrow={narrow} />
      })}
    </div>
  )
}
