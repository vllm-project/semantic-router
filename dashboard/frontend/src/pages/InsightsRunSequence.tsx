import type { CSSProperties, ReactNode } from 'react'

import { cn } from '../ui'

import { Markers, RunRowDetails } from './InsightsRunDiagram'
import { modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import {
  formatRowCost,
  sideKinds,
  toolLine,
  toolStatusText,
  toolTimeSource,
  toolTimeText,
  type RunArrowRow,
  type RunCallsRow,
  type RunColumn,
  type RunSequence as Sequence,
  type RunSideRow,
  type RunTurnRow,
} from './insightsRunSequence'
import { countText, formatRunCost, formatRunDuration, modelHue } from './insightsRunSupport'
import type { RunCall, RunSummary } from './insightsRunTypes'

// One color per agent column, for its name and the arrows to and from it.
const AGENT_HUES = [
  'var(--foreground)',
  'var(--chart-1)',
  'var(--chart-7)',
  'var(--chart-5)',
  'var(--chart-4)',
  'var(--chart-3)',
]

function agentHue(column: number): string {
  return AGENT_HUES[column % AGENT_HUES.length]
}

const GRID = 'grid grid-cols-[var(--sequence-columns)] gap-x-2.5'
const LIFELINE =
  'row-start-1 bg-[linear-gradient(var(--border-strong),var(--border-strong))] bg-[length:1px_100%] bg-center bg-no-repeat'

interface RunSequenceProps {
  run: RunSummary
  sequence: Sequence
  // The row whose details show under it, and the call picked in it.
  expanded?: string
  call?: RunCall
  /** The run-wide number of each call, by record id. */
  numbers?: Map<string, number>
  narrow?: boolean
  /** The agents the calls table follows; the other agents' rows dim. */
  agents?: string[]
  onToggle: (row: RunCallsRow) => void
  onSelectCall: (call: RunCall) => void
  onAgent?: (label: string) => void
}

function ModelChip({ run, model }: { run: RunSummary; model: string }) {
  const labels = useModelLabels()
  const hue = modelHue(run.models, model)
  return (
    <span
      className="inline-block max-w-full truncate rounded-full border px-1.5 text-[11.5px] font-semibold text-foreground"
      style={{ borderColor: hue, background: `color-mix(in oklab, ${hue} 18%, transparent)` }}
      title={model ? modelText(labels, model) : 'no model'}
    >
      {modelLabel(labels, model) || 'no model'}
    </span>
  )
}

// The lifelines a row crosses, one per agent; the side column has none.
function Lifelines({ columns, kind }: { columns: RunColumn[]; kind: string }) {
  const height =
    kind === 'idle' ? 'min-h-6' : kind === 'spawn' || kind === 'return' ? 'min-h-7' : 'min-h-11'
  return (
    <>
      {columns.map((entry, index) => (
        <span
          key={entry.agent.label}
          data-lifeline
          className={cn(
            LIFELINE,
            height,
            kind === 'idle' &&
              'bg-[repeating-linear-gradient(var(--border-strong)_0_3px,transparent_3px_7px)]',
          )}
          style={{ gridColumn: index + 1 }}
        />
      ))}
    </>
  )
}

// An agent's head: a button that follows its calls when the page offers that.
function AgentHead({
  column,
  hue,
  followed,
  onAgent,
  children,
}: {
  column: RunColumn
  hue: string
  followed: boolean
  onAgent?: (label: string) => void
  children: ReactNode
}) {
  const className =
    'flex w-full min-w-0 flex-col items-center gap-1 rounded-md px-1 py-0.5 text-center group-data-[narrow=true]:flex-row group-data-[narrow=true]:flex-wrap group-data-[narrow=true]:justify-start group-data-[narrow=true]:text-left'
  const style = { '--agent': hue } as CSSProperties
  if (!onAgent) {
    return (
      <div className={className} style={style}>
        {children}
      </div>
    )
  }
  return (
    <button
      type="button"
      aria-pressed={followed}
      title={`Follow ${column.agent.label}'s calls`}
      onClick={() => onAgent(column.agent.label)}
      className={cn(
        className,
        'transition-colors hover:bg-surface-subtle focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40',
        followed && 'bg-primary-soft/60 hover:bg-primary-soft/70',
      )}
      style={style}
    >
      {children}
    </button>
  )
}

function SequenceHeader({
  run,
  columns,
  sides,
  narrow,
  agents,
  onAgent,
}: {
  run: RunSummary
  columns: RunColumn[]
  sides: number
  narrow?: boolean
  agents: string[]
  onAgent?: (label: string) => void
}) {
  return (
    <div
      className={
        narrow
          ? 'flex flex-col gap-2 border-b border-border pb-2.5'
          : cn(GRID, 'sticky top-0 z-[5] border-b border-border bg-surface py-2.5')
      }
    >
      {columns.map((column, index) => {
        const { agent, depth } = column
        return (
          <div
            key={agent.label}
            className="min-w-0"
            style={narrow ? { marginLeft: `${depth * 0.9}rem` } : undefined}
          >
            <AgentHead
              column={column}
              hue={agentHue(index)}
              followed={agents.includes(agent.label)}
              onAgent={onAgent}
            >
              <strong className="font-mono text-[12.5px] font-semibold text-[color:var(--agent)]">
                {depth > 0 ? `↳ ${agent.label}` : agent.label}
              </strong>
              <span className="max-w-full truncate text-[11px] text-subtle-foreground">
                {[agent.kind, countText(agent.calls, 'call'), formatRunCost(agent.cost)]
                  .filter(Boolean)
                  .join(' · ')}
              </span>
              <span className="flex flex-wrap justify-center gap-1">
                {(agent.models ?? []).map((model) => (
                  <ModelChip key={model} run={run} model={model} />
                ))}
              </span>
              {agent.offered_tools?.length ? (
                <details className="max-w-full text-[11px] text-muted-foreground">
                  <summary className="cursor-pointer">
                    {countText(agent.offered_tools.length, 'tool')} offered
                  </summary>
                  <span className="block max-h-32 overflow-y-auto break-all text-left font-mono">
                    {agent.offered_tools.join(', ')}
                  </span>
                </details>
              ) : null}
            </AgentHead>
          </div>
        )
      })}
      {sides ? (
        <div
          data-side="true"
          className="flex min-w-0 flex-col items-center gap-1 text-center opacity-75"
        >
          <strong className="font-mono text-[12.5px] font-semibold">side</strong>
          <span className="text-[11px] text-subtle-foreground">
            {countText(sides, 'side request')}
          </span>
        </div>
      ) : null}
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
      <span
        className="text-[11px] text-muted-foreground data-[failed=true]:text-destructive-foreground"
        data-failed={failed > 0}
      >
        {countText(tools.length, 'tool call')}
        {failed ? `, ${failed} failed` : ''}
      </span>
    )
  }
  return (
    <span className="flex min-w-0 flex-col items-start gap-0.5">
      {tools.slice(0, 8).map((tool, index) => (
        <span
          key={tool.id || index}
          className="max-w-full truncate rounded-[5px] border border-border bg-surface-subtle px-1.5 font-mono text-[11px] text-muted-foreground data-[status=failed]:border-destructive-foreground/50 data-[status=failed]:text-destructive-foreground"
          data-status={tool.status}
          title={`${tool.name}${tool.arguments ? ` ${tool.arguments}` : ''}: ${toolStatusText(tool.status)}${toolTimeText(tool) ? `. ${toolTimeSource(tool)}` : ''}`}
        >
          {toolLine(tool)}
        </span>
      ))}
      {tools.length > 8 ? (
        <span className="text-[11px] text-muted-foreground">+{tools.length - 8} more</span>
      ) : null}
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
    <span className="flex gap-2 text-[11px] text-primary">
      {added.length ? (
        <span title={added.join(', ')}>+{countText(added.length, 'tool')} offered</span>
      ) : null}
      {removed.length ? (
        <span title={removed.join(', ')}>−{countText(removed.length, 'tool')} offered</span>
      ) : null}
    </span>
  )
}

const CARD_TITLE =
  'flex min-w-0 items-center gap-x-2 overflow-hidden whitespace-nowrap text-[12.5px]'
const CARD_META =
  'flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-[11.5px] text-muted-foreground'

function TurnCard({ run, row, narrow }: { run: RunSummary; row: RunTurnRow; narrow?: boolean }) {
  const cost = formatRowCost(row.calls)
  return (
    <>
      <span className={CARD_TITLE}>
        {narrow ? <span className="font-semibold">{row.agent}</span> : null}
        <ModelChip run={run} model={row.model} />
        <span className="text-muted-foreground">
          {[
            row.calls.length > 1 ? countText(row.calls.length, 'call') : '',
            formatRunDuration(row.end - row.begin),
          ]
            .filter(Boolean)
            .join(' · ')}
        </span>
      </span>
      <span className={CARD_META}>
        <span>{[row.decision || 'no decision', cost].filter(Boolean).join(' · ')}</span>
        <Markers markers={row.markers} />
      </span>
      <OfferedChange calls={row.calls} />
      <ToolChips calls={row.calls} />
    </>
  )
}

function SideCard({ run, row, narrow }: { run: RunSummary; row: RunSideRow; narrow?: boolean }) {
  const agents = [...new Set(row.calls.map((call) => call.agent))].join(', ')
  return (
    <>
      <span className={CARD_TITLE}>
        {narrow ? <span className="font-semibold text-foreground">side · {agents}</span> : null}
        <span>
          {row.calls.length > 1 ? countText(row.calls.length, 'side request') : 'side request'}
        </span>
      </span>
      <span className={CARD_META}>
        <span>{sideKinds(row.calls)}</span>
        {[...new Set(row.calls.map((call) => call.model ?? ''))].map((model) => (
          <ModelChip key={model} run={run} model={model} />
        ))}
        <Markers markers={row.markers} />
      </span>
    </>
  )
}

function rowClass(narrow: boolean | undefined, dimmed: boolean) {
  return cn(narrow ? 'flex flex-col' : GRID, 'transition-opacity', dimmed && 'opacity-45')
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
  const agents = props.agents ?? []
  return (
    <div
      id={`run-row-${row.key}`}
      className={rowClass(props.narrow, agents.length > 0 && !agents.includes(agent))}
      data-kind={row.kind}
      role="listitem"
      style={props.narrow ? { marginLeft: `${depth * 0.9}rem` } : undefined}
    >
      {props.narrow ? null : <Lifelines columns={columns} kind={row.kind} />}
      <button
        type="button"
        data-side={side}
        aria-expanded={open}
        className={cn(
          'z-[1] row-start-1 my-1 flex min-w-0 flex-col gap-1 self-center rounded-lg border border-border bg-surface px-2.5 py-1.5 text-left text-foreground shadow-xs transition-colors hover:border-border-strong focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40 aria-expanded:border-primary',
          side && 'border-dashed bg-transparent px-2 py-1 text-muted-foreground shadow-none',
          props.narrow && 'self-stretch',
          props.narrow && side && 'ml-5',
        )}
        style={
          props.narrow ? undefined : { gridColumn: side ? columns.length + 1 : row.column + 1 }
        }
        onClick={() => props.onToggle(row)}
      >
        {side ? (
          <SideCard run={props.run} row={row} narrow={props.narrow} />
        ) : (
          <TurnCard run={props.run} row={row} narrow={props.narrow} />
        )}
      </button>
      {open ? (
        <div className="col-span-full row-start-2 mb-2.5 mt-0.5 max-w-3xl">
          <RunRowDetails
            calls={row.calls}
            call={props.call}
            numbers={props.numbers}
            onSelectCall={props.onSelectCall}
          />
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
  dimmed,
}: {
  row: RunArrowRow
  columns: RunColumn[]
  narrow?: boolean
  dimmed: boolean
}) {
  const child = row.kind === 'spawn' ? row.to : row.from
  const left = Math.min(row.from, row.to)
  const right = Math.max(row.from, row.to)
  const span = right - left + 1
  const hue = agentHue(child)
  const rightward = row.to > row.from
  // The lifelines sit at the middle of the end columns; the line stops short
  // of the head.
  const center = `calc(100% / ${span} / 2)`
  const short = `calc(100% / ${span} / 2 + 8px)`
  const title =
    row.kind === 'spawn'
      ? `${row.parent} spawned ${row.agent}: ${row.label}`
      : `${row.agent} returned to ${row.parent}: ${row.label}`
  return (
    <div
      className={rowClass(narrow, dimmed)}
      data-kind={row.kind}
      role="listitem"
      style={narrow ? { marginLeft: `${(columns[child]?.depth ?? 1) * 0.9}rem` } : undefined}
    >
      {narrow ? null : <Lifelines columns={columns} kind={row.kind} />}
      <span
        className={cn('relative z-[1] row-start-1 self-stretch', narrow && 'py-1')}
        data-kind={row.kind}
        data-direction={rightward ? 'right' : 'left'}
        data-failed={Boolean(row.failed)}
        data-call={row.call}
        title={title}
        style={narrow ? { color: hue } : { gridColumn: `${left + 1} / ${right + 2}`, color: hue }}
      >
        {narrow ? null : (
          <>
            <span
              aria-hidden
              className={cn(
                'absolute top-[64%] border-t-2',
                row.kind === 'return' && 'border-dashed',
              )}
              style={{
                borderColor: hue,
                left: rightward ? center : short,
                right: rightward ? short : center,
              }}
            />
            <span
              aria-hidden
              className="absolute top-[calc(64%-5px)] border-y-[6px] border-y-transparent"
              style={
                rightward
                  ? { right: center, borderLeft: `10px solid ${hue}` }
                  : { left: center, borderRight: `10px solid ${hue}` }
              }
            />
          </>
        )}
        <span
          className={cn(
            'block truncate text-[11.5px] data-[failed=true]:text-destructive-foreground',
            !narrow && 'absolute left-1/2 top-0.5 -translate-x-1/2',
          )}
          data-failed={Boolean(row.failed)}
          style={narrow ? undefined : { maxWidth: `calc(100% - 100% / ${span})` }}
        >
          {arrowText(row, narrow)}
        </span>
      </span>
    </div>
  )
}

function IdleRow({ ms, columns, narrow }: { ms: number; columns: RunColumn[]; narrow?: boolean }) {
  return (
    <div className={narrow ? 'my-1 flex justify-center' : GRID} data-kind="idle" role="listitem">
      {narrow ? null : <Lifelines columns={columns} kind="idle" />}
      <span className="z-[1] col-span-full row-start-1 self-center justify-self-center rounded-full bg-surface px-2 text-[11px] text-muted-foreground">
        · {formatRunDuration(ms)} idle
      </span>
    </div>
  )
}

// A run as a sequence diagram: a lifeline per agent, its calls top to bottom
// in time order with the tools each turn called, side requests in a column of
// their own, an arrow where a subagent starts and one where its result comes
// back, and a divider where the run sat idle. A row opens its calls' details
// under it.
export function RunSequence(props: RunSequenceProps) {
  const { run, sequence, narrow, agents = [] } = props
  const columns = sequence.columns
  const sides = sequence.rows.reduce(
    (count, row) => count + (row.kind === 'side' ? row.calls.length : 0),
    0,
  )
  // The side column shows only when the run made side requests.
  const template = `repeat(${columns.length}, minmax(0, 1fr))${sides ? ' minmax(6.5rem, 0.55fr)' : ''}`
  return (
    <div
      className={cn('group min-w-0', !narrow && 'max-h-[78vh] overflow-y-auto')}
      data-narrow={Boolean(narrow)}
      data-sides={sides > 0}
      role="list"
      aria-label="Run sequence"
      style={{ '--sequence-columns': template } as CSSProperties}
    >
      <SequenceHeader
        run={run}
        columns={columns}
        sides={sides}
        narrow={narrow}
        agents={agents}
        onAgent={props.onAgent}
      />
      {sequence.rows.map((row) => {
        if (row.kind === 'turns' || row.kind === 'side') {
          return <CallsRow key={row.key} {...props} row={row} columns={columns} />
        }
        if (row.kind === 'idle') {
          return (
            <IdleRow key={row.key} ms={row.end - row.begin} columns={columns} narrow={narrow} />
          )
        }
        return (
          <ArrowRow
            key={row.key}
            row={row}
            columns={columns}
            narrow={narrow}
            dimmed={agents.length > 0 && !agents.includes(row.agent)}
          />
        )
      })}
    </div>
  )
}
