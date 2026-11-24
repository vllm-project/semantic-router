import { Fragment, useState } from 'react'
import { Link } from 'react-router-dom'

import { Lock } from 'lucide-react'

import { cn } from '../ui'
import { formatSeconds } from '../utils/dateTime'
import {
  SESSION_ACTION_WORDS,
  SESSION_REASON_TEXT,
  sessionReasonText,
} from './insightsAgentIdentity'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { buildConversationLabels } from './insightsConversationIdentity'
import { Pager } from './InsightsDisplay'
import { DEFAULT_PAGE_SIZE, PAGE_SIZES, pageHolding, pageOf } from './insightsListData'
import { ModelName } from './insightsModelLabelSupport'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'
import {
  assignReviewsToRoutes,
  compactID,
  getReviewsPath,
  getTurnPath,
  ReviewResultBadge,
  summarizeReviews,
} from './insightsReviewSupport'
import type { InsightsTrajectoryRoute } from './insightsPageRoutingTypes'

// Each subagent's dot, in order of first appearance; the main conversation's is plain.
const AGENT_HUES = [
  'var(--chart-4)',
  'var(--chart-3)',
  'var(--chart-2)',
  'var(--chart-5)',
  'var(--chart-1)',
]

function reasonText(route: InsightsTrajectoryRoute) {
  // redaction empties a reason it keeps; one never recorded is left out
  if (route.session_reason === '') return sessionReasonText('')
  const code = route.session_reason || ''
  if (code === 'session_scope_protect' && route.agent !== undefined)
    return "Kept this agent's model"
  return SESSION_REASON_TEXT[code] || code || route.selection_reasoning || 'Not recorded'
}

type Outcome = 'ok' | 'running' | 'failed'

function outcomeOf(route: InsightsTrajectoryRoute): Outcome {
  if (route.lifecycle_state === 'in_progress') return 'running'
  const status = route.response_status ?? 0
  return route.lifecycle_state === 'completed' && status < 400 ? 'ok' : 'failed'
}

// A redacted record keeps the agent field and empties it; the main conversation's has none.
const HIDDEN = '\u0000hidden'
const agentKey = (route: InsightsTrajectoryRoute) =>
  route.agent === '' ? HIDDEN : (route.agent ?? '')

// "subagent 1", "subagent 2", ... by first appearance, the main conversation as main,
// and an agent this viewer may not see as hidden.
function agentNames(routes: InsightsTrajectoryRoute[]) {
  const names = new Map<string, string>([['', 'main']])
  for (const route of routes) {
    const agent = agentKey(route)
    if (!names.has(agent)) names.set(agent, agent === HIDDEN ? 'hidden' : `subagent ${names.size}`)
  }
  return names
}

function spanText(ms: number) {
  const seconds = Math.round(ms / 1_000)
  return seconds < 60
    ? `${(ms / 1_000).toFixed(1)}s`
    : `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

// From the first request's start to the last one's end: subagents run side by side, so
// this is less than the requests' durations added up.
function wallTime(routes: InsightsTrajectoryRoute[]) {
  const starts = routes.map((route) => Date.parse(route.timestamp)).filter(Number.isFinite)
  const ends = routes.flatMap((route) => {
    const start = Date.parse(route.timestamp)
    return Number.isFinite(start) && route.duration_ms > 0 ? [start + route.duration_ms] : []
  })
  if (starts.length === 0 || ends.length === 0) return undefined
  return Math.max(0, Math.max(...ends) - Math.min(...starts))
}

// The session in one strip: who asked, which models answered and how often, how many
// times the model changed, and how the requests ended.
function ReviewSummary({
  reviews,
  session,
  truncated,
}: {
  reviews: InsightsRecord[]
  session?: string
  truncated?: boolean
}) {
  const summary = summarizeReviews(reviews)
  if (summary.total === 0) return null
  return (
    <div>
      <dt>Reviews</dt>
      <dd title="Approval reviews the agent asked for in this session">
        <Link className="font-medium text-primary hover:underline" to={getReviewsPath(session)}>
          {summary.total}
        </Link>
        {` · ${summary.allowed} allowed · ${summary.denied} denied`}
        {summary.failed > 0 ? (
          <span data-outcome="failed" className="text-destructive-foreground">
            {' '}
            · {summary.failed} failed
          </span>
        ) : null}
        {truncated ? ' · older reviews not shown' : null}
      </dd>
    </div>
  )
}

// A turn's reviews, or the review a row itself is, each linking to its record.
function ReviewMarks({ reviews }: { reviews?: InsightsRecord[] }) {
  return reviews?.map((review) => (
    <Link
      key={review.id}
      className="ml-1.5 inline-flex align-middle"
      to={getInsightsRecordPath(review.id)}
      title={`Approval review of ${review.review?.action?.tool || 'an action'}`}
    >
      <ReviewResultBadge review={review.review} />
    </Link>
  ))
}

function Summary({
  routes,
  agents,
  reviews,
  reviewsTruncated,
  session,
}: {
  routes: InsightsTrajectoryRoute[]
  agents: Map<string, string>
  reviews: InsightsRecord[]
  reviewsTruncated?: boolean
  session?: string
}) {
  const models = new Map<string, number>()
  for (const route of routes) {
    const model = route.selected_model || 'no backend'
    models.set(model, (models.get(model) ?? 0) + 1)
  }
  const switches = routes.filter((route) => route.session_action === 'switch').length
  const outcomes = routes.map(outcomeOf)
  const failed = outcomes.filter((one) => one === 'failed').length
  const running = outcomes.filter((one) => one === 'running').length
  const wall = wallTime(routes)
  const guarded = routes.filter((route) => route.session_policy_applied).length
  // the main conversation is listed only when it made a request here
  const seen = [...agents]
    .map(([agent, name], index) => ({ agent, name, hue: agentHue(agent, index) }))
    .filter(({ agent }) => routes.some((route) => agentKey(route) === agent))
  return (
    <dl
      className="grid grid-cols-2 gap-x-6 gap-y-3 px-5 pb-4 sm:grid-cols-3 lg:grid-cols-4 [&_dd]:flex [&_dd]:flex-wrap [&_dd]:items-center [&_dd]:gap-1.5 [&_dd]:text-[13px] [&_dt]:mb-1 [&_dt]:text-xs [&_dt]:text-subtle-foreground"
      aria-label="Session summary"
    >
      <div>
        <dt>Agents</dt>
        <dd>
          {seen.map(({ agent, name, hue }) => (
            <span
              key={agent || 'main'}
              className={CHIP}
              title={agent && agent !== HIDDEN ? agent : undefined}
            >
              <i className="size-2 rounded-full" style={{ background: hue }} />
              {name}
            </span>
          ))}
        </dd>
      </div>
      <div>
        <dt>Models</dt>
        <dd>
          {[...models].map(([model, count]) => (
            <span key={model} className={CHIP}>
              <ModelName name={model} compact />
              <b>×{count}</b>
            </span>
          ))}
        </dd>
      </div>
      <div>
        <dt>Switches</dt>
        <dd
          data-strong={switches > 0 || undefined}
          className={cn('font-mono tabular', switches > 0 && 'font-semibold')}
        >
          {switches}
        </dd>
      </div>
      <div>
        <dt>Results</dt>
        <dd>
          {failed > 0 ? (
            <span data-outcome="failed" className="text-destructive-foreground">
              {failed} failed
            </span>
          ) : (
            <span data-outcome="ok" className="text-success-foreground">
              all answered
            </span>
          )}
          {running > 0 ? (
            <span data-outcome="running" className="text-info-foreground">
              {' '}
              · {running} running
            </span>
          ) : null}
        </dd>
      </div>
      <div>
        <dt>Protected</dt>
        <dd title="Requests whose model the session protection decided">
          {guarded} of {routes.length}
        </dd>
      </div>
      {wall === undefined ? null : (
        <div>
          <dt>Wall time</dt>
          <dd title="From the first request's start to the last one's end">{spanText(wall)}</dd>
        </div>
      )}
      <ReviewSummary reviews={reviews} session={session} truncated={reviewsTruncated} />
    </dl>
  )
}

const CHIP =
  'inline-flex h-6 max-w-full items-center gap-1.5 rounded-full border border-border bg-surface px-2 text-xs [&>b]:font-mono [&>b]:text-[11px] [&>b]:font-medium [&>b]:text-subtle-foreground'

const agentHue = (agent: string, index: number) =>
  agent === HIDDEN
    ? 'var(--subtle-foreground)'
    : agent
      ? AGENT_HUES[(index - 1 + AGENT_HUES.length) % AGENT_HUES.length]
      : 'var(--foreground)'

function ModelCell({ route }: { route: InsightsTrajectoryRoute }) {
  const action = route.session_action || ''
  const word = SESSION_ACTION_WORDS[action] || action
  const switched = Boolean(
    route.previous_model && route.selected_model && route.previous_model !== route.selected_model,
  )
  return (
    <span className="flex min-w-0 flex-wrap items-center gap-1.5">
      {word ? (
        <span
          className={cn(
            'inline-flex items-center gap-1 rounded-md bg-secondary px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground',
            action === 'switch' && 'bg-chart-3/12 text-chart-3',
          )}
          data-action={action}
          title={`${action}${route.session_policy_applied ? ' · session protection applied' : ''}`}
        >
          {word}
          {route.session_policy_applied ? <Lock aria-label="protected" className="size-3" /> : null}
        </span>
      ) : null}
      {switched ? (
        <>
          <span className="text-muted-foreground line-through decoration-border-strong">
            <ModelName name={route.previous_model} compact />
          </span>
          <span aria-hidden="true" className="text-subtle-foreground">
            →
          </span>
        </>
      ) : null}
      <strong className="font-medium">
        <ModelName name={route.selected_model} fallback="No backend selected" compact />
      </strong>
    </span>
  )
}

// Where the request on the page sits in its session, under its route in the header,
// with a link down to the session's history.
export function InsightsSessionPosition({
  trajectory,
  current,
}: {
  trajectory: InsightsTrajectory | null
  current: string
}) {
  const routes = trajectory?.routes ?? []
  const index = routes.findIndex((route) => route.record_id === current)
  if (index < 0 || routes.length < 2) return null
  const agent = agentNames(routes).get(agentKey(routes[index]))
  return (
    <a
      className="text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
      href="#session-history"
    >
      Request {index + 1} of {routes.length} in this session · {agent} · history ↓
    </a>
  )
}

// The Codex turn a call belongs to. A review and a subagent name the turn that started
// them, so their calls join it, as the list's turn filter matches them.
const turnOf = (route?: InsightsTrajectoryRoute) => route?.parent_turn_id || route?.turn_id || ''

// The rows that open a conversation or a Codex turn above a call.
function GroupRows({
  route,
  previous,
  labels,
  turnCalls,
}: {
  route: InsightsTrajectoryRoute
  previous?: InsightsTrajectoryRoute
  labels: Map<string, string>
  turnCalls: Map<string, number>
}) {
  const turn = turnOf(route)
  const calls = turnCalls.get(turn) ?? 0
  return (
    <>
      {labels.size > 0 && (!previous || route.conversation_id !== previous.conversation_id) ? (
        <tr className="bg-surface-subtle/70">
          <td colSpan={8} className="text-xs font-medium text-muted-foreground">
            {route.conversation_id ? (
              <details>
                <summary className="cursor-pointer select-none">
                  {labels.get(route.conversation_id)}
                </summary>
                <code className="mt-1 block font-mono text-[11.5px] font-normal break-all">
                  {route.conversation_id}
                </code>
              </details>
            ) : (
              'Conversation not recorded'
            )}
          </td>
        </tr>
      ) : null}
      {turn && turn !== turnOf(previous) ? (
        <tr className="bg-surface-subtle/70">
          <td colSpan={8} className="text-xs text-muted-foreground">
            <Link
              className="font-medium text-foreground hover:text-primary hover:underline"
              to={getTurnPath(turn)}
              title={turn}
            >
              Codex turn {compactID(turn)}
            </Link>
            {` · ${calls} ${calls === 1 ? 'call' : 'calls'}`}
          </td>
        </tr>
      ) : null}
    </>
  )
}

function HistoryRow({
  route,
  number,
  current,
  agentName,
  hue,
  reviews,
}: {
  route: InsightsTrajectoryRoute
  number: number
  current?: string
  agentName?: string
  hue: string
  reviews?: InsightsRecord[]
}) {
  const agent = agentKey(route)
  const outcome = outcomeOf(route)
  return (
    <tr
      data-record-id={route.record_id}
      data-current={route.record_id === current || undefined}
      data-switch={route.session_action === 'switch' || undefined}
      aria-current={route.record_id === current ? 'true' : undefined}
    >
      <td className="font-mono text-xs tabular">
        <Link className="text-primary hover:underline" to={getInsightsRecordPath(route.record_id)}>
          {number}
        </Link>
      </td>
      <td
        title={
          agent === HIDDEN
            ? 'The agent is hidden from this viewer'
            : agent
              ? `Agent identity hash ${agent}`
              : undefined
        }
      >
        <span className="inline-flex items-center gap-1.5 font-mono text-[12px]">
          <i className="size-2 shrink-0 rounded-full" style={{ background: hue }} />
          {agentName}
        </span>
      </td>
      <td className="font-mono text-xs tabular">
        {route.turn_index + 1}
        <ReviewMarks reviews={reviews} />
      </td>
      <td>
        {route.decision ? (
          <InsightsDecisionBadge
            decision={route.decision}
            label={formatInsightsDecisionName(route.decision)}
            quiet
          />
        ) : (
          'Unresolved'
        )}
      </td>
      <td>
        <ModelCell route={route} />
      </td>
      <td
        className="max-w-[240px] text-xs text-muted-foreground"
        title={route.session_reason || undefined}
      >
        {reasonText(route)}
      </td>
      <td
        title={`${route.lifecycle_state}${route.response_status ? ` · HTTP ${route.response_status}` : ''}`}
      >
        <span
          data-outcome={outcome}
          className={cn(
            'font-mono text-xs',
            outcome === 'failed' && 'text-destructive-foreground',
            outcome === 'running' && 'text-info-foreground',
            outcome === 'ok' && 'text-muted-foreground',
          )}
        >
          {outcome === 'running' ? 'running' : route.response_status || route.lifecycle_state}
        </span>
      </td>
      <td className="text-right font-mono text-xs tabular">
        {outcome === 'running' ? '…' : formatSeconds(route.duration_ms)}
      </td>
    </tr>
  )
}

// A session's requests in time order under its summary, a page at a time,
// opening on the page that holds the request on screen.
export default function InsightsSessionRoutes({
  trajectory,
  current,
  reviews = [],
  reviewsTruncated,
}: {
  trajectory: InsightsTrajectory | null
  current?: string
  reviews?: InsightsRecord[]
  reviewsTruncated?: boolean
}) {
  const routes = trajectory?.routes ?? []
  const [size, setSize] = useState(DEFAULT_PAGE_SIZE)
  const [page, setPage] = useState<number>()
  if (routes.length === 0) return null
  const marks = assignReviewsToRoutes(routes, reviews)
  const ownReview = new Map(reviews.map((review) => [review.id, [review]]))
  const labels = buildConversationLabels(routes)
  const agents = agentNames(routes)
  const agentIndex = new Map([...agents.keys()].map((agent, index) => [agent, index]))
  const turnCalls = new Map<string, number>()
  for (const route of routes) turnCalls.set(turnOf(route), (turnCalls.get(turnOf(route)) ?? 0) + 1)
  const shown = pageOf(
    routes,
    page ??
      pageHolding(
        routes.findIndex((route) => route.record_id === current),
        size,
      ),
    size,
  )
  const offset = (shown.page - 1) * size
  return (
    <section
      id="session-history"
      className="flex min-w-0 scroll-mt-6 flex-col rounded-xl border border-border bg-card shadow-xs"
      aria-label="Session routing history"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2 px-5 pb-3 pt-4">
        <h2 className="text-sm font-semibold tracking-[-0.01em]">Session routing history</h2>
        <span className="text-xs text-subtle-foreground">
          {trajectory?.recipe || 'Default recipe'} · {routes.length} request
          {routes.length === 1 ? '' : 's'}
        </span>
      </div>
      <Summary
        routes={routes}
        agents={agents}
        reviews={reviews}
        reviewsTruncated={reviewsTruncated}
        session={trajectory?.session_id}
      />
      <div className="max-w-full overflow-x-auto border-t border-border">
        <table className="w-full border-collapse text-[13px] [&_td]:px-3 [&_td]:py-2 [&_td]:align-middle [&_td:first-child]:pl-5 [&_th]:px-3 [&_th]:py-2 [&_th:first-child]:pl-5 [&_tbody_tr]:border-t [&_tbody_tr]:border-border [&_tbody_tr[data-current]]:bg-primary-soft/50">
          <thead className="bg-surface-subtle/80">
            <tr>
              {['#', 'Agent', 'Turn', 'Decision', 'Model', 'Why', 'Result', 'Time'].map(
                (heading) => (
                  <th
                    key={heading}
                    className={cn(
                      'whitespace-nowrap text-left text-xs font-medium text-subtle-foreground',
                      heading === 'Time' && 'text-right',
                    )}
                  >
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {shown.rows.map((route, position) => {
              const agent = agentKey(route)
              return (
                <Fragment key={route.record_id}>
                  <GroupRows
                    route={route}
                    previous={position > 0 ? shown.rows[position - 1] : undefined}
                    labels={labels}
                    turnCalls={turnCalls}
                  />
                  <HistoryRow
                    route={route}
                    number={offset + position + 1}
                    current={current}
                    agentName={agents.get(agent)}
                    hue={agentHue(agent, agentIndex.get(agent) ?? 0)}
                    reviews={ownReview.get(route.record_id) ?? marks.get(route.record_id)}
                  />
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
      {routes.length > PAGE_SIZES[0] ? (
        <Pager
          className="mt-0 border-t border-border px-5 py-2.5"
          page={shown.page}
          pageSize={size}
          total={routes.length}
          noun="request"
          onPage={setPage}
          onPageSize={(next) => {
            setSize(next)
            setPage(1)
          }}
        />
      ) : null}
    </section>
  )
}
