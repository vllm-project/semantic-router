import { Fragment } from 'react'
import { Link } from 'react-router-dom'

import { formatSeconds } from '../utils/dateTime'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { buildConversationLabels } from './insightsConversationIdentity'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import type { InsightsTrajectory } from './insightsPageTypes'
import type { InsightsTrajectoryRoute } from './insightsPageRoutingTypes'
import styles from './InsightsPage.module.css'

// What the Router did with a request's model, as one word before the model; the
// recorded code stays in the title.
const ACTION_WORD: Record<string, string> = {
  establish_baseline: 'picked',
  stay: 'kept',
  switch: 'switched',
  hard_lock: 'held',
}
const REASON_TEXT: Record<string, string> = {
  new_session: 'New session',
  new_conversation: 'New conversation',
  session_scope_protect: "Kept this session's model",
  previous_model_not_in_candidates: 'The previous model is not a candidate here',
  switch_allowed: 'A better model was allowed',
  stay_has_best_adjusted_score: 'The current model still scores best',
  rescue_underpowered_model: 'Moved up from a model that fell short',
  idle_reset: 'The session was idle, so it started over',
  tool_loop: 'Inside a tool call loop',
  observe_only: 'Observed only',
}
const LOCK = 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z'
// Each subagent's dot, in order of first appearance; the main conversation's is plain.
const AGENT_HUES = ['#5cc8b8', '#a395e8', '#e0a84f', '#e8798f', '#7aa7e8']

function reasonText(route: InsightsTrajectoryRoute) {
  // redaction empties a reason it keeps; one never recorded is left out
  if (route.session_reason === '') return 'Hidden from this viewer'
  const code = route.session_reason || ''
  if (code === 'session_scope_protect' && route.agent !== undefined)
    return "Kept this agent's model"
  return REASON_TEXT[code] || code || route.selection_reasoning || 'Not recorded'
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
function Summary({
  routes,
  agents,
}: {
  routes: InsightsTrajectoryRoute[]
  agents: Map<string, string>
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
    <dl className={styles.historySummary} aria-label="Session summary">
      <div>
        <dt>Agents</dt>
        <dd>
          {seen.map(({ agent, name, hue }) => (
            <span
              key={agent || 'main'}
              className={styles.historyChip}
              title={agent && agent !== HIDDEN ? agent : undefined}
            >
              <i style={{ background: hue }} />
              {name}
            </span>
          ))}
        </dd>
      </div>
      <div>
        <dt>Models</dt>
        <dd>
          {[...models].map(([model, count]) => (
            <span key={model} className={styles.historyChip}>
              {model}
              <b>×{count}</b>
            </span>
          ))}
        </dd>
      </div>
      <div>
        <dt>Switches</dt>
        <dd data-strong={switches > 0 || undefined}>{switches}</dd>
      </div>
      <div>
        <dt>Results</dt>
        <dd>
          {failed > 0 ? (
            <span data-outcome="failed">{failed} failed</span>
          ) : (
            <span data-outcome="ok">all answered</span>
          )}
          {running > 0 ? <span data-outcome="running"> · {running} running</span> : null}
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
    </dl>
  )
}

const agentHue = (agent: string, index: number) =>
  agent === HIDDEN
    ? 'var(--color-text-muted)'
    : agent
      ? AGENT_HUES[(index - 1 + AGENT_HUES.length) % AGENT_HUES.length]
      : 'var(--color-text)'

function ModelCell({ route }: { route: InsightsTrajectoryRoute }) {
  const action = route.session_action || ''
  const word = ACTION_WORD[action] || action
  const switched = Boolean(
    route.previous_model && route.selected_model && route.previous_model !== route.selected_model,
  )
  return (
    <span className={styles.historyModel}>
      {word ? (
        <span
          className={styles.historyVerb}
          data-action={action}
          title={`${action}${route.session_policy_applied ? ' · session protection applied' : ''}`}
        >
          {word}
          {route.session_policy_applied ? (
            <svg viewBox="0 0 24 24" aria-label="protected">
              <path d={LOCK} />
            </svg>
          ) : null}
        </span>
      ) : null}
      {switched ? (
        <>
          <span className={styles.historyPrevious}>{route.previous_model}</span>
          <span aria-hidden="true">→</span>
        </>
      ) : null}
      <strong>{route.selected_model || 'No backend selected'}</strong>
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
    <a className={styles.recordSessionLink} href="#session-history">
      Request {index + 1} of {routes.length} in this session · {agent} · history ↓
    </a>
  )
}

export default function InsightsSessionRoutes({
  trajectory,
  current,
}: {
  trajectory: InsightsTrajectory | null
  current?: string
}) {
  const routes = trajectory?.routes ?? []
  if (routes.length === 0) return null
  const labels = buildConversationLabels(routes)
  const agents = agentNames(routes)
  const agentIndex = new Map([...agents.keys()].map((agent, index) => [agent, index]))
  return (
    <section
      id="session-history"
      className={`${styles.recordSection} ${styles.recordSectionWide}`}
      aria-label="Session routing history"
    >
      <div className={styles.recordSectionHeader}>
        <h2>Session routing history</h2>
        <span>
          {trajectory?.recipe || 'Default recipe'} · {routes.length} requests
        </span>
      </div>
      <Summary routes={routes} agents={agents} />
      <div className={styles.historyWrap}>
        <table className={styles.historyTable}>
          <thead>
            <tr>
              {['#', 'Agent', 'Turn', 'Decision', 'Model', 'Why', 'Result', 'Time'].map(
                (heading) => (
                  <th key={heading}>{heading}</th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {routes.map((route, index) => {
              const agent = agentKey(route)
              const outcome = outcomeOf(route)
              return (
                <Fragment key={route.record_id}>
                  {labels.size > 0 &&
                  (index === 0 || route.conversation_id !== routes[index - 1]?.conversation_id) ? (
                    <tr className={styles.historyGroup}>
                      <td colSpan={8}>
                        {route.conversation_id ? (
                          <details>
                            <summary>{labels.get(route.conversation_id)}</summary>
                            <code>{route.conversation_id}</code>
                          </details>
                        ) : (
                          'Conversation not recorded'
                        )}
                      </td>
                    </tr>
                  ) : null}
                  <tr
                    data-record-id={route.record_id}
                    data-current={route.record_id === current || undefined}
                    data-switch={route.session_action === 'switch' || undefined}
                    aria-current={route.record_id === current ? 'true' : undefined}
                  >
                    <td className={styles.historyNumber}>
                      <Link to={getInsightsRecordPath(route.record_id)}>{index + 1}</Link>
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
                      <span className={styles.historyAgent}>
                        <i style={{ background: agentHue(agent, agentIndex.get(agent) ?? 0) }} />
                        {agents.get(agent)}
                      </span>
                    </td>
                    <td className={styles.historyNumber}>{route.turn_index + 1}</td>
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
                    <td className={styles.historyReason} title={route.session_reason || undefined}>
                      {reasonText(route)}
                    </td>
                    <td
                      title={`${route.lifecycle_state}${route.response_status ? ` · HTTP ${route.response_status}` : ''}`}
                    >
                      <span className={styles.historyResult} data-outcome={outcome}>
                        {outcome === 'running'
                          ? 'running'
                          : route.response_status || route.lifecycle_state}
                      </span>
                    </td>
                    <td className={styles.historyNumber}>
                      {outcome === 'running' ? '…' : formatSeconds(route.duration_ms)}
                    </td>
                  </tr>
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
