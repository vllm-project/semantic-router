import { Fragment } from 'react'
import { Link } from 'react-router-dom'

import { formatSeconds } from '../utils/dateTime'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { buildConversationLabels } from './insightsConversationIdentity'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import type { InsightsTrajectory } from './insightsPageTypes'
import type { InsightsTrajectoryRoute } from './insightsPageRoutingTypes'
import styles from './InsightsPage.module.css'

// What the Router did with a request's model and why, in words; the recorded code stays in the cell's title.
const ACTION_TEXT: Record<string, string> = {
  establish_baseline: 'Picked the first model',
  stay: 'Kept the model',
  switch: 'Switched model',
  hard_lock: 'Held the model',
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

function reasonText(route: InsightsTrajectoryRoute) {
  const code = route.session_reason || ''
  if (code === 'session_scope_protect' && route.agent) return "Kept this agent's model"
  return REASON_TEXT[code] || code || route.selection_reasoning || 'Not recorded'
}

export default function InsightsSessionRoutes({
  trajectory,
}: {
  trajectory: InsightsTrajectory | null
}) {
  if (!trajectory?.routes?.length) return null
  const labels = buildConversationLabels(trajectory.routes)
  return (
    <section
      className={`${styles.recordSection} ${styles.recordSectionWide}`}
      aria-label="Session routing history"
    >
      <div className={styles.recordSectionHeader}>
        <h2>Session routing history</h2>
        <span>
          {trajectory.recipe || 'Default recipe'} · {trajectory.routes.length} requests
        </span>
      </div>
      <div className={styles.projectionTraceWrap}>
        <table className={styles.projectionTraceTable}>
          <thead>
            <tr>
              {['Turn', 'Agent', 'Decision', 'Model', 'Action', 'Reason', 'Result', 'Duration'].map(
                (heading) => (
                  <th key={heading}>{heading}</th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {trajectory.routes.map((route, index) => (
              <Fragment key={route.record_id}>
                {labels.size > 0 &&
                (index === 0 ||
                  route.conversation_id !== trajectory.routes?.[index - 1]?.conversation_id) ? (
                  <tr>
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
                <tr data-record-id={route.record_id}>
                  <td>
                    <Link to={getInsightsRecordPath(route.record_id)}>{route.turn_index + 1}</Link>
                  </td>
                  <td title={route.agent ? 'Agent identity hash' : undefined}>
                    {route.agent || 'main'}
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
                    {route.previous_model && route.previous_model !== route.selected_model
                      ? `${route.previous_model} → `
                      : ''}
                    {route.selected_model || 'No backend selected'}
                  </td>
                  <td title={route.session_action || undefined}>
                    {ACTION_TEXT[route.session_action || ''] ||
                      route.session_action ||
                      'Not recorded'}
                    {route.session_policy_applied ? ' · protected' : ''}
                  </td>
                  <td title={route.session_reason || undefined}>{reasonText(route)}</td>
                  <td>
                    {route.lifecycle_state}
                    {route.response_status ? ` · HTTP ${route.response_status}` : ''}
                  </td>
                  <td>
                    {route.lifecycle_state === 'in_progress'
                      ? 'In progress'
                      : formatSeconds(route.duration_ms)}
                  </td>
                </tr>
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
