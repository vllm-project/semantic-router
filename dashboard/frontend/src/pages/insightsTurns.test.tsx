import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { InsightsTurnFilter } from './InsightsPage'
import InsightsRecordSection from './InsightsRecordSection'
import InsightsSessionRoutes from './InsightsSessionRoutes'
import { buildReplayQueryString } from './insightsPageFilters'
import type { InsightsTrajectoryRoute } from './insightsPageRoutingTypes'
import { buildInsightsRecordSections } from './insightsPageSupport'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'
import { assignReviewsToRoutes } from './insightsReviewSupport'

const TURN = '0199a1b2-0000-4000-8000-00000000000a'
const NEXT_TURN = '0199a1b2-0000-4000-8000-00000000000b'

const render = (node: React.ReactNode, entry = '/insights') =>
  renderToStaticMarkup(<MemoryRouter initialEntries={[entry]}>{node}</MemoryRouter>)

function record(fields: Partial<InsightsRecord>): InsightsRecord {
  return {
    id: 'record',
    timestamp: '2026-11-15T09:00:00Z',
    session_id: 'codex-session',
    turn_index: 0,
    decision_tier: 0,
    decision_priority: 0,
    signals: {},
    ...fields,
  }
}

function route(
  record_id: string,
  timestamp: string,
  fields: Partial<InsightsTrajectoryRoute> = {},
): InsightsTrajectoryRoute {
  return {
    record_id,
    timestamp,
    turn_index: 0,
    session_policy_applied: false,
    lifecycle_state: 'completed',
    response_status: 200,
    duration_ms: 500,
    selected_model: 'gpt-6-luna',
    ...fields,
  }
}

function sessionRouting(value: InsightsRecord) {
  const sections = buildInsightsRecordSections(value, { isReadonly: true })
  const index = sections.findIndex((section) => section.title === 'Session Routing')
  return index < 0
    ? ''
    : render(<InsightsRecordSection section={sections[index]} sectionIndex={index} />)
}

// A main call and a subagent call of one turn, then the next turn.
const trajectory: InsightsTrajectory = {
  object: 'router_replay.trajectory',
  session_id: 'codex-session',
  record_count: 3,
  turn_count: 2,
  messages: [],
  routes: [
    route('main-a', '2026-11-15T09:00:00Z', { turn_id: TURN }),
    route('subagent-a', '2026-11-15T09:00:10Z', {
      agent: '9f2c1a',
      turn_id: 'child-turn',
      parent_turn_id: TURN,
    }),
    route('main-b', '2026-11-15T09:01:00Z', { turn_id: NEXT_TURN, turn_index: 1 }),
  ],
}

// A review of the first turn that started after the next turn did.
const review = record({
  id: 'review',
  timestamp: '2026-11-15T09:01:30Z',
  request_kind: 'codex_approval_review',
  turn_id: 'reviewer-turn',
  parent_turn_id: TURN,
  review: { schema_version: 1, identification_source: 'entrypoint', status: 'completed' },
})

describe('Codex turns in Insights', () => {
  it("shows a record's thread and turns, each turn opening the list of that turn", () => {
    const html = sessionRouting(
      record({ thread_id: 'reviewer-thread', turn_id: 'reviewer-turn', parent_turn_id: TURN }),
    )
    expect(html).toContain('Codex thread</span><div>reviewer-thread')
    expect(html).toMatch(
      /Codex turn<\/span><div><a href="\/insights\?turn=reviewer-turn"[^>]*>reviewer-turn</,
    )
    expect(html).toMatch(
      new RegExp(`Parent turn</span><div><a href="/insights\\?turn=${TURN}"[^>]*>${TURN}<`),
    )

    const plain = sessionRouting(record({}))
    expect(plain).toContain('Session')
    expect(plain).not.toContain('Codex')
    // A Codex call outside any session still shows where it belongs.
    expect(sessionRouting(record({ session_id: undefined, turn_id: TURN }))).toContain('Codex turn')
  })

  it('reads the turn from the URL and shows it as a filter that can be cleared', () => {
    const html = render(<InsightsTurnFilter />, `/insights?turn=${TURN}`)
    expect(html).toContain(`title="${TURN}"`)
    expect(html).toContain('Codex turn 0199a1b2…0000000a')
    expect(html).toContain('aria-label="Clear turn filter"')
    expect(render(<InsightsTurnFilter />)).toBe('')
  })

  it('asks the replay API for the turn', () => {
    const filters = {
      searchTerm: '',
      filter: 'all' as const,
      recipeFilter: 'all',
      decisionFilter: 'all',
      modelFilter: 'all',
      turn: '',
    }
    expect(buildReplayQueryString(filters)).toBe('')
    const pagination = { limit: 25, offset: 25 }
    const query = buildReplayQueryString(
      { ...filters, modelFilter: 'gpt-6-luna', turn: TURN },
      pagination,
    )
    const params = new URLSearchParams(query)
    expect(Object.fromEntries(params)).toEqual({
      limit: '25',
      offset: '25',
      showDetails: 'false',
      model: 'gpt-6-luna',
      turn_id: TURN,
    })
  })

  it('marks a review on the turn it names, not the latest turn before it', () => {
    const marks = assignReviewsToRoutes(trajectory.routes!, [review])
    expect(marks.get('main-a')?.map((one) => one.id)).toEqual(['review'])
    expect(marks.has('main-b')).toBe(false)
    expect(
      assignReviewsToRoutes(trajectory.routes!, [{ ...review, parent_turn_id: undefined }]).has(
        'main-b',
      ),
    ).toBe(true)
  })

  it("groups a session's calls by Codex turn, subagents with the turn that spawned them", () => {
    const html = render(<InsightsSessionRoutes trajectory={trajectory} reviews={[review]} />)
    expect(html.match(new RegExp(`href="/insights\\?turn=${TURN}"`, 'g'))).toHaveLength(1)
    expect(html).toContain('Codex turn 0199a1b2…0000000a</a> · 2 calls')
    expect(html).toContain('Codex turn 0199a1b2…0000000b</a> · 1 call')
    // The subagent's call and the review sit under the first turn, before the next one opens.
    const nextTurn = html.indexOf(`href="/insights?turn=${NEXT_TURN}"`)
    for (const mark of ['data-record-id="subagent-a"', 'href="/insights/review"']) {
      expect(html.indexOf(mark)).toBeGreaterThan(-1)
      expect(html.indexOf(mark)).toBeLessThan(nextTurn)
    }

    const legacy = {
      ...trajectory,
      routes: trajectory.routes!.map(
        ({ turn_id: _turn, parent_turn_id: _parent, ...rest }) => rest,
      ),
    }
    expect(render(<InsightsSessionRoutes trajectory={legacy} />)).not.toContain('Codex turn')
  })
})
