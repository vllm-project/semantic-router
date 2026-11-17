import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { ModelLabelsContext } from './insightsModelLabelSupport'
import { RunCallDetails, RunLegend, RunRowDetails } from './InsightsRunDiagram'
import { RunSequence } from './InsightsRunSequence'
import { buildRunSequence } from './insightsRunSequence'
import type { RunCall, RunResponse } from './insightsRunTypes'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}
const START = Date.parse('2026-11-16T09:00:00Z')

function call(id: string, agent: string, second: number, fields: Partial<RunCall> = {}): RunCall {
  return {
    record_id: id,
    agent,
    model: 'qwen-msg',
    decision: 'default',
    timestamp: new Date(START + second * 1000).toISOString(),
    duration_ms: 1000,
    lifecycle_state: 'completed',
    ...fields,
  }
}

const failed = call('rec-fail', 'subagent 1', 5, {
  model: 'qwen-chat',
  lifecycle_state: 'failed',
  response_status: 502,
  decision: 'remember',
  signals: ['keyword:email'],
  effort: 'high',
  reason: '',
  markers: ['failure'],
})

// A main agent's three turns, a title and a quota check, a subagent that
// fails, a minute idle and a last priced turn at the catalog's list price.
const response: RunResponse = {
  object: 'router_replay.run',
  run: {
    session_id: 'run-1',
    started_at: new Date(START).toISOString(),
    ended_at: new Date(START + 80_000).toISOString(),
    duration_ms: 80_000,
    calls: 8,
    agents: [
      {
        label: 'main',
        calls: 6,
        first_call_at: '',
        last_call_at: '',
        cost,
        models: ['qwen-msg'],
        offered_tools: ['Bash', 'Read'],
      },
      {
        label: 'subagent 1',
        parent: 'main',
        kind: 'collab_spawn',
        calls: 1,
        first_call_at: '',
        last_call_at: '',
        spawned_after_record_id: 'rec-3',
        task: 'Explore · Find callers',
        returned_to_record_id: 'rec-4',
        return_status: 'failed',
        cost,
        models: ['qwen-chat'],
      },
    ],
    models: [
      { name: 'qwen-msg', value: 4 },
      { name: 'qwen-chat', value: 1 },
    ],
    outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
    cost,
  },
  calls: [
    call('rec-1', 'main', 0, {
      tool_calls: [{ name: 'Read', summary: 'calc.py', status: 'succeeded' }],
    }),
    call('rec-title', 'main', 1, { model: 'qwen-chat', request_kind: 'session_title' }),
    call('rec-2', 'main', 2, {
      tool_calls: [
        { name: 'Edit', summary: 'calc.py', status: 'succeeded' },
        { name: 'Bash', status: 'failed' },
      ],
    }),
    call('rec-quota', 'main', 2.5, { model: 'qwen-chat', request_kind: 'auxiliary' }),
    call('rec-3', 'main', 4, {
      tool_calls: [
        {
          name: 'Agent',
          summary: 'Explore · Find callers',
          arguments: '{"subagent_type":"Explore","description":"Find callers"}',
          status: 'failed',
        },
        { name: 'Bash', summary: 'pytest -q', arguments: '{"command":"pytest -q"}' },
      ],
    }),
    failed,
    call('rec-4', 'main', 70, {
      tool_calls: [
        { name: 'Read', summary: 'calc.py', status: 'succeeded' },
        {
          name: 'Bash',
          summary: 'pytest -q',
          status: 'failed',
          timing_source: 'gateway',
          client_ms: 13000,
        },
      ],
      offered_tools_added: ['mcp__docs__search'],
      cost: 0.012,
      currency: 'USD',
      pricing_source: 'catalog_list_price',
      markers: ['switch'],
    }),
  ],
}

function render(props: { expanded?: string; call?: RunCall; narrow?: boolean } = {}) {
  return renderToStaticMarkup(
    <MemoryRouter>
      <RunSequence
        run={response.run}
        sequence={buildRunSequence(response)}
        onToggle={() => undefined}
        onSelectCall={() => undefined}
        {...props}
      />
    </MemoryRouter>,
  )
}

describe('run sequence', () => {
  it('heads a lifeline per agent with the models it used, and a side column', () => {
    const html = render()
    expect(html).toContain('<strong>main</strong>')
    expect(html).toContain('<strong>↳ subagent 1</strong>')
    expect(html).toContain('collab_spawn · 1 call · no price')
    expect(html).toContain('<strong>side</strong><span>2 side requests</span>')
    expect(html.match(/class="[^"]*lifeline/g)?.length).toBeGreaterThan(0)
  })

  it("collapses an agent's consecutive turns into one row with the model, count and time", () => {
    const html = render()
    // rec-3 spawned the subagent, so it ends the main agent's first row.
    expect(html).toContain('3 calls · 5.0s')
    expect(html).toContain('<span>default</span>')
    // The last turn: its price is the catalog's, and it switched model.
    expect(html).toContain('default · $0.012 list price')
    expect(html).toContain('data-marker="switch"')
  })

  it('leaves the side column out of a run without side requests', () => {
    const quiet = { ...response, calls: response.calls.filter((entry) => !entry.request_kind) }
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <RunSequence
          run={response.run}
          sequence={buildRunSequence(quiet)}
          onToggle={() => undefined}
          onSelectCall={() => undefined}
        />
      </MemoryRouter>,
    )
    expect(html).toContain('data-sides="false"')
    expect(html).toContain('--sequence-columns:repeat(2, minmax(0, 1fr))"')
  })

  it('puts side requests in the side column, never on the main lifeline', () => {
    const html = render()
    expect(html).toContain('2 side requests</span>')
    expect(html).toContain('session_title, auxiliary')
    expect(html).toMatch(/data-side="true" aria-expanded="false" style="grid-column:3"/)
  })

  it('draws a solid spawn arrow right and a dashed return arrow left, with an idle divider', () => {
    const html = render()
    expect(html).toContain(
      'data-kind="spawn" data-direction="right" data-failed="false" data-call="rec-3" title="main spawned subagent 1: Explore · Find callers"',
    )
    expect(html).toContain(
      'data-kind="return" data-direction="left" data-failed="true" data-call="rec-4" title="subagent 1 returned to main: failed · 1 call · no price"',
    )
    expect(html).toContain('grid-column:1 / 3;--span:2;--arrow:#7aa7e8')
    expect(html).toContain('<span>Explore · Find callers</span>')
    expect(html.indexOf('data-kind="return"')).toBeLessThan(html.indexOf('$0.012 list price'))
    expect(html).toContain('· 1m 4s idle')
  })

  it("shows a turn's tool calls as chips and a collapsed row's as a count", () => {
    const html = render()
    // The first row collapses three calls, the last spawning the subagent.
    expect(html).toContain('data-failed="true">5 tool calls, 2 failed')
    // a line each, with the time the gateway saw it take and its result
    expect(html).toMatch(/data-status="succeeded" title="Read: ok">Read · calc.py · ok</)
    expect(html).toMatch(
      /data-status="failed" title="Bash: failed\. Estimated by the gateway[^"]*">Bash · pytest -q · ≈13\.0s · failed</,
    )
  })

  it("shows a tool's own run and its approval wait apart where a harness plugin reported it", () => {
    const at = (second: number) => new Date(START + second * 1000).toISOString()
    const plugin = { timing_source: 'plugin', status: 'succeeded' } as const
    const turn = call('rec-p', 'main', 0, {
      tool_calls: [
        // Claude Code: started after the person approved it, 5 s after the ask
        {
          ...plugin,
          name: 'Bash',
          summary: 'touch a.txt',
          approval_requested_at: at(1),
          started_at: at(6),
          ended_at: at(6.4),
          approval_ms: 5000,
        },
        // two reads of one response, side by side, each with its own time
        { ...plugin, name: 'Read', summary: 'a.py', started_at: at(1.1), ended_at: at(1.3) },
        { ...plugin, name: 'Read', summary: 'b.py', started_at: at(1.1), ended_at: at(2.6) },
        // Codex: its hooks do not say when the person answered
        {
          ...plugin,
          name: 'Bash',
          summary: 'git push',
          approval_requested_at: at(1.2),
          started_at: at(1),
          ended_at: at(9),
        },
        {
          ...plugin,
          name: 'Write',
          summary: 'hosts',
          status: 'denied',
          approval_requested_at: at(1),
          ended_at: at(3),
        },
      ],
    })
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <RunRowDetails calls={[turn]} onSelectCall={() => undefined} />
      </MemoryRouter>,
    )
    expect(html).toContain(
      '<code>touch a.txt</code><span title="As the harness&#x27;s vSR plugin reported it: the tool&#x27;s own run, and its approval wait apart" data-timing="plugin">ran 400ms, waited 5.0s for approval</span><span>ok</span>',
    )
    expect(html).toContain(
      '<code>a.py</code><span title="As the harness&#x27;s vSR plugin reported it: the tool&#x27;s own run, and its approval wait apart" data-timing="plugin">ran 200ms</span>',
    )
    expect(html).toContain('data-timing="plugin">ran 1.5s</span>')
    expect(html).toContain('data-timing="plugin">ran 8.0s, approval wait included</span>')
    expect(html).toContain('data-timing="plugin">asked for approval</span><span>refused</span>')
    expect(html).not.toContain('≈')
  })

  it('lists the tools an agent was offered and marks where that changed', () => {
    const html = render()
    expect(html).toContain('<summary>2 tools offered</summary><span>Bash, Read</span>')
    expect(html).toContain('title="mcp__docs__search">+1 tool offered</span>')
  })

  it("opens a row's calls and the picked call's details under it", () => {
    const sequence = buildRunSequence(response)
    const row = sequence.rows[0]
    const html = render({ expanded: row.key, call: response.calls[2] })
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('aria-label="Step details"')
    expect(html).toContain('data-selected="true"')
    expect(html).toContain('Open the replay record')
    // The whole tool list of the row, with what was kept of the arguments.
    expect(html).toContain('aria-label="Tool calls"')
    expect(html).toContain('<strong>Edit</strong><code>calc.py</code><span></span><span>ok</span>')
    expect(html).toContain(
      '<strong>Bash</strong><code>arguments not kept</code><span></span><span>failed</span>',
    )
    // The details sit in the row, under its card.
    expect(html.indexOf('aria-label="Step details"')).toBeGreaterThan(
      html.indexOf('aria-expanded="true"'),
    )
  })

  it('turns the lifelines into one indented list with agent tags when narrow', () => {
    const html = render({ narrow: true })
    expect(html).toContain('data-narrow="true"')
    expect(html).not.toMatch(/class="[^"]*lifeline/)
    expect(html).toContain('subagent 1</span>')
    expect(html).toContain('side · main</span>')
    expect(html).toContain('↳ main spawns subagent 1: Explore · Find callers')
    expect(html).toContain('↩ subagent 1 failed · 1 call · no price')
    expect(html).toContain('--depth:1')
  })
})

describe('run call details', () => {
  it('details a step with its route, a hidden reason, and its record', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <RunCallDetails call={failed} />
      </MemoryRouter>,
    )
    expect(html).toContain('keyword:email')
    expect(html).toContain('high')
    expect(html).toContain('failed 502')
    expect(html).toContain('Hidden from this viewer')
    expect(html).toContain('href="/insights/rec-fail"')
  })

  it('shows each model by its label and keeps the configured name beside it', () => {
    const labels = { 'qwen-msg': 'Qwen3.6 35B A3B', 'qwen-chat': 'qwen36' }
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <ModelLabelsContext.Provider value={labels}>
          <RunLegend run={response.run} />
          <RunSequence
            run={response.run}
            sequence={buildRunSequence(response)}
            onToggle={() => undefined}
            onSelectCall={() => undefined}
          />
          <RunRowDetails calls={[failed]} call={failed} onSelectCall={() => undefined} />
        </ModelLabelsContext.Provider>
      </MemoryRouter>,
    )
    expect(html).toContain('Qwen3.6 35B A3B<small>qwen-msg</small>')
    expect(html).toContain('title="Qwen3.6 35B A3B (qwen-msg)">Qwen3.6 35B A3B</span>')
    expect(html).toContain('qwen36<small>qwen-chat</small>')
  })
})
