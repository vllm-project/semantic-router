import { describe, expect, it } from 'vitest'

import {
  agentHashText,
  compactSessionId,
  displaySessionId,
  housekeepingDetail,
  requestClassOf,
  runAgentFor,
  sessionActionLabel,
  sessionReasonText,
  spawnsSubagent,
  traceToolNames,
  type AgentRuleIndex,
} from './insightsAgentIdentity'
import {
  apiClientCall,
  claudeRun,
  claudeSpawnCall,
  claudeSubagentCall,
  codexCall,
  codexSubagentCall,
  openCodeCall,
  openCodeTitleCall,
} from './insightsAgentFixtures'
import { buildAgentRuleIndex } from './insightsConfigFacts'

const rules: AgentRuleIndex = buildAgentRuleIndex({
  recipes: [
    {
      name: 'default',
      signals: {
        agent: [
          { name: 'exploration_subagent', role: 'subagent', types: ['Explore', 'collab_spawn'] },
          { name: 'housekeeping', request_classes: ['compaction', 'auxiliary'] },
          { name: 'any_main', role: 'main', request_classes: ['main'] },
        ],
      },
    },
  ],
})

describe('request class', () => {
  it('reads main and subagent calls from the agent identity', () => {
    expect(requestClassOf(claudeSpawnCall)).toBe('main')
    expect(requestClassOf(claudeSubagentCall)).toBe('subagent')
    expect(requestClassOf(codexCall)).toBe('main')
    expect(requestClassOf(codexSubagentCall)).toBe('subagent')
    expect(requestClassOf(openCodeCall)).toBe('main')
  })

  it('keeps a redacted subagent a subagent', () => {
    expect(
      requestClassOf({ ...claudeSubagentCall, agent: { harness: 'claude_code', id: '' } }),
    ).toBe('subagent')
    expect(agentHashText('')).toBe('hidden')
    expect(agentHashText('3fa9c2d81e07ab')).toBe('#3fa9c2d8')
  })

  it("marks a harness's session title call as housekeeping", () => {
    expect(requestClassOf(openCodeTitleCall)).toBe('housekeeping')
    expect(housekeepingDetail(openCodeTitleCall)).toBe('Session title')
  })

  it('marks a compaction by the housekeeping-only agent rule it matched', () => {
    const compaction = { ...claudeSpawnCall, signals: { agent: ['housekeeping'] } }
    expect(requestClassOf(compaction)).toBe('main')
    expect(requestClassOf(compaction, rules)).toBe('housekeeping')
    expect(housekeepingDetail(compaction, rules)).toBe(
      'Compaction or side request · rule housekeeping',
    )
    // a rule that also names other classes does not make a call housekeeping
    expect(requestClassOf({ ...claudeSpawnCall, signals: { agent: ['any_main'] } }, rules)).toBe(
      'main',
    )
  })

  it('separates reviews, Router calls and plain API calls', () => {
    expect(requestClassOf({ ...codexCall, request_kind: 'codex_approval_review' })).toBe('review')
    expect(requestClassOf({ ...codexCall, parent_request_id: 'req-1' })).toBe('router')
    expect(requestClassOf(apiClientCall)).toBe('api')
  })
})

describe('agent context', () => {
  it('shows the session id the harness sent and keeps the scoped one for links', () => {
    expect(displaySessionId('key_redacted:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483')).toBe(
      '49b948ac-48b9-4fd5-b4e8-ccb8f6fba483',
    )
    expect(displaySessionId('key_3fa9c2:ses_e158')).toBe('ses_e158')
    expect(displaySessionId('cc-start-09a886187e53d62b')).toBe('cc-start-09a886187e53d62b')
    expect(compactSessionId('key_redacted:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483')).toBe(
      '49b948ac…fba483',
    )
  })

  it('lists the tools a call used and spots the call that spawned a subagent', () => {
    expect(traceToolNames(openCodeCall.tool_trace)).toEqual(['bash', 'task'])
    expect(
      traceToolNames({
        tool_names: ['Read'],
        steps: [{ type: 'assistant_tool_call', tool_name: 'Edit' }],
      }),
    ).toEqual(['Read', 'Edit'])
    expect(spawnsSubagent(claudeSpawnCall)).toBe(true)
    expect(spawnsSubagent(openCodeCall)).toBe(true)
    expect(spawnsSubagent(codexCall)).toBe(false)
  })

  it("finds a record's agent in its run by the agent id", () => {
    expect(runAgentFor(claudeRun.run, claudeSpawnCall)?.label).toBe('main')
    expect(runAgentFor(claudeRun.run, claudeSubagentCall)?.label).toBe('subagent 1')
    expect(
      runAgentFor(claudeRun.run, { agent: { harness: 'claude_code', id: '' } }),
    ).toBeUndefined()
    expect(runAgentFor(null, claudeSpawnCall)).toBeUndefined()
  })

  it('words the session routing action and reason', () => {
    expect(sessionActionLabel('establish_baseline')).toBe('Picked the session model')
    expect(sessionActionLabel('stay')).toBe('Kept the session model')
    expect(sessionActionLabel('none')).toBe('Not applied')
    expect(sessionActionLabel(undefined)).toBeUndefined()
    expect(sessionReasonText('missing_identity')).toBe('No session identity on the request')
    expect(sessionReasonText('')).toBe('Hidden from this viewer')
    expect(sessionReasonText('brand_new_reason')).toBe('brand new reason')
  })
})
