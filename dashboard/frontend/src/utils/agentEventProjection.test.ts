import { describe, expect, it } from 'vitest'

import type { AgentEvent, AgentLiveModelStepEvent } from '../generated/managementApiContract'
import {
  activeAgentTurnId,
  agentTurnIsTerminal,
  pendingApproval,
  projectAgentTimeline,
} from './agentEventProjection'

const base = {
  sessionId: 'session-1',
  turnId: 'turn-1',
  createdAt: '2026-08-23T00:00:00Z',
}

describe('Agent event projection', () => {
  it('projects streamed text and one completed tool row from durable events', () => {
    const events: AgentEvent[] = [
      {
        ...base,
        sequence: 1,
        type: 'user_input',
        payload: { content: [{ type: 'text', text: 'Build it' }] },
      },
      {
        ...base,
        sequence: 2,
        type: 'assistant_delta',
        payload: {
          modelStepId: 'step-1',
          chunkIndex: 0,
          delta: { kind: 'text', text: 'Ready ' },
        },
      },
      {
        ...base,
        sequence: 3,
        type: 'tool_request',
        payload: {
          invocationId: 'invocation-1',
          toolName: 'routing.validate_recipe',
          arguments: { recipeId: 'blend' },
          class: 'read',
        },
      },
      {
        ...base,
        sequence: 4,
        type: 'tool_result',
        payload: {
          invocationId: 'invocation-1',
          toolName: 'routing.validate_recipe',
          status: 'completed',
          result: { recipeId: 'blend' },
        },
      },
      {
        ...base,
        sequence: 5,
        type: 'assistant_delta',
        payload: {
          modelStepId: 'step-2',
          chunkIndex: 0,
          delta: { kind: 'text', text: 'to review.' },
        },
      },
      { ...base, sequence: 6, type: 'terminal', payload: { status: 'completed' } },
    ]

    const timeline = projectAgentTimeline(events)
    expect(timeline).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'message',
          role: 'assistant',
          text: 'Ready ',
          streaming: false,
        }),
        expect.objectContaining({
          kind: 'message',
          role: 'assistant',
          text: 'to review.',
          streaming: false,
        }),
        expect.objectContaining({
          kind: 'tool',
          invocationId: 'invocation-1',
          status: 'completed',
          classification: 'read',
        }),
      ]),
    )
    expect(activeAgentTurnId(events)).toBeNull()
    expect(agentTurnIsTerminal(events, 'turn-1')).toBe(true)
    expect(agentTurnIsTerminal(events, 'turn-2')).toBe(false)
  })

  it('keeps an immutable review pending until an approval result arrives', () => {
    const approval = {
      planId: 'plan-1',
      planDigest: `sha256:${'a'.repeat(64)}`,
      planRevision: 3,
      planEtag: '"plan-3"',
      expiresAt: '2099-08-23T00:00:00Z',
      summary: { entrypointName: 'blend-v2' },
    }
    const request: AgentEvent = {
      ...base,
      sequence: 1,
      type: 'approval_request',
      payload: approval,
    }

    expect(pendingApproval([request])).toEqual(approval)
    expect(
      pendingApproval([
        request,
        {
          ...base,
          sequence: 2,
          type: 'approval_result',
          payload: { planId: 'plan-1', status: 'committed' },
        },
      ]),
    ).toBeNull()
  })

  it('shows provisional text immediately and removes it once durable text exists', () => {
    const preview: AgentLiveModelStepEvent = {
      ...base,
      modelStepId: 'step-live',
      phase: 'delta',
      ordinal: 1,
      delta: { kind: 'text', text: 'Thinking now' },
    }
    expect(projectAgentTimeline([], [preview])).toEqual([
      expect.objectContaining({
        id: 'assistant-preview-step-live',
        text: 'Thinking now',
        streaming: true,
      }),
    ])

    const durable: AgentEvent<'assistant_delta'> = {
      ...base,
      sequence: 1,
      type: 'assistant_delta',
      payload: {
        modelStepId: 'step-live',
        chunkIndex: 0,
        delta: { kind: 'text', text: 'Thinking now' },
      },
    }
    const timeline = projectAgentTimeline([durable], [preview])
    expect(timeline).toHaveLength(1)
    expect(timeline[0]).toEqual(
      expect.objectContaining({
        id: 'assistant-step-live',
        text: 'Thinking now',
        streaming: false,
      }),
    )
  })

  it('attaches the durable Router summary to the matching assistant response', () => {
    const modelStepId = '11111111-1111-4111-8111-111111111111'
    const events: AgentEvent[] = [
      {
        ...base,
        sequence: 1,
        type: 'assistant_delta',
        payload: {
          modelStepId,
          chunkIndex: 0,
          delta: { kind: 'text', text: 'Done.' },
        },
      },
      {
        ...base,
        sequence: 2,
        type: 'model_step_summary',
        payload: {
          modelStepId,
          requestId: 'request-42',
          selectedRecipe: 'balance',
          selectedDecision: 'Complex',
          selectedModel: 'remote/frontier',
          selectedAlgorithm: 'static',
          responsePath: 'upstream',
          latencyMilliseconds: 420,
          ttftMilliseconds: 84,
          usage: { inputTokens: 120, outputTokens: 48, totalTokens: 168 },
        },
      },
    ]

    expect(projectAgentTimeline(events)).toEqual([
      expect.objectContaining({
        id: `assistant-${modelStepId}`,
        text: 'Done.',
        metadata: expect.objectContaining({
          requestId: 'request-42',
          selectedModel: 'remote/frontier',
          latencyMilliseconds: 420,
          usage: { inputTokens: 120, outputTokens: 48, totalTokens: 168 },
        }),
      }),
    ])
  })

  it('keeps a summary visible on the latest page and reconciles earlier text on replay', () => {
    const modelStepId = '22222222-2222-4222-8222-222222222222'
    const summary: AgentEvent<'model_step_summary'> = {
      ...base,
      sequence: 2,
      type: 'model_step_summary',
      payload: {
        modelStepId,
        requestId: 'request-43',
        selectedModel: 'local/fast',
        latencyMilliseconds: 18,
      },
    }
    const latest = projectAgentTimeline([summary])
    expect(latest).toEqual([
      expect.objectContaining({
        text: '',
        metadata: expect.objectContaining({ requestId: 'request-43' }),
      }),
    ])

    const earlier: AgentEvent<'assistant_delta'> = {
      ...base,
      sequence: 1,
      type: 'assistant_delta',
      payload: {
        modelStepId,
        chunkIndex: 0,
        delta: { kind: 'text', text: 'Recovered.' },
      },
    }
    expect(projectAgentTimeline([summary, earlier])).toEqual([
      expect.objectContaining({
        text: 'Recovered.',
        metadata: expect.objectContaining({ requestId: 'request-43' }),
      }),
    ])
  })
})
