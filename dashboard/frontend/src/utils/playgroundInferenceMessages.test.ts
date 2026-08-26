import { describe, expect, it } from 'vitest'

import {
  applyPlaygroundInferenceDelta,
  assertPlaygroundAssistantText,
  completePlaygroundInferenceMessage,
  type PlaygroundInferenceMessage,
  type PlaygroundInferenceMetadata,
} from './playgroundInferenceMessages'

describe('OpenAI Playground message projection', () => {
  it('reveals each stream delta before attaching final headers, usage, and timing', () => {
    const initial: PlaygroundInferenceMessage[] = [
      {
        id: 'assistant-1',
        role: 'assistant',
        content: '',
        createdAt: '2026-08-25T00:00:00Z',
        status: 'streaming',
      },
    ]

    const first = applyPlaygroundInferenceDelta(initial, 'assistant-1', 'Hel')
    expect(first[0]).toMatchObject({ content: 'Hel', status: 'streaming' })
    expect(first[0].metadata).toBeUndefined()

    const second = applyPlaygroundInferenceDelta(first, 'assistant-1', 'lo')
    expect(second[0]).toMatchObject({ content: 'Hello', status: 'streaming' })
    expect(second[0].metadata).toBeUndefined()

    const metadata: PlaygroundInferenceMetadata = {
      headers: { 'x-vsr-selected-model': 'local/model' },
      latencyMilliseconds: 120,
      ttftMilliseconds: 30,
      usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
    }
    const complete = completePlaygroundInferenceMessage(second, 'assistant-1', metadata)

    expect(complete[0]).toMatchObject({ content: 'Hello', status: 'complete', metadata })
  })

  it('does not accept an empty or tool-only stream as a successful assistant message', () => {
    expect(() => assertPlaygroundAssistantText('')).toThrow(
      'Router completed the stream without assistant text.',
    )
    expect(() => assertPlaygroundAssistantText('  \n')).toThrow()
    expect(() => assertPlaygroundAssistantText('Done')).not.toThrow()
  })
})
