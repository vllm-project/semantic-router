import { describe, expect, it } from 'vitest'

import type { PlaygroundTask } from '../components/ChatComponentTypes'
import { normalizePlaygroundQueues, prunePlaygroundQueues } from './playgroundQueueStorage'

const task = (
  id: string,
  conversationId: string,
  createdAt: number
): PlaygroundTask => ({
  id,
  conversationId,
  prompt: `prompt ${id}`,
  createdAt,
  requestOptions: {
    enableClawMode: false,
    enableWebSearch: false,
    model: 'auto',
  },
})

describe('playgroundQueueStorage', () => {
  it('filters invalid restored queue entries', () => {
    const restored = normalizePlaygroundQueues({
      convA: [
        task('a1', 'convA', 1),
        { id: 'bad', conversationId: 'wrong', prompt: 'bad', createdAt: 2, requestOptions: {} },
      ],
      convB: 'not-an-array',
      convC: [],
    })

    expect(restored).toEqual({
      convA: [task('a1', 'convA', 1)],
    })
  })

  it('caps tasks per conversation and keeps newest queued work', () => {
    const queues = prunePlaygroundQueues({
      convA: [
        task('a1', 'convA', 1),
        task('a2', 'convA', 2),
        task('a3', 'convA', 3),
      ],
    }, { maxTasksPerConversation: 2 })

    expect(queues.convA.map(item => item.id)).toEqual(['a2', 'a3'])
  })

  it('caps conversations by newest task timestamp', () => {
    const queues = prunePlaygroundQueues({
      old: [task('old1', 'old', 1)],
      newest: [task('new1', 'newest', 3)],
      middle: [task('mid1', 'middle', 2)],
    }, { maxConversations: 2 })

    expect(Object.keys(queues)).toEqual(['newest', 'middle'])
  })
})
