import { afterEach, describe, expect, it, vi } from 'vitest'

import { getResults, listTasks } from './evaluationApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('evaluation API read contracts', () => {
  it('preserves the server status filter and forwards abort signals', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn(async () => new Response('[]', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await listTasks('failed', controller.signal)

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/evaluation/tasks?status=failed',
      { signal: controller.signal },
    )
  })

  it('encodes result identifiers before issuing an on-demand report request', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ task: { id: 'task one' }, results: [] }), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await getResults('task one')

    expect(fetchMock).toHaveBeenCalledWith('/api/evaluation/results/task%20one', {
      signal: undefined,
    })
  })
})
