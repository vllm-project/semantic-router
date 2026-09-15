import { afterEach, describe, expect, it, vi } from 'vitest'

import type { EvaluationRunEvent } from '../types/evaluationPlane'
import { subscribeToEvaluationRun } from './evaluationPlaneApi'
import { RUN_ID, run } from '../test/evaluationPlaneApiFixture'

afterEach(() => vi.unstubAllGlobals())

describe('Evaluation Plane API event stream', () => {
  it('keeps native SSE reconnect active, deduplicates event ids, and stops at terminal events', () => {
    class FakeEventSource {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSED = 2
      static instances: FakeEventSource[] = []

      readonly listeners = new Map<string, EventListener[]>()
      readonly close = vi.fn(() => {
        this.readyState = FakeEventSource.CLOSED
      })
      readyState = FakeEventSource.CONNECTING
      onmessage: ((event: MessageEvent<string>) => void) | null = null
      onerror: ((event: Event) => void) | null = null

      constructor(readonly url: string) {
        FakeEventSource.instances.push(this)
      }

      addEventListener(name: string, listener: EventListener) {
        this.listeners.set(name, [...(this.listeners.get(name) || []), listener])
      }

      emit(name: string, event: EvaluationRunEvent) {
        const message = { data: JSON.stringify(event) } as MessageEvent<string>
        this.listeners.get(name)?.forEach((listener) => listener(message))
      }

      fail(readyState: number) {
        this.readyState = readyState
        this.onerror?.({ type: 'error' } as Event)
      }
    }

    vi.stubGlobal('EventSource', FakeEventSource)
    const onEvent = vi.fn()
    const onTerminal = vi.fn()
    const onError = vi.fn()
    const unsubscribe = subscribeToEvaluationRun(run, onEvent, onTerminal, onError)
    const source = FakeEventSource.instances[0]

    expect(source?.url).toBe(`/api/evaluation/v1/runs/${RUN_ID}/events`)
    source?.fail(FakeEventSource.CONNECTING)
    expect(source?.close).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()

    const progress: EvaluationRunEvent = {
      id: '1',
      run_id: RUN_ID,
      type: 'progress',
      timestamp: '2026-08-29T00:00:00Z',
      message: 'Routing track started',
    }
    source?.emit('progress', progress)
    source?.emit('progress', progress)
    expect(onEvent).toHaveBeenCalledTimes(1)

    source?.emit('completed', {
      ...progress,
      id: '2',
      type: 'completed',
      message: 'Evaluation completed',
      progress: {
        percent: 100,
        completed: 1,
        total: 1,
        message: 'Evaluation completed',
      },
    })
    expect(onEvent).toHaveBeenCalledTimes(2)
    expect(onTerminal).toHaveBeenCalledTimes(1)
    expect(source?.close).toHaveBeenCalledTimes(1)
    source?.emit('progress', { ...progress, id: '3' })
    expect(onEvent).toHaveBeenCalledTimes(2)

    unsubscribe()
  })

  it('terminates a server-closed SSE stream instead of retrying it', () => {
    class ClosedEventSource {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSED = 2
      static instance: ClosedEventSource | null = null

      readonly close = vi.fn()
      readyState = ClosedEventSource.CONNECTING
      onmessage: ((event: MessageEvent<string>) => void) | null = null
      onerror: ((event: Event) => void) | null = null

      constructor(readonly url: string) {
        ClosedEventSource.instance = this
      }

      addEventListener() {}
    }

    vi.stubGlobal('EventSource', ClosedEventSource)
    const onError = vi.fn()
    subscribeToEvaluationRun(run, vi.fn(), vi.fn(), onError)
    const source = ClosedEventSource.instance
    if (!source) throw new Error('Expected the EventSource test double to be constructed.')

    source.readyState = ClosedEventSource.CLOSED
    source.onerror?.({ type: 'error' } as Event)

    expect(source.close).toHaveBeenCalledTimes(1)
    expect(onError).toHaveBeenCalledWith(
      new Error('Evaluation event stream was closed by the server.'),
    )
  })
})
