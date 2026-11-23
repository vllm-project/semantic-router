import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createMonacoLoad, isMonacoCancellation, nextMonacoLoadState } from './editorMonacoLoad'

const abandoned = { type: 'cancelation', msg: 'operation is manually canceled' }
const chunkError = new TypeError('Failed to fetch dynamically imported module')
const workerError = { type: 'error' }

describe('nextMonacoLoadState', () => {
  it('starts once and becomes ready when the bundle loads', () => {
    expect(nextMonacoLoadState('idle', { type: 'start' })).toBe('loading')
    expect(nextMonacoLoadState('loading', { type: 'start' })).toBe('loading')
    expect(nextMonacoLoadState('loading', { type: 'loaded' })).toBe('ready')
    expect(nextMonacoLoadState('ready', { type: 'start' })).toBe('ready')
  })

  it('falls back on a chunk, loader or worker failure, before or after the editor is ready', () => {
    expect(nextMonacoLoadState('loading', { type: 'error', error: chunkError })).toBe('failed')
    expect(nextMonacoLoadState('ready', { type: 'error', error: workerError })).toBe('failed')
    expect(nextMonacoLoadState('ready', { type: 'error', error: undefined })).toBe('failed')
  })

  it('does not treat a cancelled load as a failure', () => {
    expect(isMonacoCancellation(abandoned)).toBe(true)
    expect(isMonacoCancellation(chunkError)).toBe(false)
    expect(isMonacoCancellation(null)).toBe(false)
    expect(nextMonacoLoadState('loading', { type: 'error', error: abandoned })).toBe('loading')
  })

  it('keeps the fallback once it has failed', () => {
    expect(nextMonacoLoadState('failed', { type: 'loaded' })).toBe('failed')
    expect(nextMonacoLoadState('failed', { type: 'start' })).toBe('failed')
  })
})

describe('createMonacoLoad', () => {
  let warn: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  })
  afterEach(() => {
    warn.mockRestore()
  })

  it('loads once for every editor and notifies on each change', async () => {
    const load = vi.fn(async () => undefined)
    const monaco = createMonacoLoad(load)
    const listener = vi.fn()
    monaco.subscribe(listener)

    expect(monaco.getState()).toBe('idle')
    const first = monaco.start()
    const second = monaco.start()
    expect(second).toBe(first)
    expect(monaco.getState()).toBe('loading')
    await first

    expect(load).toHaveBeenCalledTimes(1)
    expect(monaco.getState()).toBe('ready')
    expect(listener).toHaveBeenCalledTimes(2)
    expect(warn).not.toHaveBeenCalled()
  })

  it('fails when the bundle cannot be imported or throws while installing', async () => {
    const rejected = createMonacoLoad(() => Promise.reject(chunkError))
    await rejected.start()
    expect(rejected.getState()).toBe('failed')

    const thrown = createMonacoLoad(() => {
      throw new Error('loader.config failed')
    })
    await thrown.start()
    expect(thrown.getState()).toBe('failed')
    expect(warn).toHaveBeenCalledTimes(2)
  })

  it('fails when the worker reports an error, during or after the load', async () => {
    const during = createMonacoLoad(async (fail) => fail(workerError))
    await during.start()
    expect(during.getState()).toBe('failed')

    let reportWorkerError: (error: unknown) => void = () => undefined
    const after = createMonacoLoad(async (fail) => {
      reportWorkerError = fail
    })
    await after.start()
    expect(after.getState()).toBe('ready')
    reportWorkerError(workerError)
    expect(after.getState()).toBe('failed')
  })

  it('fails when an editor throws while mounting, and stops notifying an unsubscribed view', async () => {
    const monaco = createMonacoLoad(async () => undefined)
    const listener = vi.fn()
    const unsubscribe = monaco.subscribe(listener)
    await monaco.start()
    unsubscribe()
    monaco.fail(new Error('editor.create failed'))
    expect(monaco.getState()).toBe('failed')
    expect(listener).toHaveBeenCalledTimes(2)
  })
})
