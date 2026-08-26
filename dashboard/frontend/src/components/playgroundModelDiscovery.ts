export const PLAYGROUND_MODEL_DISCOVERY_TIMEOUT_MS = 8_000

export class PlaygroundModelDiscoveryTimeoutError extends Error {
  constructor() {
    super('Model discovery timed out. Try again.')
    this.name = 'PlaygroundModelDiscoveryTimeoutError'
  }
}

export async function fetchPlaygroundModelPayload(
  endpoint: string,
  signal: AbortSignal,
  getAccessToken: () => Promise<string>,
  timeoutMilliseconds = PLAYGROUND_MODEL_DISCOVERY_TIMEOUT_MS,
): Promise<unknown> {
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError')

  const requestController = new AbortController()
  const abortFromParent = () => requestController.abort(signal.reason)
  signal.addEventListener('abort', abortFromParent, { once: true })

  let rejectOnAbort: (() => void) | undefined
  const aborted = new Promise<never>((_, reject) => {
    rejectOnAbort = () =>
      reject(requestController.signal.reason ?? new DOMException('Aborted', 'AbortError'))
    requestController.signal.addEventListener('abort', rejectOnAbort, { once: true })
  })
  const timeout = globalThis.setTimeout(
    () => requestController.abort(new DOMException('Timed out', 'TimeoutError')),
    Math.max(1, timeoutMilliseconds),
  )

  try {
    const response = await Promise.race([
      (async () => {
        const accessToken = await getAccessToken()
        if (requestController.signal.aborted) {
          throw requestController.signal.reason ?? new DOMException('Aborted', 'AbortError')
        }
        return fetch(endpoint, {
          cache: 'no-store',
          credentials: 'omit',
          headers: {
            Accept: 'application/json',
            Authorization: `Bearer ${accessToken}`,
          },
          signal: requestController.signal,
        })
      })(),
      aborted,
    ])
    if (!response.ok) {
      throw new Error(`Model discovery failed with status ${response.status}`)
    }
    const payload = await (response.json() as Promise<unknown>)
    return payload
  } catch (error) {
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError')
    if (requestController.signal.aborted) throw new PlaygroundModelDiscoveryTimeoutError()
    throw error
  } finally {
    globalThis.clearTimeout(timeout)
    signal.removeEventListener('abort', abortFromParent)
    if (rejectOnAbort) requestController.signal.removeEventListener('abort', rejectOnAbort)
  }
}
