export type OpenClawRequestResult<T> =
  | { status: 'success'; value: T }
  | { status: 'aborted' }
  | { status: 'error'; error: unknown }

interface OpenClawRequestHandlers<T> {
  onStart?: () => void
  onSuccess?: (value: T) => void
  onError?: (error: unknown) => void
  onFinish?: () => void
}

export interface LatestOpenClawRequest {
  run<T>(
    request: (signal: AbortSignal) => Promise<T>,
    handlers?: OpenClawRequestHandlers<T>,
    externalSignal?: AbortSignal,
  ): Promise<OpenClawRequestResult<T>>
  cancel(): void
  isInFlight(): boolean
}

export function isOpenClawAbortError(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && 'name' in error && error.name === 'AbortError',
  )
}

export function getOpenClawErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback
}

export function createLatestOpenClawRequest(): LatestOpenClawRequest {
  let generation = 0
  let activeController: AbortController | null = null

  return {
    async run<T>(
      request: (signal: AbortSignal) => Promise<T>,
      handlers: OpenClawRequestHandlers<T> = {},
      externalSignal?: AbortSignal,
    ): Promise<OpenClawRequestResult<T>> {
      const requestGeneration = ++generation
      activeController?.abort()
      const controller = new AbortController()
      activeController = controller
      const forwardAbort = () => controller.abort()

      if (externalSignal?.aborted) controller.abort()
      else externalSignal?.addEventListener('abort', forwardAbort, { once: true })

      handlers.onStart?.()
      try {
        const value = await request(controller.signal)
        if (requestGeneration !== generation || controller.signal.aborted) {
          return { status: 'aborted' }
        }
        handlers.onSuccess?.(value)
        return { status: 'success', value }
      } catch (error) {
        if (
          requestGeneration !== generation ||
          controller.signal.aborted ||
          isOpenClawAbortError(error)
        ) {
          return { status: 'aborted' }
        }
        handlers.onError?.(error)
        return { status: 'error', error }
      } finally {
        externalSignal?.removeEventListener('abort', forwardAbort)
        if (requestGeneration === generation) {
          activeController = null
          handlers.onFinish?.()
        }
      }
    },
    cancel() {
      generation += 1
      activeController?.abort()
      activeController = null
    },
    isInFlight() {
      return activeController !== null
    },
  }
}

function extractOpenClawError(body: string): string {
  if (!body.trim()) return ''
  try {
    const parsed: unknown = JSON.parse(body)
    if (parsed && typeof parsed === 'object') {
      const record = parsed as Record<string, unknown>
      if (typeof record.error === 'string') return record.error
      if (typeof record.message === 'string') return record.message
    }
  } catch {
    // Preserve the response body when it is not JSON.
  }
  return body.trim()
}

export async function fetchOpenClawJSON<T>(
  url: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, { ...init, signal })
  const body = await response.text()
  if (!response.ok) {
    const detail = extractOpenClawError(body)
    throw new Error(detail || `OpenClaw request failed (${response.status} ${response.statusText})`)
  }
  if (!body.trim()) return undefined as T
  return JSON.parse(body) as T
}
