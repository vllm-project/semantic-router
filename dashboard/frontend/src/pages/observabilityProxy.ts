export type ObservabilityService = 'Prometheus' | 'Jaeger' | 'Grafana'

/**
 * Why a read through the Dashboard's `/embedded/<service>/` proxy failed:
 * the service is not configured, the session may not read it, it did not
 * answer, it answered too slowly, or it rejected the query itself.
 */
export type ObservabilityFailure = 'unconfigured' | 'denied' | 'unreachable' | 'timeout' | 'query'

export class ObservabilityError extends Error {
  readonly failure: ObservabilityFailure
  readonly status?: number

  constructor(failure: ObservabilityFailure, message: string, status?: number) {
    super(message)
    this.name = 'ObservabilityError'
    this.failure = failure
    this.status = status
  }
}

/** The message of a Prometheus (`{status, error}`) or Jaeger (`{errors: [{msg}]}`) error body. */
export function observabilityErrorMessage(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null
  const { status, error, errors } = body as { status?: unknown; error?: unknown; errors?: unknown }
  if (status === 'error' && typeof error === 'string' && error.trim()) return error.trim()
  if (Array.isArray(errors)) {
    const messages = errors
      .map((entry) =>
        entry && typeof entry === 'object' ? (entry as { msg?: unknown }).msg : null,
      )
      .filter((message): message is string => typeof message === 'string' && message.trim() !== '')
    if (messages.length) return messages.join('; ')
  }
  return null
}

/** The message of the Dashboard's own `{error, message}` error body. */
function dashboardErrorMessage(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null
  const { error, message } = body as { error?: unknown; message?: unknown }
  return typeof error === 'string' && typeof message === 'string' && message.trim()
    ? message.trim()
    : null
}

export function classifyObservabilityFailure(
  service: ObservabilityService,
  status: number,
  contentType: string,
  body: unknown,
): ObservabilityError {
  if (status === 401 || status === 403) {
    return new ObservabilityError(
      'denied',
      `Access to ${service} was denied. Check your session and permissions, then try again.`,
      status,
    )
  }
  const detail = observabilityErrorMessage(body)
  if (detail) return new ObservabilityError('query', detail, status)
  // The Dashboard's own endpoints answer 503 when the service is not
  // configured, 502 when it cannot be reached or read, 504 when it is too slow.
  const explained = dashboardErrorMessage(body)
  if (explained) {
    const failure: ObservabilityFailure =
      status === 503
        ? 'unconfigured'
        : status === 502
          ? 'unreachable'
          : status === 504
            ? 'timeout'
            : 'query'
    return new ObservabilityError(failure, explained, status)
  }
  // The Dashboard answers an unconfigured service with its own HTML page.
  if (status === 503 && contentType.includes('text/html')) {
    return new ObservabilityError(
      'unconfigured',
      `${service} is not connected to the Dashboard.`,
      status,
    )
  }
  if (status === 502 || status === 503 || status === 504) {
    return new ObservabilityError(
      'unreachable',
      `${service} is unavailable (HTTP ${status}). Try again when it is ready.`,
      status,
    )
  }
  return new ObservabilityError('query', `${service} answered HTTP ${status}.`, status)
}

const NOT_JSON = Symbol('not JSON')

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return NOT_JSON
  }
}

/**
 * GETs JSON from a proxied observability API. Failures are classified as
 * {@link ObservabilityError}; a caller abort rethrows the browser's AbortError.
 * `timeoutMs` bounds the whole read, the body's download included.
 */
export async function fetchObservabilityJson(
  service: ObservabilityService,
  url: string,
  {
    signal,
    timeoutMs = 15_000,
    fetcher = fetch,
  }: { signal?: AbortSignal; timeoutMs?: number; fetcher?: typeof fetch } = {},
): Promise<unknown> {
  signal?.throwIfAborted()
  const controller = new AbortController()
  let timedOut = false
  const abort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', abort, { once: true })
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  try {
    const response = await fetcher(url, {
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    })
    const contentType = response.headers.get('Content-Type') ?? ''
    // A body that stops arriving rejects here, so the catch below reports a
    // timeout or a dropped connection rather than an answer that is not data.
    const body = contentType.includes('json') ? parseJson(await response.text()) : NOT_JSON
    if (!response.ok) {
      throw classifyObservabilityFailure(
        service,
        response.status,
        contentType,
        body === NOT_JSON ? null : body,
      )
    }
    if (body === NOT_JSON) {
      throw new ObservabilityError(
        'query',
        `${service} answered with a page instead of data. Check the ${service} address the Dashboard proxies to.`,
        response.status,
      )
    }
    return body
  } catch (error) {
    if (error instanceof ObservabilityError) throw error
    if (timedOut) {
      throw new ObservabilityError('timeout', `${service} took too long to respond. Try again.`)
    }
    if (signal?.aborted) throw error
    throw new ObservabilityError(
      'unreachable',
      `Could not reach ${service}. Check your connection and try again.`,
    )
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}
