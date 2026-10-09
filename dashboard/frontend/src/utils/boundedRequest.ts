// Keep the deadline active through body decoding, and distinguish a request
// timeout from a caller leaving the page or changing its authenticated session.
export async function withRequestTimeout<T>(
  load: (signal: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
  timeoutMs = 15_000,
): Promise<T> {
  signal?.throwIfAborted()
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', abort, { once: true })
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await load(controller.signal)
  } catch (error) {
    signal?.throwIfAborted()
    if (controller.signal.aborted) throw new Error('The request timed out. Please retry.')
    throw error
  } finally {
    clearTimeout(timeout)
    signal?.removeEventListener('abort', abort)
  }
}
