interface LatestRequestHandlers<T> {
  onStart?: () => void
  onSuccess?: (value: T) => void
  onError?: (error: unknown) => void
  onFinish?: () => void
}

export interface LatestMCPRequestRunner {
  run<T>(
    request: (signal: AbortSignal) => Promise<T>,
    handlers?: LatestRequestHandlers<T>,
    externalSignal?: AbortSignal,
  ): Promise<T | undefined>
  cancel(): void
  isInFlight(): boolean
}

export function isMCPAbortError(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && 'name' in error && error.name === 'AbortError',
  )
}

export function getMCPRequestErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback
}

export function createLatestMCPRequestRunner(): LatestMCPRequestRunner {
  let generation = 0
  let activeController: AbortController | null = null

  return {
    async run<T>(
      request: (signal: AbortSignal) => Promise<T>,
      handlers: LatestRequestHandlers<T> = {},
      externalSignal?: AbortSignal,
    ): Promise<T | undefined> {
      const requestGeneration = ++generation
      activeController?.abort()
      const controller = new AbortController()
      activeController = controller
      const forwardAbort = () => controller.abort()

      if (externalSignal?.aborted) {
        controller.abort()
      } else {
        externalSignal?.addEventListener('abort', forwardAbort, { once: true })
      }

      handlers.onStart?.()
      try {
        const value = await request(controller.signal)
        if (requestGeneration !== generation || controller.signal.aborted) return undefined
        handlers.onSuccess?.(value)
        return value
      } catch (error) {
        if (
          requestGeneration === generation &&
          !controller.signal.aborted &&
          !isMCPAbortError(error)
        ) {
          handlers.onError?.(error)
        }
        return undefined
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
