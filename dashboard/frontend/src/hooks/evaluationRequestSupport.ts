export interface EvaluationRequestOptions {
  allowHidden?: boolean
}

export interface EvaluationRequestController<T> {
  run: (options?: EvaluationRequestOptions) => Promise<T | undefined>
  invalidate: () => void
  isInFlight: () => boolean
}

interface EvaluationRequestDependencies {
  isHidden?: () => boolean
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function createEvaluationRequest<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  dependencies: EvaluationRequestDependencies = {},
): EvaluationRequestController<T> {
  const isHidden = dependencies.isHidden ?? (() => document.hidden)
  let generation = 0
  let inFlight: Promise<T | undefined> | null = null
  let abortController: AbortController | null = null

  return {
    run(options = {}) {
      if (!options.allowHidden && isHidden()) return Promise.resolve(undefined)
      if (inFlight) return inFlight

      const requestGeneration = ++generation
      const controller = new AbortController()
      abortController = controller

      const trackedRequest = fetcher(controller.signal)
        .then((value) => (requestGeneration === generation ? value : undefined))
        .catch((error: unknown) => {
          if (requestGeneration !== generation || isAbortError(error)) return undefined
          throw error
        })
        .finally(() => {
          if (inFlight === trackedRequest) inFlight = null
          if (abortController === controller) abortController = null
        })

      inFlight = trackedRequest
      return trackedRequest
    },
    invalidate() {
      generation += 1
      abortController?.abort()
      abortController = null
      inFlight = null
    },
    isInFlight() {
      return inFlight !== null
    },
  }
}
