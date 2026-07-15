interface VisibilityAwareRequestOptions {
  isHidden?: () => boolean
}

interface RunRequestOptions {
  allowHidden?: boolean
}

export function createVisibilityAwareRequest(
  task: () => Promise<void>,
  options: VisibilityAwareRequestOptions = {},
) {
  const isHidden = options.isHidden ?? (() => typeof document !== 'undefined' && document.hidden)
  let inFlight: Promise<void> | null = null

  return {
    run({ allowHidden = false }: RunRequestOptions = {}): Promise<void> {
      if (!allowHidden && isHidden()) {
        return Promise.resolve()
      }
      if (inFlight) {
        return inFlight
      }

      const request = Promise.resolve().then(task)
      const trackedRequest = request.finally(() => {
        if (inFlight === trackedRequest) {
          inFlight = null
        }
      })
      inFlight = trackedRequest
      return trackedRequest
    },
  }
}
