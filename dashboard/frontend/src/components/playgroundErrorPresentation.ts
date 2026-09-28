export interface PlaygroundErrorPresentation {
  message: string
  technicalDetails?: string
}

export type PlaygroundErrorInput = string | PlaygroundErrorPresentation | null

export class PlaygroundRequestFailure extends Error {
  readonly productMessage: string
  readonly technicalDetails: string

  constructor(productMessage: string, technicalDetails: string) {
    super(technicalDetails)
    this.name = 'PlaygroundRequestFailure'
    this.productMessage = productMessage
    this.technicalDetails = technicalDetails
  }
}

function httpFailureMessage(status: number): string {
  if (status === 401 || status === 403) {
    return 'This request is not authorized. Check your access and model configuration, then try again.'
  }
  if (status === 404) {
    return 'The selected model is not available. Refresh the configuration, then try again.'
  }
  if (status === 429) {
    return 'The model service is busy. Wait a moment, then try again.'
  }
  if (status >= 500) {
    return 'The model service is temporarily unavailable. Try again.'
  }
  return 'The model request was rejected. Review the model and request settings, then try again.'
}

export function playgroundHTTPFailure(
  status: number,
  statusText: string,
  responseBody: string,
): PlaygroundRequestFailure {
  const responseLine = `HTTP ${status}${statusText.trim() ? ` ${statusText.trim()}` : ''}`
  const body = responseBody.trim()
  return new PlaygroundRequestFailure(
    httpFailureMessage(status),
    body ? `${responseLine}\n${body}` : responseLine,
  )
}

export function playgroundResponseFailure(
  productMessage: string,
  technicalDetails: string,
): PlaygroundRequestFailure {
  return new PlaygroundRequestFailure(productMessage, technicalDetails)
}

export function normalizePlaygroundError(
  error: Exclude<PlaygroundErrorInput, null>,
): PlaygroundErrorPresentation {
  return typeof error === 'string' ? { message: error } : error
}

export function playgroundErrorPresentation(error: unknown): PlaygroundErrorPresentation {
  if (error instanceof PlaygroundRequestFailure) {
    return { message: error.productMessage, technicalDetails: error.technicalDetails }
  }
  if (error instanceof Error && error.name === 'TimeoutError') {
    return {
      message: 'The request timed out. Try again, or reduce the request size.',
      technicalDetails: `${error.name}: ${error.message}`,
    }
  }
  return {
    message: 'The request could not be completed. Check the selected model and try again.',
    technicalDetails:
      error instanceof Error
        ? `${error.name}: ${error.message}`
        : `Unknown failure: ${String(error)}`,
  }
}
