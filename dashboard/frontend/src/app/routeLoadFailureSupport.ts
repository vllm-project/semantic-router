const CHUNK_FAILURE_PATTERNS = [
  /chunkloaderror/i,
  /loading (?:css )?chunk [\w-]+ failed/i,
  /failed to fetch dynamically imported module/i,
  /error loading dynamically imported module/i,
  /importing a module script failed/i,
  /unable to preload css/i,
]

export function routeLoadErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return typeof error === 'string' ? error : 'Unknown route error'
}

export function isRouteChunkLoadError(error: unknown): boolean {
  const name = error instanceof Error ? error.name : ''
  const message = routeLoadErrorMessage(error)
  return CHUNK_FAILURE_PATTERNS.some((pattern) => pattern.test(`${name}: ${message}`))
}

export function getRouteLoadFailureCopy(error: unknown, routeLabel: string) {
  if (isRouteChunkLoadError(error)) {
    return {
      eyebrow: 'Route update interrupted',
      title: `${routeLabel} needs to reconnect`,
      description:
        'A dashboard update or temporary network issue interrupted this page bundle. Retry the route first; reload the dashboard if the deployed version changed.',
    }
  }

  return {
    eyebrow: 'Route unavailable',
    title: `${routeLabel} could not open`,
    description:
      'The page encountered an unexpected error before it was ready. Retry without leaving your current dashboard session.',
  }
}
