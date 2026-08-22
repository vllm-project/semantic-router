const RECOVERY_PREFIX = 'vllm-sr:asset-recovery:'

export function isRouteAssetLoadError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error || '')
  return /unable to preload css|failed to fetch dynamically imported module|error loading dynamically imported module|loading chunk [\w-]+ failed/i.test(
    message,
  )
}

export function recoverRouteAssetOnce(
  error: unknown,
  reload = () => window.location.reload(),
): boolean {
  if (!isRouteAssetLoadError(error) || typeof window === 'undefined') return false

  const message = error instanceof Error ? error.message : String(error)
  const asset = message.match(/\/assets\/[^\s)'"?]+/)?.[0] || 'route-chunk'
  const key = `${RECOVERY_PREFIX}${window.location.pathname}:${asset}`
  try {
    if (window.sessionStorage.getItem(key)) return false
    window.sessionStorage.setItem(key, '1')
  } catch {
    // A privacy-restricted session still deserves one best-effort refresh.
  }
  reload()
  return true
}
