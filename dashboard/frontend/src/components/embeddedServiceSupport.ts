/**
 * Where a proxied service sent its availability probe, when it left the
 * service's proxy path: a service that redirects to the Dashboard itself, or
 * to its own login page outside the proxy, cannot load in a frame.
 */
export function embedRedirectTarget(
  response: { redirected: boolean; url: string },
  availabilityUrl: string,
  origin: string,
): string | null {
  if (!response.redirected || !response.url) return null
  const landed = new URL(response.url, origin)
  const root = new URL(availabilityUrl, origin).pathname.replace(/[^/]*$/, '')
  if (landed.origin === origin && landed.pathname.startsWith(root)) return null
  return landed.origin === origin ? landed.pathname : landed.origin
}
