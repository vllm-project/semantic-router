const PUBLIC_API_PATH = /^\/v1(?:\/|$)/

function canonicalPublicBase(value: string): string {
  const candidate = value.trim()
  if (!candidate) return ''
  const url = new URL(candidate)
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    throw new Error('Router public URL must be an HTTP(S) origin.')
  }
  if (url.search || url.hash) throw new Error('Router public URL cannot include query or fragment.')
  if (url.pathname !== '/' && url.pathname !== '') {
    throw new Error('Router public URL must be an origin without a path.')
  }
  return url.origin
}

export function routerPublicEndpoint(publicBase: string, path: string): string {
  if (!PUBLIC_API_PATH.test(path)) throw new Error('Router public API path must start with /v1.')
  const base = canonicalPublicBase(publicBase)
  return base ? `${base}${path}` : path
}

export function siblingRouterPublicEndpoint(inferenceEndpoint: string, path: string): string {
  if (!PUBLIC_API_PATH.test(path)) throw new Error('Router public API path must start with /v1.')
  if (!/^https?:\/\//i.test(inferenceEndpoint)) return path
  const endpoint = new URL(inferenceEndpoint)
  return `${endpoint.origin}${path}`
}
