/**
 * The release a version names, without its pre-release or build part:
 * v0.5.0 for v0.5.0-dev.e1ba40498f. Anything else, such as a bare commit, stays as it is.
 */
export function shortRelease(version: string): string {
  const text = version.trim()
  const release = /^v?(\d+\.\d+(?:\.\d+)?)/.exec(text)
  return release ? `v${release[1]}` : text || '—'
}
