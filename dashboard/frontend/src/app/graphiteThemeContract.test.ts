import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

interface ThemeSource {
  path: string
  source: string
}

function readThemeSources(directory: URL, relativeDirectory = '..'): ThemeSource[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const relativePath = `${relativeDirectory}/${entry.name}`
    const entryUrl = new URL(`${entry.name}${entry.isDirectory() ? '/' : ''}`, directory)

    if (entry.isDirectory()) {
      return readThemeSources(entryUrl, relativePath)
    }
    if (!entry.isFile() || !/\.(css|ts|tsx)$/.test(entry.name)) {
      return []
    }
    return [{ path: relativePath, source: readFileSync(entryUrl, 'utf8') }]
  })
}

const themeSources = readThemeSources(new URL('../', import.meta.url))
const legacyBrandTokens = [
  '#76b900',
  '#5a8f00',
  '#8fd400',
  '#00d4ff',
  'rgba(118, 185, 0',
  'rgb(118, 185, 0',
  '--nv-',
]

describe('graphite dashboard theme contract', () => {
  it('defines the shared AMD-aligned surface and focus tokens', () => {
    const globalStyles = readFileSync(new URL('../index.css', import.meta.url), 'utf8')

    expect(globalStyles).toContain('--surface-canvas: #050505')
    expect(globalStyles).toContain('--brand-amd: #e31b23')
    expect(globalStyles).toContain('--focus-ring:')
    expect(globalStyles).toContain('prefers-reduced-motion: reduce')
  })

  it('does not reintroduce the retired NVIDIA green and neon-cyan palette', () => {
    const violations = themeSources.flatMap(({ path, source }) => {
      if (path.endsWith('.test.ts')) {
        return []
      }
      const normalizedSource = source.toLowerCase()
      return legacyBrandTokens
        .filter((token) => normalizedSource.includes(token.toLowerCase()))
        .map((token) => `${path}: ${token}`)
    })

    expect(violations).toEqual([])
  })
})
