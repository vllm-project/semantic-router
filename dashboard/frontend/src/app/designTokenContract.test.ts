import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const theme = readFileSync(new URL('../ui/styles/theme.css', import.meta.url), 'utf8')
const indexHtml = readFileSync(new URL('../../index.html', import.meta.url), 'utf8')

const roles = [
  'background',
  'foreground',
  'surface',
  'surface-subtle',
  'card',
  'popover',
  'primary',
  'primary-foreground',
  'muted-foreground',
  'subtle-foreground',
  'border',
  'input',
  'ring',
  'success',
  'warning',
  'destructive',
  'sidebar',
  'chart-1',
  'chart-8',
]

function block(selector: string): string {
  const start = theme.indexOf(`${selector} {`)
  expect(start, `${selector} block`).toBeGreaterThanOrEqual(0)
  return theme.slice(start, theme.indexOf('\n}', start))
}

function sources(directory: URL, relative = ''): { path: string; source: string }[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const url = new URL(`${entry.name}${entry.isDirectory() ? '/' : ''}`, directory)
    if (entry.isDirectory()) return sources(url, `${relative}${entry.name}/`)
    if (!/\.(css|tsx?)$/.test(entry.name) || entry.name.endsWith('.test.ts')) return []
    return [{ path: `${relative}${entry.name}`, source: readFileSync(url, 'utf8') }]
  })
}

describe('design token contract', () => {
  it('defines every semantic role for both themes', () => {
    const light = block(':root')
    const dark = block('.dark')
    for (const role of roles) {
      expect(light, `light --${role}`).toMatch(new RegExp(`--${role}:`))
    }
    for (const role of [
      'background',
      'foreground',
      'surface',
      'primary',
      'border',
      'success',
      'warning',
      'destructive',
      'sidebar',
    ]) {
      expect(dark, `dark --${role}`).toMatch(new RegExp(`--${role}:`))
    }
  })

  it('keeps the vLLM brand colors and status colors apart', () => {
    expect(theme).toMatch(/vLLM blue/i)
    expect(theme).toContain('--brand-gold:')
    expect(theme).toContain('prefers-reduced-motion: reduce')
  })

  it('advertises both color schemes to the browser before the app loads', () => {
    expect(indexHtml).toContain('media="(prefers-color-scheme: dark)"')
    expect(indexHtml).toContain('media="(prefers-color-scheme: light)"')
    expect(indexHtml).toContain('/theme-init.js')
  })

  it('uses tokens instead of raw colors in the design system and shell', () => {
    const allowed = new Set(['brand/brand.tsx', 'AppSidebar.tsx'])
    const violations = [
      ...sources(new URL('../ui/', import.meta.url)),
      ...sources(new URL('../shell/', import.meta.url)),
    ]
      .filter(({ path }) => !path.startsWith('styles/') && !allowed.has(path))
      .filter(({ source }) => /#[0-9a-fA-F]{6}\b|rgba?\(/.test(source))
      .map(({ path }) => path)
    expect(violations).toEqual([])
  })

  it('does not reintroduce the retired palettes', () => {
    const retired = ['#76b900', '#5a8f00', '#8fd400', '#00d4ff', '--nv-', '#e31b23']
    const violations = sources(new URL('../', import.meta.url)).flatMap(({ path, source }) =>
      retired
        .filter((token) => source.toLowerCase().includes(token))
        .map((token) => `${path}: ${token}`),
    )
    expect(violations).toEqual([])
  })
})
