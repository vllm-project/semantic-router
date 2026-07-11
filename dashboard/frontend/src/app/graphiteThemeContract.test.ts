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
  '#081000',
  'rgba(118, 185, 0',
  'rgb(118, 185, 0',
  '--nv-',
]

const graphiteSurfaceContracts = [
  {
    path: '../pages/MLSetupPage.module.css',
    forbiddenPatterns: [/#(?:3b82f6|2563eb|60a5fa)\b/i, /rgba?\(\s*59\s*,\s*130\s*,\s*246\b/i],
  },
  {
    path: '../components/MCPConfigPanel.module.css',
    forbiddenPatterns: [
      /#(?:1e1e2e|181825|313244|cdd6f4|a6adc8|89b4fa|cba6f7|a6e3a1|101311|0f1304)\b/i,
      /rgba?\(\s*(?:137\s*,\s*180\s*,\s*250|166\s*,\s*227\s*,\s*161|203\s*,\s*166\s*,\s*247)\b/i,
      /rgba?\(\s*(?:6\s*,\s*10\s*,\s*9|10\s*,\s*14\s*,\s*13|7\s*,\s*11\s*,\s*10|3\s*,\s*6\s*,\s*6|5\s*,\s*8\s*,\s*8)\b/i,
    ],
  },
  {
    path: '../pages/SetupWizardPage.module.css',
    forbiddenPatterns: [
      /#(?:060706|d8f7a5|f5f7ff)\b/i,
      /rgba?\(\s*(?:12\s*,\s*18\s*,\s*36|9\s*,\s*12\s*,\s*24|0\s*,\s*180\s*,\s*216)\b/i,
    ],
  },
  {
    path: '../pages/FleetSimPage.module.css',
    forbiddenPatterns: [
      /#d9e3f2\b/i,
      /rgba?\(\s*(?:7\s*,\s*10\s*,\s*19|9\s*,\s*12\s*,\s*22|8\s*,\s*11\s*,\s*20|3\s*,\s*6\s*,\s*14|6\s*,\s*10\s*,\s*18)\b/i,
    ],
  },
  {
    path: '../pages/OpenClawPage.module.css',
    forbiddenPatterns: [
      /#(?:0f172a|020617|3b82f6|22d3ee|a8b4c8|cbd5e1)\b/i,
      /rgba?\(\s*(?:15\s*,\s*23\s*,\s*42|2\s*,\s*6\s*,\s*23|34\s*,\s*211\s*,\s*238)\b/i,
      /--color-accent-cyan\b/i,
    ],
  },
  {
    path: '../pages/InsightsPage.module.css',
    forbiddenPatterns: [
      /rgba?\(\s*(?:2\s*,\s*5\s*,\s*5|6\s*,\s*10\s*,\s*9|9\s*,\s*13\s*,\s*13|8\s*,\s*13\s*,\s*13|8\s*,\s*12\s*,\s*12|12\s*,\s*18\s*,\s*18)\b/i,
    ],
  },
  {
    path: '../pages/ConfigPageTaxonomyClassifiers.module.css',
    forbiddenPatterns: [/#f4ffe7\b/i, /rgba?\(\s*24\s*,\s*34\s*,\s*17\b/i],
  },
  {
    path: '../pages/topology/components/CustomNodes/CustomNodes.module.css',
    forbiddenPatterns: [/#e8efe1\b/i, /rgba?\(\s*20\s*,\s*30\s*,\s*20\b/i],
  },
  {
    path: '../components/RouterModelInventory.module.css',
    forbiddenPatterns: [/#bef264\b/i],
  },
] as const

describe('graphite dashboard theme contract', () => {
  it('defines the shared AMD-aligned surface and focus tokens', () => {
    const globalStyles = readFileSync(new URL('../index.css', import.meta.url), 'utf8')

    expect(globalStyles).toContain('--surface-canvas: #050505')
    expect(globalStyles).toContain('--brand-amd: #e31b23')
    expect(globalStyles).toContain('--focus-ring:')
    expect(globalStyles).toContain('prefers-reduced-motion: reduce')
  })

  it('does not reintroduce the retired legacy brand palette', () => {
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

  it('keeps targeted graphite surfaces free of legacy blue and green tints', () => {
    const violations = graphiteSurfaceContracts.flatMap(({ path, forbiddenPatterns }) => {
      const source = readFileSync(new URL(path, import.meta.url), 'utf8')

      return forbiddenPatterns
        .filter((pattern) => pattern.test(source))
        .map((pattern) => `${path}: ${pattern}`)
    })

    expect(violations).toEqual([])
  })
})
