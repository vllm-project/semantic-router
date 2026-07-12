import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  DASHBOARD_COLOR_BENDS_MOTION,
  DASHBOARD_MOTION_COLORS,
} from '../components/dashboardMotionTheme'

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

const dashboardIndex = readFileSync(new URL('../../index.html', import.meta.url), 'utf8')
const themeSources = [
  ...readThemeSources(new URL('../', import.meta.url)),
  { path: '../../index.html', source: dashboardIndex },
]
const legacyBrandTokens = [
  '#76b900',
  '#5a8f00',
  '#8fd400',
  '#00d4ff',
  '#081000',
  'rgba(118, 185, 0',
  'rgb(118, 185, 0',
  '0x76b900',
  '0x8fd400',
  '0x00d4ff',
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
    path: '../components/ConfigNav.module.css',
    forbiddenPatterns: [
      /#(?:6366f1|8b5cf6)\b/i,
      /rgba?\(\s*(?:99\s*,\s*102\s*,\s*241|139\s*,\s*92\s*,\s*246)\b/i,
    ],
  },
  {
    path: '../components/ChatComponent.module.css',
    forbiddenPatterns: [
      /rgba?\(\s*(?:40\s*,\s*44\s*,\s*52|19\s*,\s*21\s*,\s*27|46\s*,\s*50\s*,\s*58|21\s*,\s*23\s*,\s*29|7\s*,\s*8\s*,\s*9)\b/i,
    ],
  },
  {
    path: '../components/DashboardSurfaceHero.module.css',
    forbiddenPatterns: [/backdrop-filter\s*:/i, /border-radius:\s*999px/i, /radial-gradient\(/i],
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
    forbiddenPatterns: [
      /#(?:bef264|102113|101c12|dcfce7)\b/i,
      /rgba?\(\s*(?:34\s*,\s*197\s*,\s*94|134\s*,\s*239\s*,\s*172|190\s*,\s*242\s*,\s*100|12\s*,\s*32\s*,\s*18|16\s*,\s*33\s*,\s*19)\b/i,
    ],
  },
  {
    path: '../components/InsightsCharts.tsx',
    forbiddenPatterns: [
      /#(?:718096|5a6c7d|606c7a|556b7d|f59e0b)\b/i,
      /118\s*\+\s*\(156\s*-\s*118\)/i,
      /185\s*\+\s*\(163\s*-\s*185\)/i,
      /--color-text-primary\b/i,
    ],
  },
  {
    path: '../components/InsightsCharts.module.css',
    forbiddenPatterns: [
      /rgba?\(\s*(?:6\s*,\s*10\s*,\s*9|12\s*,\s*16\s*,\s*16)\b/i,
      /--color-(?:accent-cyan|text-primary)\b/i,
    ],
  },
  {
    path: '../lib/dslLanguage.ts',
    forbiddenPatterns: [
      /(?:6A9955|C586C0|4EC9B0|DCDCAA|9CDCFE|CE9178|D7BA7D|B5CEA8|569CD6|ED1C24)/i,
      /editor\.background['"]?:\s*['"]#1a1a1a/i,
    ],
  },
] as const

describe('graphite dashboard theme contract', () => {
  it('defines the shared AMD-aligned surface and focus tokens', () => {
    const globalStyles = readFileSync(new URL('../index.css', import.meta.url), 'utf8')

    expect(globalStyles).toContain('--surface-canvas: #050505')
    expect(globalStyles).toContain('--brand-amd: #e31b23')
    expect(globalStyles).toContain('--focus-ring:')
    expect(globalStyles).toContain('prefers-reduced-motion: reduce')
    expect(dashboardIndex).toContain('<meta name="theme-color" content="#050505" />')
  })

  it('keeps expressive motion aligned to the graphite and AMD palette', () => {
    expect(DASHBOARD_MOTION_COLORS).toEqual(['#f5f5f7', '#e31b23', '#5f636a'])
    expect(DASHBOARD_COLOR_BENDS_MOTION).toEqual({
      rotation: 20,
      speed: 0.2,
      scale: 1,
      frequency: 1,
      warpStrength: 1,
      mouseInfluence: 1,
      parallax: 0.5,
      noise: 0.08,
      autoRotate: 0.8,
    })
  })

  it('keeps the DSL editor and analytics surfaces on the shared palette', () => {
    const dslTheme = readFileSync(new URL('../lib/dslLanguage.ts', import.meta.url), 'utf8')
    const insightsCharts = readFileSync(
      new URL('../components/InsightsCharts.tsx', import.meta.url),
      'utf8',
    )

    expect(dslTheme).toContain("'editor.background': '#050505'")
    expect(dslTheme).toContain("'editorCursor.foreground': '#e31b23'")
    expect(insightsCharts).toMatch(/const CHART_COLORS = \[\s*'#e31b23'/)
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
