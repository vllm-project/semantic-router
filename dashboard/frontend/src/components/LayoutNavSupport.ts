import { FLEET_SIM_NAV_ITEMS } from '../utils/fleetSimApi'

export type LayoutDropdownKey = 'manager' | 'analysisOps'

export type LayoutConfigSection =
  | 'models'
  | 'signals'
  | 'projections'
  | 'decisions'
  | 'global-config'
  | 'mcp'

type LayoutRouteMenuItem = {
  kind: 'route'
  label: string
  to: string
}

type LayoutConfigMenuItem = {
  kind: 'config'
  label: string
  configSection: LayoutConfigSection
}

export type LayoutMenuItem = LayoutRouteMenuItem | LayoutConfigMenuItem

export interface LayoutMenuSection {
  title?: string
  items: LayoutMenuItem[]
}

export interface LayoutNavLink {
  label: string
  to: string
  matchMode?: 'exact' | 'prefix'
}

export const PRIMARY_NAV_LINKS: LayoutNavLink[] = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Playground', to: '/playground' },
  { label: 'Brain', to: '/topology' },
  { label: 'DSL', to: '/builder' },
  { label: 'Insight', to: '/insights', matchMode: 'prefix' },
]

export const SECONDARY_NAV_LINKS: LayoutNavLink[] = []

export const MANAGER_MENU_SECTIONS: LayoutMenuSection[] = [
  {
    items: [
      { kind: 'route', label: 'Users', to: '/users' },
      { kind: 'route', label: 'ClawOS', to: '/clawos' },
    ],
  },
  {
    items: [
      { kind: 'config', label: 'Models', configSection: 'models' },
      { kind: 'config', label: 'Decisions', configSection: 'decisions' },
      { kind: 'config', label: 'Signals', configSection: 'signals' },
      { kind: 'config', label: 'Projections', configSection: 'projections' },
    ],
  },
]

export const KNOWLEDGE_BASE_MENU_SECTIONS: LayoutMenuSection[] = [
  {
    title: 'Knowledge Base',
    items: [
      { kind: 'route', label: 'Bases', to: '/knowledge-bases/bases' },
      { kind: 'route', label: 'Groups', to: '/knowledge-bases/groups' },
      { kind: 'route', label: 'Labels', to: '/knowledge-bases/labels' },
    ],
  },
]

export const ANALYSIS_OPERATIONS_MENU_SECTIONS: LayoutMenuSection[] = [
  {
    title: 'Analysis',
    items: [
      { kind: 'config', label: 'Global Config', configSection: 'global-config' },
      { kind: 'route', label: 'Evaluation', to: '/evaluation' },
      { kind: 'route', label: 'Ratings', to: '/ratings' },
      { kind: 'route', label: 'ML Setup', to: '/ml-setup' },
      { kind: 'config', label: 'MCP Setup', configSection: 'mcp' },
    ],
  },
  {
    title: 'Observability',
    items: [
      { kind: 'route', label: 'Status', to: '/status' },
      { kind: 'route', label: 'Logs', to: '/logs' },
      { kind: 'route', label: 'Monitoring', to: '/monitoring' },
      { kind: 'route', label: 'Tracing', to: '/tracing' },
    ],
  },
]

export const FLEET_SIM_MENU_SECTIONS: LayoutMenuSection[] = [
  {
    title: 'Simulator',
    items: FLEET_SIM_NAV_ITEMS.map((item) => ({
      kind: 'route' as const,
      label: item.label,
      to: item.to,
    })),
  },
]

export function isLayoutMenuItemActive(
  item: LayoutMenuItem,
  pathname: string,
  isConfigPage: boolean,
  configSection?: string
): boolean {
  if (item.kind === 'config') {
    return isConfigPage && configSection === item.configSection
  }

  return pathname === item.to
}

export function hasActiveLayoutMenuSection(
  sections: LayoutMenuSection[],
  pathname: string,
  isConfigPage: boolean,
  configSection?: string
): boolean {
  return sections.some(section =>
    section.items.some(item => isLayoutMenuItemActive(item, pathname, isConfigPage, configSection))
  )
}

export function filterLayoutMenuSections(
  sections: LayoutMenuSection[],
  predicate: (item: LayoutMenuItem) => boolean
): LayoutMenuSection[] {
  return sections
    .map(section => ({
      ...section,
      items: section.items.filter(predicate),
    }))
    .filter(section => section.items.length > 0)
}
