export type LayoutDropdownKey = 'manager' | 'analysisOps'

export type LayoutConfigSection = 'models' | 'signals' | 'decisions' | 'router-config' | 'mcp'

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
}

export const PRIMARY_NAV_LINKS: LayoutNavLink[] = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Playground', to: '/playground' },
  { label: 'Brain', to: '/topology' },
  { label: 'DSL', to: '/builder' },
]

export const SECONDARY_NAV_LINKS: LayoutNavLink[] = [
  { label: 'ClawOS', to: '/clawos' },
]

export const MANAGER_MENU_SECTIONS: LayoutMenuSection[] = [
  {
    items: [
      { kind: 'config', label: 'Models', configSection: 'models' },
      { kind: 'config', label: 'Decisions', configSection: 'decisions' },
      { kind: 'config', label: 'Signals', configSection: 'signals' },
    ],
  },
]

export const ANALYSIS_OPERATIONS_MENU_SECTIONS: LayoutMenuSection[] = [
  {
    title: 'Analysis',
    items: [
      { kind: 'route', label: 'Evaluation', to: '/evaluation' },
      { kind: 'route', label: 'Replay', to: '/replay' },
      { kind: 'route', label: 'Ratings', to: '/ratings' },
    ],
  },
  {
    title: 'Operations',
    items: [
      { kind: 'config', label: 'Router Config', configSection: 'router-config' },
      { kind: 'config', label: 'MCP Servers', configSection: 'mcp' },
      { kind: 'route', label: 'Status', to: '/status' },
      { kind: 'route', label: 'Logs', to: '/logs' },
      { kind: 'route', label: 'Grafana', to: '/monitoring' },
      { kind: 'route', label: 'Tracing', to: '/tracing' },
    ],
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
