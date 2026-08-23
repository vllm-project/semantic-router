import { FLEET_SIM_NAV_ITEMS } from '../utils/fleetSimApi'
import type { ProductIconName } from './ProductIcon'

export type LayoutDropdownKey = 'build' | 'operate'

export type LayoutConfigSection =
  | 'models'
  | 'signals'
  | 'projections'
  | 'decisions'
  | 'entrypoints-recipes'
  | 'agent'

type LayoutRouteMenuItem = {
  kind: 'route'
  label: string
  icon: ProductIconName
  to: string
  matchMode?: 'exact' | 'prefix'
  activePathPattern?: RegExp
}

type LayoutConfigMenuItem = {
  kind: 'config'
  label: string
  icon: ProductIconName
  configSection: LayoutConfigSection
}

export type LayoutMenuItem = LayoutRouteMenuItem | LayoutConfigMenuItem

export interface LayoutMenuSection {
  title: string
  description?: string
  items: LayoutMenuItem[]
}

export interface LayoutMenuCategory {
  key: string
  label: string
  description: string
  sections: LayoutMenuSection[]
}

export interface LayoutNavLink {
  label: string
  icon: ProductIconName
  to: string
  matchMode?: 'exact' | 'prefix'
  activePathPattern?: RegExp
}

export const PRIMARY_NAV_LINKS: LayoutNavLink[] = [
  { label: 'Dashboard', icon: 'dashboard', to: '/dashboard' },
  { label: 'Playground', icon: 'playground', to: '/playground' },
  {
    label: 'Access',
    icon: 'key',
    to: '/access/usage',
    activePathPattern: /^\/(?:access(?:\/|$)|logs(?:\/|$))/,
  },
]

export const BUILD_MENU_CATEGORIES: LayoutMenuCategory[] = [
  {
    key: 'routing',
    label: 'Routing',
    description: 'Design the signal-to-decision path that selects each model route.',
    sections: [
      {
        title: 'Models',
        description: 'Configure provider models and compose the available fleet.',
        items: [
          { kind: 'config', label: 'Models', icon: 'model', configSection: 'models' },
          {
            kind: 'config',
            label: 'Mixture-of-Models',
            icon: 'mixture',
            configSection: 'entrypoints-recipes',
          },
        ],
      },
      {
        title: 'Routing Logic',
        description: 'Define the signals, projections, and decisions that select a route.',
        items: [
          { kind: 'config', label: 'Signals', icon: 'signal', configSection: 'signals' },
          {
            kind: 'config',
            label: 'Projections',
            icon: 'projection',
            configSection: 'projections',
          },
          {
            kind: 'config',
            label: 'Decisions',
            icon: 'decision',
            configSection: 'decisions',
          },
        ],
      },
      {
        title: 'Design',
        description: 'Inspect the routing graph, request outcomes, or author its DSL.',
        items: [
          { kind: 'route', label: 'Brain Topology', icon: 'topology', to: '/topology' },
          { kind: 'route', label: 'DSL Builder', icon: 'code', to: '/builder' },
          {
            kind: 'route',
            label: 'Insights',
            icon: 'insight',
            to: '/insights',
            matchMode: 'prefix',
          },
        ],
      },
    ],
  },
  {
    key: 'integrations',
    label: 'Integrations',
    description: 'Bring trusted tools into vLLM-SR.',
    sections: [
      {
        title: 'Integrations',
        description: 'Choose how the Agent works and what it can use.',
        items: [
          { kind: 'config', label: 'vLLM-SR Agent', icon: 'tool', configSection: 'agent' },
          { kind: 'route', label: 'OpenClaw', icon: 'claw', to: '/openclaw' },
        ],
      },
    ],
  },
]

export const ANALYZE_MENU_CATEGORIES: LayoutMenuCategory[] = [
  {
    key: 'fleet-simulation',
    label: 'Fleet Simulation',
    description: 'Plan heterogeneous capacity before traffic reaches the live fleet.',
    sections: [
      {
        title: 'Plan',
        description: 'Define workloads and compare fleet strategies.',
        items: FLEET_SIM_NAV_ITEMS.slice(0, 2).map((item) => ({
          kind: 'route' as const,
          label: item.label,
          icon: 'fleet' as const,
          to: item.to,
        })),
      },
      {
        title: 'Inventory',
        description: 'Model the hardware pools available to the router.',
        items: FLEET_SIM_NAV_ITEMS.slice(2, 3).map((item) => ({
          kind: 'route' as const,
          label: item.label,
          icon: 'fleet' as const,
          to: item.to,
        })),
      },
      {
        title: 'Runs',
        description: 'Review completed and in-progress simulations.',
        items: FLEET_SIM_NAV_ITEMS.slice(3).map((item) => ({
          kind: 'route' as const,
          label: item.label,
          icon: 'fleet' as const,
          to: item.to,
        })),
      },
    ],
  },
]

export const OPERATE_MENU_CATEGORIES: LayoutMenuCategory[] = [
  {
    key: 'runtime',
    label: 'Runtime',
    description: 'Check service readiness and diagnose the live routing path.',
    sections: [
      {
        title: 'Health',
        description: 'Track router services and loaded model readiness.',
        items: [
          { kind: 'route', label: 'Status', icon: 'status', to: '/status' },
          {
            kind: 'route',
            label: 'Plugin Operations',
            icon: 'puzzle',
            to: '/plugins',
            matchMode: 'prefix',
          },
        ],
      },
      {
        title: 'Diagnostics',
        description: 'Read runtime events and investigate failures.',
        items: [{ kind: 'route', label: 'Logs', icon: 'logs', to: '/logs' }],
      },
    ],
  },
  {
    key: 'observability',
    label: 'Observability',
    description: 'Follow metrics and traces across every routed request.',
    sections: [
      {
        title: 'Metrics',
        description: 'Open the operational dashboard for fleet and router telemetry.',
        items: [{ kind: 'route', label: 'Grafana', icon: 'chart', to: '/monitoring' }],
      },
      {
        title: 'Tracing',
        description: 'Inspect request paths across the serving system.',
        items: [{ kind: 'route', label: 'Tracing', icon: 'trace', to: '/tracing' }],
      },
    ],
  },
  {
    key: 'platform',
    label: 'Platform',
    description: 'Manage router-wide defaults and infrastructure bindings.',
    sections: [
      {
        title: 'Platform',
        description: 'Configure router-wide defaults and infrastructure bindings.',
        items: [
          { kind: 'route', label: 'Evaluation', icon: 'evaluation', to: '/evaluation' },
          { kind: 'route', label: 'ML Setup', icon: 'compute', to: '/ml-setup' },
        ],
      },
    ],
  },
]

export function isLayoutMenuItemActive(
  item: LayoutMenuItem,
  pathname: string,
  isConfigPage: boolean,
  configSection?: string,
): boolean {
  if (item.kind === 'config') {
    return isConfigPage && configSection === item.configSection
  }

  if (item.activePathPattern?.test(pathname)) {
    return true
  }

  return item.matchMode === 'prefix' ? pathname.startsWith(item.to) : pathname === item.to
}

export function hasActiveLayoutMenuCategory(
  categories: LayoutMenuCategory[],
  pathname: string,
  isConfigPage: boolean,
  configSection?: string,
): boolean {
  return categories.some((category) =>
    category.sections.some((section) =>
      section.items.some((item) =>
        isLayoutMenuItemActive(item, pathname, isConfigPage, configSection),
      ),
    ),
  )
}

export function findActiveLayoutMenuCategory(
  categories: LayoutMenuCategory[],
  pathname: string,
  isConfigPage: boolean,
  configSection?: string,
): string | undefined {
  return categories.find((category) =>
    category.sections.some((section) =>
      section.items.some((item) =>
        isLayoutMenuItemActive(item, pathname, isConfigPage, configSection),
      ),
    ),
  )?.key
}

export function filterLayoutMenuCategories(
  categories: LayoutMenuCategory[],
  predicate: (item: LayoutMenuItem, category: LayoutMenuCategory) => boolean,
): LayoutMenuCategory[] {
  return categories
    .map((category) => ({
      ...category,
      sections: category.sections
        .map((section) => ({
          ...section,
          items: section.items.filter((item) => predicate(item, category)),
        }))
        .filter((section) => section.items.length > 0),
    }))
    .filter((category) => category.sections.length > 0)
}
