import { FLEET_SIM_NAV_ITEMS } from '../utils/fleetSimApi'

export type LayoutDropdownKey = 'build' | 'analyze' | 'operate'

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
  matchMode?: 'exact' | 'prefix'
  activePathPattern?: RegExp
}

type LayoutConfigMenuItem = {
  kind: 'config'
  label: string
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
  to: string
  matchMode?: 'exact' | 'prefix'
}

export const PRIMARY_NAV_LINKS: LayoutNavLink[] = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Playground', to: '/playground' },
]

export const BUILD_MENU_CATEGORIES: LayoutMenuCategory[] = [
  {
    key: 'routing',
    label: 'Routing',
    description: 'Design the signal-to-decision path that selects each model route.',
    sections: [
      {
        title: 'Design',
        description: 'Author and inspect the routing graph.',
        items: [
          { kind: 'route', label: 'Config Builder', to: '/builder' },
          { kind: 'route', label: 'Brain Topology', to: '/topology' },
        ],
      },
      {
        title: 'Evidence',
        description: 'Define the facts and projections every decision can use.',
        items: [
          { kind: 'config', label: 'Signals', configSection: 'signals' },
          { kind: 'config', label: 'Projections', configSection: 'projections' },
        ],
      },
      {
        title: 'Dispatch',
        description: 'Bind policy outcomes to the available model fleet.',
        items: [
          { kind: 'config', label: 'Decisions', configSection: 'decisions' },
          { kind: 'config', label: 'Models', configSection: 'models' },
        ],
      },
    ],
  },
  {
    key: 'knowledge',
    label: 'Knowledge',
    description: 'Bring governed context into signal extraction and route policy.',
    sections: [
      {
        title: 'Knowledge Base',
        description: 'Manage the retrieval inventory used by knowledge signals.',
        items: [
          {
            kind: 'route',
            label: 'Bases',
            to: '/knowledge-bases/bases',
            activePathPattern: /^\/knowledge-bases\/[^/]+\/map\/?$/,
          },
          { kind: 'route', label: 'Groups', to: '/knowledge-bases/groups' },
          { kind: 'route', label: 'Labels', to: '/knowledge-bases/labels' },
        ],
      },
    ],
  },
  {
    key: 'integrations',
    label: 'Integrations & Policy',
    description: 'Connect external capabilities and enforce request-path controls.',
    sections: [
      {
        title: 'Integrations',
        description: 'Extend the control plane with tools and agent runtimes.',
        items: [
          { kind: 'config', label: 'MCP Servers', configSection: 'mcp' },
          { kind: 'route', label: 'ClawOS', to: '/clawos' },
        ],
      },
      {
        title: 'Policy',
        description: 'Review the security controls applied around routing.',
        items: [{ kind: 'route', label: 'Security Policy', to: '/security' }],
      },
    ],
  },
]

export const ANALYZE_MENU_CATEGORIES: LayoutMenuCategory[] = [
  {
    key: 'outcomes',
    label: 'Outcomes',
    description: 'Inspect routing choices, measure quality, and tune model behavior.',
    sections: [
      {
        title: 'Inspect',
        description: 'Understand what the router selected and why.',
        items: [{ kind: 'route', label: 'Insights', to: '/insights', matchMode: 'prefix' }],
      },
      {
        title: 'Evaluate',
        description: 'Benchmark signal and system-level behavior.',
        items: [{ kind: 'route', label: 'Evaluation', to: '/evaluation' }],
      },
      {
        title: 'Tune',
        description: 'Prepare and validate the router model stack.',
        items: [{ kind: 'route', label: 'ML Setup', to: '/ml-setup' }],
      },
    ],
  },
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
          to: item.to,
        })),
      },
      {
        title: 'Inventory',
        description: 'Model the hardware pools available to the router.',
        items: FLEET_SIM_NAV_ITEMS.slice(2, 3).map((item) => ({
          kind: 'route' as const,
          label: item.label,
          to: item.to,
        })),
      },
      {
        title: 'Runs',
        description: 'Review completed and in-progress simulations.',
        items: FLEET_SIM_NAV_ITEMS.slice(3).map((item) => ({
          kind: 'route' as const,
          label: item.label,
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
        items: [{ kind: 'route', label: 'Status', to: '/status' }],
      },
      {
        title: 'Diagnostics',
        description: 'Read runtime events and investigate failures.',
        items: [{ kind: 'route', label: 'Logs', to: '/logs' }],
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
        items: [{ kind: 'route', label: 'Grafana', to: '/monitoring' }],
      },
      {
        title: 'Tracing',
        description: 'Inspect request paths across the serving system.',
        items: [{ kind: 'route', label: 'Tracing', to: '/tracing' }],
      },
    ],
  },
  {
    key: 'platform-access',
    label: 'Platform & Access',
    description: 'Manage global defaults and who can change the control plane.',
    sections: [
      {
        title: 'Platform',
        description: 'Configure router-wide defaults and infrastructure bindings.',
        items: [{ kind: 'config', label: 'Global Config', configSection: 'global-config' }],
      },
      {
        title: 'Access',
        description: 'Administer dashboard identities and roles.',
        items: [{ kind: 'route', label: 'Users', to: '/users' }],
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
