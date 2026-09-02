import type { ProductIconName } from './ProductIcon'

export type LayoutDropdownKey = 'build' | 'operate'

export type LayoutConfigSection =
  | 'models'
  | 'signals'
  | 'projections'
  | 'decisions'
  | 'entrypoints-recipes'
  | 'global-config'
  | 'mcp'

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
}

export const PRIMARY_NAV_LINKS: LayoutNavLink[] = [
  { label: 'Dashboard', icon: 'dashboard', to: '/dashboard' },
  { label: 'Playground', icon: 'playground', to: '/playground' },
]

export const BUILD_MENU_CATEGORIES: LayoutMenuCategory[] = [
  {
    key: 'routing',
    label: 'Routing',
    description: 'Design the signal-to-decision path that selects each model route.',
    sections: [
      {
        title: 'Models',
        description: 'Connect models and compose public model endpoints.',
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
        title: 'Intelligence',
        description: 'Shape the evidence and decisions behind every route.',
        items: [
          { kind: 'config', label: 'Signals', icon: 'signal', configSection: 'signals' },
          {
            kind: 'config',
            label: 'Projections',
            icon: 'projection',
            configSection: 'projections',
          },
          { kind: 'config', label: 'Decisions', icon: 'decision', configSection: 'decisions' },
        ],
      },
      {
        title: 'Build',
        description: 'See the whole system or edit its configuration.',
        items: [
          { kind: 'route', label: 'Brain', icon: 'topology', to: '/topology' },
          { kind: 'route', label: 'Builder', icon: 'code', to: '/builder' },
        ],
      },
    ],
  },
  {
    key: 'outcomes',
    label: 'Outcomes',
    description: 'Inspect routing choices, measure quality, and tune model behavior.',
    sections: [
      {
        title: 'Inspect',
        description: 'Understand what the router selected and why.',
        items: [
          {
            kind: 'route',
            label: 'Insights',
            icon: 'insight',
            to: '/insights',
            matchMode: 'prefix',
          },
        ],
      },
      {
        title: 'Evaluate',
        description: 'Benchmark signal and system-level behavior.',
        items: [{ kind: 'route', label: 'Evaluation', icon: 'evaluation', to: '/evaluation' }],
      },
      {
        title: 'Tune',
        description: 'Prepare and validate the router model stack.',
        items: [{ kind: 'route', label: 'ML Setup', icon: 'compute', to: '/ml-setup' }],
      },
    ],
  },
  {
    key: 'knowledge',
    label: 'Knowledge Base',
    description: 'Bring governed context into signal extraction and route policy.',
    sections: [
      {
        title: 'Knowledge Base',
        description: 'Manage the retrieval inventory used by knowledge signals.',
        items: [
          {
            kind: 'route',
            label: 'Bases',
            icon: 'database',
            to: '/knowledge-bases/bases',
            activePathPattern: /^\/knowledge-bases\/[^/]+\/map\/?$/,
          },
          { kind: 'route', label: 'Groups', icon: 'database', to: '/knowledge-bases/groups' },
          { kind: 'route', label: 'Labels', icon: 'label', to: '/knowledge-bases/labels' },
        ],
      },
    ],
  },
  {
    key: 'integrations',
    label: 'Integration',
    description: 'Connect external capabilities to the routing workspace.',
    sections: [
      {
        title: 'Integrations',
        description: 'Extend the control plane with tools and agent runtimes.',
        items: [
          { kind: 'config', label: 'MCP Servers', icon: 'tool', configSection: 'mcp' },
          { kind: 'route', label: 'OpenClaw', icon: 'claw', to: '/openclaw' },
        ],
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
        items: [{ kind: 'route', label: 'Status', icon: 'status', to: '/status' }],
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
    key: 'platform-access',
    label: 'Platform & Access',
    description: 'Manage global defaults and who can change the control plane.',
    sections: [
      {
        title: 'Platform',
        description: 'Configure router-wide defaults and infrastructure bindings.',
        items: [
          {
            kind: 'config',
            label: 'Global Config',
            icon: 'settings',
            configSection: 'global-config',
          },
        ],
      },
      {
        title: 'Access',
        description: 'Administer dashboard identities and roles.',
        items: [{ kind: 'route', label: 'Users', icon: 'user', to: '/users' }],
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
