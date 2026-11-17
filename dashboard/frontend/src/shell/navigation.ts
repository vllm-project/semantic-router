import type { LucideIcon } from 'lucide-react'
import {
  Blend,
  Blocks,
  Boxes,
  Braces,
  BrainCircuit,
  ChartGantt,
  ChartLine,
  FileJson,
  FlaskConical,
  HeartPulse,
  House,
  Layers,
  LibraryBig,
  ListTree,
  MessagesSquare,
  Network,
  Radar,
  Rocket,
  ScrollText,
  Settings,
  SlidersHorizontal,
  Split,
  UsersRound,
} from 'lucide-react'

export interface NavItem {
  id: string
  label: string
  to: string
  icon: LucideIcon
  description: string
  keywords?: string[]
  badge?: string
  end?: boolean
}

export interface NavGroup {
  id: string
  label?: string
  /** Footer groups collapse into one sidebar entry that opens their first item. */
  placement?: 'main' | 'footer'
  icon?: LucideIcon
  items: NavItem[]
}

/**
 * The dashboard's information architecture. Every destination is one click from
 * the sidebar; groups follow the operator's jobs rather than backend modules.
 */
export const navigation: NavGroup[] = [
  {
    id: 'start',
    items: [
      { id: 'home', label: 'Home', to: '/', icon: House, end: true, description: 'Health, traffic and what needs attention', keywords: ['overview', 'dashboard'] },
      { id: 'get-started', label: 'Get started', to: '/get-started', icon: Rocket, description: 'Endpoint, model names and code to call the router', keywords: ['connect', 'api', 'curl', 'quickstart', 'endpoint'] },
      { id: 'playground', label: 'Playground', to: '/playground', icon: MessagesSquare, description: 'Chat through the router and inspect each routing step', keywords: ['chat', 'test'] },
    ],
  },
  {
    id: 'routing',
    label: 'Routing',
    items: [
      { id: 'recipes', label: 'Recipes', to: '/routing/recipes', icon: Layers, description: 'Public model names and the recipes behind them', keywords: ['entrypoints', 'mixture-of-models', 'mom', 'public models'] },
      { id: 'decisions', label: 'Decisions', to: '/routing/decisions', icon: Split, description: 'Rules that map signals to models and plugins', keywords: ['routes', 'rules'] },
      { id: 'signals', label: 'Signals', to: '/routing/signals', icon: Radar, description: 'Evidence extracted from every request', keywords: ['keywords', 'domain', 'embedding', 'classifier'] },
      { id: 'projections', label: 'Projections', to: '/routing/projections', icon: Blend, description: 'Derived scores that combine signals', keywords: ['scores', 'partitions'] },
      { id: 'topology', label: 'Topology', to: '/routing/topology', icon: Network, description: 'The whole routing graph, end to end', keywords: ['brain', 'graph', 'map'] },
      { id: 'editor', label: 'Config editor', to: '/routing/editor', icon: Braces, description: 'Edit, validate and deploy routing as code', keywords: ['builder', 'dsl', 'yaml', 'deploy'] },
    ],
  },
  {
    id: 'models',
    label: 'Models',
    items: [
      { id: 'backends', label: 'Backend models', to: '/models/backends', icon: Boxes, description: 'Model endpoints the router can send traffic to', keywords: ['providers', 'endpoints', 'llm'] },
      { id: 'decision-models', label: 'Decision models', to: '/models/decision', icon: BrainCircuit, description: 'System One models that power routing intelligence', keywords: ['system one', 'vela', 'decision 3.0'] },
      { id: 'hub', label: 'Model Hub', to: '/models/hub', icon: LibraryBig, description: 'Model catalog with specs, pricing and capabilities', keywords: ['catalog', 'providers', 'pricing'] },
    ],
  },
  {
    id: 'observe',
    label: 'Observe',
    items: [
      { id: 'requests', label: 'Requests', to: '/observe/requests', icon: ListTree, description: 'Every routed request with its decision, cost and trace', keywords: ['insights', 'replay', 'records'] },
      { id: 'evaluations', label: 'Evaluations', to: '/observe/evaluations', icon: FlaskConical, description: 'Benchmark recipes and models with sr-bench', keywords: ['sr-bench', 'benchmark'] },
      { id: 'status', label: 'Status', to: '/observe/status', icon: HeartPulse, description: 'Service health and model readiness', keywords: ['health', 'uptime'] },
      { id: 'logs', label: 'Logs', to: '/observe/logs', icon: ScrollText, description: 'Live output from router, gateway and dashboard', keywords: ['events'] },
      { id: 'metrics', label: 'Metrics', to: '/observe/metrics', icon: ChartLine, description: 'Latency, throughput and errors over time', keywords: ['grafana', 'prometheus'] },
      { id: 'traces', label: 'Traces', to: '/observe/traces', icon: ChartGantt, description: 'Distributed traces across the serving path', keywords: ['jaeger', 'tracing'] },
    ],
  },
  {
    id: 'settings',
    label: 'Settings',
    placement: 'footer',
    icon: Settings,
    items: [
      { id: 'router-settings', label: 'Router settings', to: '/settings/router', icon: SlidersHorizontal, description: 'Router-wide defaults, services and stores', keywords: ['global config'] },
      { id: 'integrations', label: 'Integrations', to: '/settings/integrations', icon: Blocks, description: 'MCP servers and tools', keywords: ['mcp', 'tools'] },
      { id: 'access', label: 'Users & access', to: '/settings/access', icon: UsersRound, description: 'People, roles, invitations and audit log', keywords: ['users', 'rbac', 'audit'] },
      { id: 'schema', label: 'Schema reference', to: '/settings/schema', icon: FileJson, description: 'Every configuration field, documented', keywords: ['reference', 'contract'] },
    ],
  },
]

export const allNavItems: NavItem[] = navigation.flatMap((group) => group.items)

export function findNavItem(pathname: string): { item: NavItem; group: NavGroup } | undefined {
  let best: { item: NavItem; group: NavGroup } | undefined
  for (const group of navigation) {
    for (const item of group.items) {
      const matches = item.end ? pathname === item.to : pathname === item.to || pathname.startsWith(`${item.to}/`)
      if (matches && (!best || item.to.length > best.item.to.length)) best = { item, group }
    }
  }
  return best
}
