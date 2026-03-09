export interface SkillTemplate {
  id: string
  name: string
  description: string
  emoji: string
  category: string
  builtin: boolean
}

export interface IdentityConfig {
  name: string
  emoji: string
  role: string
  vibe: string
  principles: string
  boundaries: string
}

export interface ContainerConfig {
  containerName: string
  gatewayPort: number
  authToken: string
  modelBaseUrl: string
  modelName: string
  memoryBackend: string
  memoryBaseUrl: string
  vectorStore: string
  browserEnabled: boolean
  baseImage: string
  networkMode: string
}

export interface OpenClawStatus {
  running: boolean
  containerName: string
  gatewayUrl: string
  port: number
  healthy: boolean
  error: string
  image?: string
  createdAt?: string
  teamId?: string
  teamName?: string
  agentName?: string
  agentEmoji?: string
  agentRole?: string
  agentVibe?: string
  agentPrinciples?: string
}

export interface TeamProfile {
  id: string
  name: string
  vibe?: string
  role?: string
  principal?: string
  description?: string
  createdAt?: string
  updatedAt?: string
}

export interface ProvisionResponse {
  success: boolean
  message: string
  workspaceDir: string
  configPath: string
  containerId: string
  dockerCmd: string
  composeYaml: string
}

export const PROVISION_STEPS = [
  { key: 'identity', label: 'Identity & Team' },
  { key: 'skills', label: 'Skills' },
  { key: 'config', label: 'Configuration' },
  { key: 'deploy', label: 'Deploy' },
]

interface KernelFeature {
  title: string
  module: string
  description: string
  icon: string
}

export const OPENCLAW_FEATURES: KernelFeature[] = [
  {
    title: 'Intelligent Routing',
    module: 'Routing Orchestrator',
    description: 'Model selection with cost-accuracy balance driven by vLLM SR routing intelligence.',
    icon: '\u{1F9ED}',
  },
  {
    title: 'Safety Guardrails',
    module: 'Policy & Safety Manager',
    description: 'Protect agents from jailbreak attacks, PII leakage, and hallucination risk.',
    icon: '\u{1F6E1}\uFE0F',
  },
  {
    title: 'Hierarchical Memory Storage',
    module: 'Memory Context Manager',
    description: 'Persistent context and memory management for long-horizon, multi-step execution.',
    icon: '\u{1F9E0}',
  },
  {
    title: 'Knowledge Sharing',
    module: 'Knowledge Exchanger',
    description: 'Cross-agent experience and knowledge sharing for faster team learning loops.',
    icon: '\u{1F501}',
  },
  {
    title: 'Isolation & Team Management',
    module: 'Tenant & Isolation Manager',
    description: 'Multi-agent isolation with centralized team operations in one control plane.',
    icon: '\u{1F9E9}',
  },
]

const FALLBACK_MODEL_BASE_URL = 'http://127.0.0.1:8801/v1'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object'

const toPort = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    const normalized = Math.trunc(value)
    if (normalized >= 1 && normalized <= 65535) return normalized
    return null
  }
  if (typeof value === 'string') {
    const normalized = Number.parseInt(value.trim(), 10)
    if (Number.isFinite(normalized) && normalized >= 1 && normalized <= 65535) {
      return normalized
    }
  }
  return null
}

const normalizeListenerHost = (value: unknown): string => {
  const raw = typeof value === 'string' ? value.trim() : ''
  if (!raw || raw === '0.0.0.0' || raw === '::' || raw === '[::]') {
    return '127.0.0.1'
  }
  return raw
}

const formatHostForUrl = (host: string): string => {
  if (host.includes(':') && !host.startsWith('[') && !host.endsWith(']')) {
    return `[${host}]`
  }
  return host
}

const extractListenerCandidates = (config: unknown): Record<string, unknown>[] => {
  if (!isRecord(config)) return []

  const listeners = Array.isArray(config.listeners) ? config.listeners : []
  const apiServer = isRecord(config.api_server) ? config.api_server : null
  const apiServerListeners = apiServer && Array.isArray(apiServer.listeners) ? apiServer.listeners : []

  return [...listeners, ...apiServerListeners].filter(isRecord)
}

export const deriveModelBaseUrlFromRouterConfig = (config: unknown): string | null => {
  const listeners = extractListenerCandidates(config)
  for (const listener of listeners) {
    const port = toPort(listener.port)
    if (!port) continue
    const host = formatHostForUrl(normalizeListenerHost(listener.address))
    return `http://${host}:${port}/v1`
  }
  return null
}

export const getInitialModelBaseUrl = (): string => FALLBACK_MODEL_BASE_URL
