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

export const truncateText = (value?: string, maxLength = 180): string => {
  const text = (value || '').trim()
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength).trim()}...`
}

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
    description:
      'Model selection with cost-accuracy balance driven by vLLM SR routing intelligence.',
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

export const getInitialModelBaseUrl = (routerPublicUrl = ''): string => {
  const candidate = routerPublicUrl.trim()
  if (!candidate) return FALLBACK_MODEL_BASE_URL
  try {
    const url = new URL(candidate)
    return `${url.origin}/v1`
  } catch {
    return FALLBACK_MODEL_BASE_URL
  }
}
