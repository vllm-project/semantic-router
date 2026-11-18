import type { SystemStatus } from '../utils/routerRuntime'
import {
  countSignalsInProfile,
  effectiveChatEntrypoints,
  listRoutingScopes,
  type RoutingScopedConfigLike,
} from '../utils/routingScopes'
import { getNormalizedModels } from './configPageModelNormalization'
import { providerLogoKey } from './configPageModelPresentation'
import type { ConfigData, NormalizedModel } from './configPageSupport'
import { countModels } from './dashboardPageStats'
import type { RouterConfig } from './dashboardPageTypes'
import { getDecisionRuntimeSummary } from './dashboardRouterIntelligenceSupport'
import { modelProviderId } from './insightsConfigFacts'
import { timeAgo } from './insightsDisplayFormat'
import type { InsightsUnavailableReason } from './insightsPageApi'

/** The default public model of a stack whose configuration names none. */
export const DEFAULT_PUBLIC_MODEL = 'vllm-sr/auto'
/** The listener port of the local quickstart, when nothing else names one. */
export const DEFAULT_LISTENER_PORT = 8899
/** The variable `vllm-sr connect` and the agent settings read the listener key from. */
export const API_KEY_ENV = 'VLLM_SR_API_KEY'
/** The Dashboard variable that declares the gateway address clients use behind a proxy. */
export const PUBLIC_BASE_URL_ENV = 'VLLM_SR_PUBLIC_BASE_URL'

// ---------------------------------------------------------------- endpoint

interface ListenerLike {
  name?: unknown
  port?: unknown
  tls?: unknown
  apis?: unknown
  api_keys?: unknown
  api_key_files?: unknown
}

/**
 * Where the address comes from: a base URL the viewer typed, the public URL
 * the deployment declares, or the listener on the host this dashboard is
 * open at, which is only a guess behind a proxy.
 */
export type GatewayEndpointSource = 'custom' | 'public' | 'listener'

export interface GatewayEndpoint {
  /** Scheme, host and port, as `vllm-sr connect --base-url` takes it. */
  origin: string
  /** The OpenAI-compatible base URL SDKs take: the origin and /v1. */
  apiBase: string
  /** The listener that serves chat. */
  listener?: string
  /** Clients send a key: the listener checks client keys, or the public address requires one. */
  requiresKey: boolean
  source: GatewayEndpointSource
}

const nonEmpty = (value: unknown) => Array.isArray(value) && value.length > 0

// A listener without `apis` serves Chat Completions; with them, only the ones it names.
function servesChat(listener: ListenerLike): boolean {
  if (!listener.apis || typeof listener.apis !== 'object') return true
  const apis = listener.apis as Record<string, unknown>
  return 'chat_completions' in apis || 'openai' in apis
}

function browserHost(hostname: string): string {
  const host = hostname.trim()
  return !host || host === '0.0.0.0' || host === '::' || host === '[::]' ? 'localhost' : host
}

/** A typed base URL as an origin: no trailing slash and no /v1. Undefined when it is no http(s) URL. */
export function normalizeBaseUrl(value: string): string | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  try {
    const url = new URL(trimmed)
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return undefined
    const path = url.pathname.replace(/\/+$/, '').replace(/\/v1$/, '')
    return `${url.origin}${path}`
  } catch {
    return undefined
  }
}

/**
 * The inference address clients call: a base URL the viewer typed, else the
 * public URL the deployment declares, else the first listener that serves
 * Chat Completions, as `vllm-sr connect` reaches it, on the host the
 * dashboard is open at. The listener checks its client keys behind any
 * proxy, so they count for every address.
 */
export function resolveGatewayEndpoint({
  listeners,
  fallbackPort,
  hostname,
  override,
  publicBaseUrl,
  publicRequiresKey = false,
}: {
  listeners?: unknown
  fallbackPort?: number
  hostname: string
  override?: string
  publicBaseUrl?: string
  publicRequiresKey?: boolean
}): GatewayEndpoint {
  const candidates = Array.isArray(listeners)
    ? listeners.filter((item): item is ListenerLike => Boolean(item && typeof item === 'object'))
    : []
  const listener = candidates.find(servesChat)
  const listenerName = typeof listener?.name === 'string' ? listener.name : undefined
  const listenerKey = Boolean(
    listener && (nonEmpty(listener.api_keys) || nonEmpty(listener.api_key_files)),
  )
  const declared = publicBaseUrl ? normalizeBaseUrl(publicBaseUrl) : undefined
  const requiresKey = listenerKey || Boolean(declared && publicRequiresKey)
  const custom = override ? normalizeBaseUrl(override) : undefined
  const chosen = custom ?? declared
  if (chosen) {
    return {
      origin: chosen,
      apiBase: `${chosen}/v1`,
      listener: listenerName,
      requiresKey,
      source: custom ? 'custom' : 'public',
    }
  }
  const port =
    typeof listener?.port === 'number' && listener.port > 0
      ? listener.port
      : fallbackPort && fallbackPort > 0
        ? fallbackPort
        : DEFAULT_LISTENER_PORT
  const scheme = listener?.tls ? 'https' : 'http'
  const origin = `${scheme}://${browserHost(hostname)}:${port}`
  return {
    origin,
    apiBase: `${origin}/v1`,
    listener: listenerName,
    requiresKey,
    source: 'listener',
  }
}

// ---------------------------------------------------------------- public models

export interface PublicModel {
  name: string
  recipe: string
  api: 'chat' | 'systemone'
  isDefault: boolean
  description?: string
  decisions: number
  signals: number
  /** Providers of the backend models the recipe's decisions name, for their logos. */
  providers: string[]
}

interface EntrypointLike {
  api?: string
  model_names?: unknown
  recipe?: unknown
  default?: unknown
}

function decisionModels(decisions: unknown): string[] {
  if (!Array.isArray(decisions)) return []
  return decisions.flatMap((decision) => {
    const refs = (decision as { modelRefs?: unknown } | null)?.modelRefs
    return Array.isArray(refs)
      ? refs.flatMap((ref) => {
          const model = (ref as { model?: unknown } | null)?.model
          return typeof model === 'string' && model ? [model] : []
        })
      : []
  })
}

/**
 * The logo a backend model shows: its card's publisher logo, else the
 * organisation its id is published under, as in Qwen/Qwen3-8B, else the logo
 * of the provider serving it, else vLLM's.
 */
export function backendModelLogo(name: string, model?: NormalizedModel): string {
  const logo = model?.card_override?.presentation?.logo
  if (logo?.startsWith('package:')) return logo.slice('package:'.length)
  const vendor = modelProviderId(model?.provider_model_id, {}) ?? modelProviderId(name, {})
  if (vendor) return vendor
  const provider = model?.backend_refs
    ?.find((ref) => ref.provider?.trim())
    ?.provider?.trim()
    .toLowerCase()
  return provider ? providerLogoKey(provider) : 'vllm'
}

/**
 * The model names apps send, each with the recipe behind it: every chat
 * entrypoint (with the implicit `vllm-sr/auto` when no entrypoint serves the
 * default recipe) and then the System One entrypoints.
 */
export function listPublicModels(config: RouterConfig | null): PublicModel[] {
  if (!config) return []
  const scoped = config as RouterConfig & RoutingScopedConfigLike
  const scopes = new Map(listRoutingScopes(scoped).map((scope) => [scope.id, scope]))
  const backends = new Map(
    getNormalizedModels(config as unknown as ConfigData).map((model) => [model.name, model]),
  )
  const providerDefaults = (config.providers as { defaults?: { model?: unknown } } | undefined)
    ?.defaults
  const defaultModel =
    typeof providerDefaults?.model === 'string' ? providerDefaults.model : undefined
  const authored = (Array.isArray(config.entrypoints) ? config.entrypoints : []) as EntrypointLike[]
  const defaults = new Set(
    authored
      .filter((entry) => entry.default === true)
      .flatMap((entry) => (Array.isArray(entry.model_names) ? entry.model_names : [])),
  )
  const systemOne = authored.filter((entry) => entry.api === 'systemone')
  const describe = (name: string, recipe: string, api: PublicModel['api']): PublicModel => {
    const scope = scopes.get(recipe)
    const routed = decisionModels(scope?.routing.decisions)
    const logos = (routed.length > 0 ? routed : defaultModel ? [defaultModel] : []).map((model) =>
      backendModelLogo(model, backends.get(model)),
    )
    return {
      name,
      recipe,
      api,
      isDefault: defaults.has(name),
      description: scope?.description,
      decisions: Array.isArray(scope?.routing.decisions) ? scope.routing.decisions.length : 0,
      signals: countSignalsInProfile(scope?.routing).total,
      providers: [...new Set(logos)],
    }
  }
  const seen = new Set<string>()
  const models: PublicModel[] = []
  const add = (entry: EntrypointLike, api: PublicModel['api']) => {
    const recipe = typeof entry.recipe === 'string' && entry.recipe ? entry.recipe : 'default'
    for (const raw of Array.isArray(entry.model_names) ? entry.model_names : []) {
      const name = typeof raw === 'string' ? raw.trim() : ''
      if (!name || seen.has(name)) continue
      seen.add(name)
      models.push(describe(name, recipe, api))
    }
  }
  effectiveChatEntrypoints(scoped).forEach((entry) => add(entry, 'chat'))
  systemOne.forEach((entry) => add(entry, 'systemone'))
  if (!models.some((model) => model.isDefault)) {
    const first = models.find((model) => model.api === 'chat')
    if (first) first.isDefault = true
  }
  return models
}

// ---------------------------------------------------------------- snippets

export type SnippetLanguage = 'bash' | 'python' | 'ts' | 'json' | 'yaml'

export interface Snippet {
  label: string
  language: SnippetLanguage
  code: string
}

export interface SnippetTarget {
  origin: string
  apiBase: string
  model: string
  requiresKey: boolean
}

const PROMPT = 'Hello!'

/** A plain request in each client: curl, then the OpenAI Python and TypeScript SDKs. */
export function buildRequestSnippets({ apiBase, model, requiresKey }: SnippetTarget): Snippet[] {
  const curlAuth = requiresKey ? `\n  -H "Authorization: Bearer $${API_KEY_ENV}" \\` : ''
  return [
    {
      label: 'cURL',
      language: 'bash',
      code: `curl -sS -i ${apiBase}/chat/completions \\
  -H 'Content-Type: application/json' \\${curlAuth}
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "${PROMPT}"}]
  }'`,
    },
    {
      label: 'Python',
      language: 'python',
      code: `${requiresKey ? 'import os\n\n' : ''}from openai import OpenAI

client = OpenAI(
    base_url="${apiBase}",
    api_key=${requiresKey ? `os.environ["${API_KEY_ENV}"]` : '"none"  # the listener checks no key'},
)

response = client.chat.completions.with_raw_response.create(
    model="${model}",
    messages=[{"role": "user", "content": "${PROMPT}"}],
)
print(response.headers.get("x-vsr-selected-model"))
print(response.parse().choices[0].message.content)`,
    },
    {
      label: 'TypeScript',
      language: 'ts',
      code: `import OpenAI from "openai"

const client = new OpenAI({
  baseURL: "${apiBase}",
  apiKey: ${requiresKey ? `process.env.${API_KEY_ENV}` : '"none" // the listener checks no key'},
})

const { data, response } = await client.chat.completions
  .create({
    model: "${model}",
    messages: [{ role: "user", content: "${PROMPT}" }],
  })
  .withResponse()

console.log(response.headers.get("x-vsr-selected-model"))
console.log(data.choices[0].message.content)`,
    },
  ]
}

export type AgentHarnessId = 'claude_code' | 'codex' | 'opencode' | 'deepseek_harness'

export interface AgentSnippet extends Snippet {
  harness: AgentHarnessId
  /** Where the settings go, for a snippet that is a file. */
  file?: string
  /** One sentence on what the snippet does. */
  note: string
}

function connectCommand(harness: string, { origin, model }: SnippetTarget): string {
  const modelFlag = model === DEFAULT_PUBLIC_MODEL ? '' : ` \\\n  --model ${model}`
  return `vllm-sr connect ${harness} \\\n  --base-url ${origin}${modelFlag}`
}

const keyExport = (requiresKey: boolean) =>
  requiresKey ? `export ${API_KEY_ENV}=<the listener's API key>\n` : ''

// OpenCode sends a models entry's id; its key only names it inside the provider.
function openCodeModelKey(model: string): string {
  return model.replace(/^vllm-sr\//, '').replace(/[^\w.-]+/g, '-') || 'auto'
}

/**
 * How each coding agent reaches the gateway, as the agent-harness guide gives
 * it: `vllm-sr connect` for Claude Code and Codex, the provider settings for
 * OpenCode, and a profile patch for DeepSeek Harness.
 */
export function buildAgentSnippets(target: SnippetTarget): AgentSnippet[] {
  const { apiBase, model, requiresKey } = target
  const key = openCodeModelKey(model)
  return [
    {
      harness: 'claude_code',
      label: 'Claude Code',
      language: 'bash',
      code: `${keyExport(requiresKey)}${connectCommand('claude-code', target)}
claude`,
      note: 'Writes the gateway into the env block of ~/.claude/settings.json, checks a reply and a streamed tool loop, and asks before writing.',
    },
    {
      harness: 'codex',
      label: 'Codex',
      language: 'bash',
      code: `${keyExport(requiresKey)}${connectCommand('codex', target)}
codex --profile vllm-sr`,
      note: 'Writes the vllm-sr profile to ~/.codex/vllm-sr.config.toml and leaves ~/.codex/config.toml as it is.',
    },
    {
      harness: 'opencode',
      label: 'OpenCode',
      language: 'json',
      file: 'opencode.json',
      code: `{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm-sr": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM Semantic Router",
      "options": {
        "baseURL": "${apiBase}",
        "apiKey": "{env:${API_KEY_ENV}}"
      },
      "models": {
        "${key}": { "id": "${model}", "limit": { "context": 131072, "output": 32000 } }
      }
    }
  },
  "model": "vllm-sr/${key}"
}`,
      note: 'Goes in opencode.json in the project or in ~/.config/opencode/. Set limit.context to the entrypoint’s context window, where OpenCode compacts the session.',
    },
    {
      harness: 'deepseek_harness',
      label: 'DeepSeek Harness',
      language: 'yaml',
      file: '~/.dsh/profiles/<profile>/cordis.patch.yml',
      code: `- id: llm-pi-ai
  config:
    providers:
      vllm-sr:
        displayName: vLLM Semantic Router
        api: openai-completions
        baseURL: ${apiBase}
        apiKeyEnv: ${API_KEY_ENV}
        models:
          - id: ${model}

- id: agent-default-model
  config:
    provider: vllm-sr
    model: ${model}`,
      note: 'A patch for your dsh profile: a vllm-sr provider on the gateway’s /v1, which new sessions start on. dsh reads the key from ~/.dsh/.env.',
    },
  ]
}

// ---------------------------------------------------------------- checklist

export type StartStepId = 'router' | 'backend' | 'decision-model' | 'first-request'
export type StartStepState = 'done' | 'todo' | 'pending'

export interface StartStep {
  id: StartStepId
  label: string
  detail: string
  state: StartStepState
  to: string
}

export interface ReplayFacts {
  /** Loading until the first answer. */
  state: 'loading' | 'ready' | 'unavailable'
  count: number
  latestAt?: string
  reason?: InsightsUnavailableReason
}

const ROUTER_DETAIL: Record<string, string> = {
  healthy: 'Healthy',
  degraded: 'Running, with services that need attention',
  starting: 'Starting up',
  standby: 'On standby',
}

/** What a first-time setup still lacks, from the live status, configuration and Router Replay. */
export function buildStartChecklist({
  status,
  statusFailed,
  config,
  replay,
  now = Date.now(),
}: {
  status: SystemStatus | null
  statusFailed: boolean
  config: RouterConfig | null
  replay: ReplayFacts
  now?: number
}): StartStep[] {
  const overall = status?.overall?.toLowerCase()
  const running = overall === 'healthy' || overall === 'degraded'
  const router: StartStep = {
    id: 'router',
    label: 'Run the router',
    detail: status
      ? running
        ? `${ROUTER_DETAIL[overall ?? ''] ?? 'Running'}${status.version ? ` · ${status.version}` : ''}`
        : (ROUTER_DETAIL[overall ?? ''] ?? 'Not running · start it with vllm-sr serve')
      : statusFailed
        ? 'Status is unavailable'
        : 'Checking…',
    state: running ? 'done' : status || statusFailed ? 'todo' : 'pending',
    to: '/observe/status',
  }

  const models = config ? countModels(config) : 0
  const backend: StartStep = {
    id: 'backend',
    label: 'Connect a backend model',
    detail: config
      ? models > 0
        ? `${models} backend model${models === 1 ? '' : 's'} configured`
        : 'No backend model yet'
      : 'Configuration is unavailable',
    state: models > 0 ? 'done' : config ? 'todo' : 'pending',
    to: '/models/backends',
  }

  const runtime = config ? getDecisionRuntimeSummary(config, status?.models) : null
  const decision: StartStep = {
    id: 'decision-model',
    label: 'Deploy a decision model',
    detail: runtime
      ? runtime.state === 'ready'
        ? `${runtime.model} is ready`
        : runtime.state === 'attention'
          ? `${runtime.model} needs attention`
          : `${runtime.model} is not reported by the runtime`
      : 'Configuration is unavailable',
    state: runtime?.state === 'ready' ? 'done' : runtime ? 'todo' : 'pending',
    to: '/models/decision',
  }

  const firstRequest: StartStep = {
    id: 'first-request',
    label: 'Send your first request',
    detail:
      replay.state === 'ready'
        ? replay.count > 0
          ? replay.latestAt
            ? `Last request ${timeAgo(replay.latestAt, now)}`
            : 'Requests recorded'
          : 'No request recorded yet'
        : replay.state === 'unavailable'
          ? replay.reason === 'replay_off'
            ? 'Router Replay is not recording'
            : replay.reason === 'forbidden'
              ? 'Your role cannot read Router Replay'
              : 'Router Replay is unreachable'
          : 'Checking…',
    state:
      replay.state === 'ready' && replay.count > 0
        ? 'done'
        : replay.state === 'loading'
          ? 'pending'
          : 'todo',
    to: '/get-started',
  }

  return [router, backend, decision, firstRequest]
}
