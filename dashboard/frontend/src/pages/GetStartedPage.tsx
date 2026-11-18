import { useMemo, useState, type FormEvent, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowUpRight,
  BrainCircuit,
  Check,
  CircleDot,
  KeyRound,
  Layers,
  Pencil,
  Play,
  Rocket,
  ShieldAlert,
  Undo2,
} from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CodeBlock,
  CopyField,
  IconTile,
  InlineCode,
  Input,
  PageContainer,
  PageHeader,
  ProviderLogo,
  ProviderStack,
  Skeleton,
  StatusDot,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  cn,
} from '../ui'
import {
  playgroundErrorPresentation,
  type PlaygroundErrorPresentation,
} from '../components/playgroundErrorPresentation'
import {
  formatRoutingMetadataValue,
  MATCHED_SIGNAL_KEY_PREFIX,
} from '../components/routingMetadataDisplay'
import {
  X_VSR_COST,
  X_VSR_COST_CURRENCY,
  X_VSR_REPLAY_ID,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_RECIPE,
} from '../generated/vsrHeaders'
import { useDashboardObservation } from '../utils/useDashboardObservation'
import { decisionModelHealth } from './dashboardRouterIntelligenceSupport'
import type { ModelRuntimeInventory } from './decisionModelManagement'
import { formatLatency, timeAgo } from './insightsDisplayFormat'
import { HarnessLogo } from './InsightsHarness'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'
import { buildModelProviders, modelProviderId, type ConfigDocument } from './insightsConfigFacts'
import { StartChecklistCard } from './StartChecklist'
import {
  API_KEY_ENV,
  DEFAULT_PUBLIC_MODEL,
  PUBLIC_BASE_URL_ENV,
  buildAgentSnippets,
  buildRequestSnippets,
  buildStartChecklist,
  listPublicModels,
  normalizeBaseUrl,
  resolveGatewayEndpoint,
  type AgentHarnessId,
  type GatewayEndpointSource,
  type PublicModel,
} from './startSupport'
import {
  sendGatewayTestRequest,
  useGetStartedData,
  type TestRequestResult,
} from './useGetStartedData'

const GUIDE_URL = 'https://vllm-sr.ai/docs/installation/agent-harness'
const AGENT_GUIDES: Partial<Record<AgentHarnessId, string>> = {
  claude_code: `${GUIDE_URL}#connect-with-vllm-sr-connect`,
  codex: `${GUIDE_URL}#connect-with-vllm-sr-connect`,
  opencode: `${GUIDE_URL}#opencode`,
}
const BASE_URL_KEY = 'vsr-get-started-base-url'
const EASE = [0.16, 1, 0.3, 1] as const

function readStoredBaseUrl(): string {
  try {
    return window.localStorage.getItem(BASE_URL_KEY) ?? ''
  } catch {
    return ''
  }
}

function writeStoredBaseUrl(value: string) {
  try {
    if (value) window.localStorage.setItem(BASE_URL_KEY, value)
    else window.localStorage.removeItem(BASE_URL_KEY)
  } catch {
    // the choice lasts for this page only
  }
}

function Step({
  index,
  title,
  description,
  done,
  last,
  id,
  children,
}: {
  index: number
  title: string
  description?: ReactNode
  done?: boolean
  last?: boolean
  id?: string
  children: ReactNode
}) {
  return (
    <li className="relative flex gap-4" id={id}>
      <div className="flex flex-col items-center">
        <span
          className={cn(
            'z-10 grid size-7 shrink-0 place-items-center rounded-full border text-xs font-semibold tabular transition-colors',
            done
              ? 'border-transparent bg-success text-white'
              : 'border-border-strong bg-surface text-muted-foreground',
          )}
        >
          {done ? <Check className="size-3.5" strokeWidth={3} /> : index}
        </span>
        {last ? null : <span className="mt-1 w-px flex-1 bg-border" />}
      </div>
      <section
        aria-labelledby={`step-${index}`}
        className={cn('flex min-w-0 flex-1 flex-col gap-3', last ? 'pb-2' : 'pb-9')}
      >
        <div className="pt-0.5">
          <h2 id={`step-${index}`} className="text-[15px] font-semibold tracking-[-0.012em]">
            {title}
          </h2>
          {description ? (
            <p className="mt-0.5 text-[13px] text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {children}
      </section>
    </li>
  )
}

function ModelChoice({
  model,
  active,
  onSelect,
}: {
  model: PublicModel
  active: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      className={cn(
        'relative flex min-w-0 flex-col gap-2 rounded-xl border bg-card p-3.5 text-left shadow-xs outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/40',
        active
          ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
          : 'border-border hover:border-border-strong',
      )}
    >
      <span className="flex min-w-0 items-center gap-2">
        <Layers className="size-4 shrink-0 text-chart-1" />
        <span className="truncate font-mono text-[12.5px] font-semibold" title={model.name}>
          {model.name}
        </span>
        {model.isDefault ? (
          <Badge size="sm" tone="brand" className="shrink-0">
            default
          </Badge>
        ) : null}
      </span>
      <span className="line-clamp-2 min-h-8 text-xs leading-relaxed text-muted-foreground">
        {model.description || `Routes through the ${model.recipe} recipe.`}
      </span>
      <span className="mt-auto flex items-center justify-between gap-2 pt-1">
        {model.providers.length > 0 ? (
          <ProviderStack providers={model.providers} size={14} />
        ) : (
          <span className="font-mono text-[11px] text-subtle-foreground">{model.recipe}</span>
        )}
        <span className="text-[11px] text-subtle-foreground tabular">
          {model.decisions} decision{model.decisions === 1 ? '' : 's'}
        </span>
      </span>
      {active ? (
        <motion.span
          layoutId="get-started-model-check"
          transition={{ duration: 0.28, ease: EASE }}
          className="absolute -right-1.5 -top-1.5 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground shadow-sm"
        >
          <Check className="size-3" strokeWidth={3} />
        </motion.span>
      ) : null}
    </button>
  )
}

function EndpointNote({ source, listener }: { source: GatewayEndpointSource; listener?: string }) {
  if (source === 'custom') return <span>The base URL you entered, saved in this browser.</span>
  if (source === 'public') return <span>The public address this deployment declares.</span>
  return (
    <span>
      {listener ? (
        <>
          From listener <InlineCode>{listener}</InlineCode> on the host this dashboard is open at;
        </>
      ) : (
        'From the listener port setup reports, on the host this dashboard is open at;'
      )}{' '}
      set <InlineCode>{PUBLIC_BASE_URL_ENV}</InlineCode> when the router sits behind a proxy.
    </span>
  )
}

function EndpointStep({
  apiBase,
  model,
  listener,
  source,
  publicDeclared,
  ready,
  requiresKey,
  onCustomBase,
}: {
  apiBase: string
  model: string
  listener?: string
  source: GatewayEndpointSource
  /** The deployment declares a public address, which a typed one replaces. */
  publicDeclared: boolean
  /** Setup has reported any public address, so the derived guess never shows first. */
  ready: boolean
  requiresKey: boolean
  onCustomBase: (value: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const invalid = Boolean(draft.trim()) && !normalizeBaseUrl(draft)

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const value = normalizeBaseUrl(draft)
    if (!value) return
    onCustomBase(value)
    setEditing(false)
  }

  return (
    <>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Base URL</span>
          {editing ? (
            <form onSubmit={submit} className="flex items-center gap-2">
              <Input
                autoFocus
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="https://gateway.example.com"
                aria-label="Gateway base URL"
                aria-invalid={invalid || undefined}
                className="font-mono text-[12.5px]"
              />
              <Button type="submit" size="sm" disabled={!draft.trim() || invalid}>
                Use
              </Button>
              <Button type="button" size="sm" variant="ghost" onClick={() => setEditing(false)}>
                Cancel
              </Button>
            </form>
          ) : ready ? (
            <CopyField value={apiBase} />
          ) : (
            <Skeleton className="h-9" />
          )}
        </div>
        <div className="flex min-w-0 flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Model</span>
          <CopyField value={model} />
        </div>
      </div>
      <div className="flex flex-col gap-1.5 text-xs text-muted-foreground">
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <EndpointNote source={source} listener={listener} />
          {editing || !ready ? null : source === 'custom' ? (
            <Button
              variant="ghost"
              size="xs"
              className="h-6 px-1.5"
              onClick={() => onCustomBase('')}
            >
              <Undo2 />
              {publicDeclared ? 'Use the public address' : 'Use the listener'}
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="xs"
              className="h-6 px-1.5"
              onClick={() => {
                setDraft(apiBase)
                setEditing(true)
              }}
            >
              <Pencil />
              {source === 'public' ? 'Use another address' : 'Behind another address?'}
            </Button>
          )}
        </p>
        {requiresKey ? (
          <p className="flex items-start gap-2">
            <KeyRound className="mt-0.5 size-3.5 shrink-0 text-info" />
            <span>
              {source === 'listener'
                ? 'The listener checks client keys.'
                : 'This address requires a client key.'}{' '}
              Send yours as a bearer token; the agent setups read it from{' '}
              <InlineCode>{API_KEY_ENV}</InlineCode>.
            </span>
          </p>
        ) : (
          <p className="flex items-start gap-2">
            <ShieldAlert className="mt-0.5 size-3.5 shrink-0 text-warning" />
            <span>
              The listener checks no client key. Add <InlineCode>api_keys</InlineCode> in{' '}
              <Link to="/settings/router" className="font-medium text-primary hover:underline">
                Router settings
              </Link>{' '}
              {source === 'public'
                ? 'unless the proxy in front checks keys.'
                : 'before you expose it.'}
            </span>
          </p>
        )}
      </div>
    </>
  )
}

function AgentStep({
  snippets,
  agent,
  onAgent,
}: {
  snippets: ReturnType<typeof buildAgentSnippets>
  agent: AgentHarnessId
  onAgent: (agent: AgentHarnessId) => void
}) {
  return (
    <Tabs
      variant="pill"
      value={agent}
      onValueChange={(value) => onAgent(value as AgentHarnessId)}
      className="gap-3"
    >
      <TabsList className="max-w-full flex-wrap">
        {snippets.map((snippet) => (
          <TabsTrigger key={snippet.harness} value={snippet.harness}>
            <HarnessLogo harness={snippet.harness} size={14} />
            {snippet.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {snippets.map((snippet) => (
        <TabsContent
          key={snippet.harness}
          value={snippet.harness}
          className="flex flex-col gap-2.5"
        >
          <CodeBlock
            title={snippet.file ?? 'Terminal'}
            tabs={[{ label: snippet.label, language: snippet.language, code: snippet.code }]}
          />
          <p className="flex flex-wrap items-baseline gap-x-2 text-xs leading-relaxed text-muted-foreground">
            <span>{snippet.note}</span>
            {AGENT_GUIDES[snippet.harness] ? (
              <a
                href={AGENT_GUIDES[snippet.harness]}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-0.5 font-medium text-primary hover:underline"
              >
                Guide
                <ArrowUpRight className="size-3" />
              </a>
            ) : null}
          </p>
        </TabsContent>
      ))}
    </Tabs>
  )
}

interface RouteCell {
  label: string
  value: ReactNode
}

function ModelValue({ name, providers }: { name: string; providers: Record<string, string> }) {
  const provider = modelProviderId(name, providers)
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      {provider ? <ProviderLogo provider={provider} size={14} /> : null}
      <span className="truncate">{name}</span>
    </span>
  )
}

function resultCells(result: TestRequestResult, providers: Record<string, string>): RouteCell[] {
  const { headers } = result
  const signals = Object.entries(headers)
    .filter(([key]) => key.startsWith(MATCHED_SIGNAL_KEY_PREFIX))
    .flatMap(([key, value]) =>
      value.split(',').map((name) => formatRoutingMetadataValue(key, name.trim())),
    )
  const cost = headers[X_VSR_COST]
  return [
    {
      label: 'Signals matched',
      value: signals.length ? signals.slice(0, 3).join(' · ') : 'None reported',
    },
    {
      label: 'Decision',
      value: headers[X_VSR_SELECTED_DECISION]
        ? formatRoutingMetadataValue(X_VSR_SELECTED_DECISION, headers[X_VSR_SELECTED_DECISION])
        : 'Not reported',
    },
    {
      label: 'Model',
      value: headers[X_VSR_SELECTED_MODEL] ? (
        <ModelValue name={headers[X_VSR_SELECTED_MODEL]} providers={providers} />
      ) : (
        'Not reported'
      ),
    },
    {
      label: cost ? 'Latency · cost' : 'Latency',
      value: `${formatLatency(result.latencyMs)}${cost ? ` · ${cost} ${headers[X_VSR_COST_CURRENCY] ?? ''}`.trimEnd() : ''}`,
    },
  ]
}

function recordCells(record: InsightsRecord, providers: Record<string, string>): RouteCell[] {
  return [
    { label: 'Public model', value: record.original_model || '—' },
    {
      label: 'Decision',
      value: record.decision ? formatInsightsDecisionName(record.decision) : 'Unresolved',
    },
    {
      label: 'Model',
      value: record.selected_model ? (
        <ModelValue name={record.selected_model} providers={providers} />
      ) : (
        'No backend'
      ),
    },
    { label: 'Latency', value: formatLatency(record.duration_ms) },
  ]
}

function WatchStep({
  model,
  latest,
  replay,
  providers,
}: {
  model: string
  latest: InsightsRecord | null
  replay: ReturnType<typeof useGetStartedData>['replay']
  providers: Record<string, string>
}) {
  const [phase, setPhase] = useState<'idle' | 'sending' | 'done' | 'failed'>('idle')
  const [result, setResult] = useState<TestRequestResult | null>(null)
  const [error, setError] = useState<PlaygroundErrorPresentation | null>(null)

  const send = async () => {
    setPhase('sending')
    setError(null)
    try {
      setResult(await sendGatewayTestRequest(model))
      setPhase('done')
    } catch (failure) {
      setError(playgroundErrorPresentation(failure))
      setPhase('failed')
    }
  }

  const replayId = result?.headers[X_VSR_REPLAY_ID]
  const traceId = result ? replayId : latest?.id
  const cells = result
    ? resultCells(result, providers)
    : latest
      ? recordCells(latest, providers)
      : null
  const headline =
    phase === 'sending'
      ? 'Routing your test request…'
      : result
        ? `Routed in ${formatLatency(result.latencyMs)}`
        : latest
          ? `Last request ${timeAgo(latest.timestamp)}`
          : replay.state === 'unavailable'
            ? replay.reason === 'replay_off'
              ? 'Router Replay is not recording requests'
              : replay.reason === 'forbidden'
                ? 'Your role cannot read recorded requests'
                : 'The router is not reachable'
            : 'Listening for your first request…'
  const tone = result || latest ? 'success' : replay.state === 'unavailable' ? 'warning' : 'info'

  return (
    <Card className="overflow-hidden" data-testid="get-started-watch">
      <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3">
        <StatusDot tone={tone} pulse={!result && !latest && replay.state !== 'unavailable'} />
        <span className="text-[13px] font-medium" aria-live="polite">
          {headline}
        </span>
        <Button
          size="sm"
          className="ml-auto"
          onClick={() => void send()}
          loading={phase === 'sending'}
          variant={result ? 'outline' : 'default'}
        >
          {phase === 'sending' ? null : <Play />}
          {result ? 'Send another' : 'Send test request'}
        </Button>
      </div>
      {error ? (
        <div className="border-b border-border px-4 py-3">
          <Alert tone="danger" title="The test request failed">
            <span className="block">{error.message}</span>
            {error.technicalDetails ? (
              <span className="mt-1 block whitespace-pre-wrap font-mono text-[11.5px] text-subtle-foreground">
                {error.technicalDetails}
              </span>
            ) : null}
          </Alert>
        </div>
      ) : null}
      <AnimatePresence mode="wait" initial={false}>
        {cells ? (
          <motion.div
            key={result ? 'result' : `record-${latest?.id}`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.28, ease: EASE }}
            className="grid grid-cols-1 divide-y divide-border sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4 lg:divide-x"
          >
            {cells.map((cell) => (
              <div key={cell.label} className="flex min-w-0 flex-col gap-1 px-4 py-3">
                <span className="text-xs text-subtle-foreground">{cell.label}</span>
                <span className="min-w-0 truncate text-[13px] font-medium">{cell.value}</span>
              </div>
            ))}
          </motion.div>
        ) : (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-start gap-3 px-4 py-5 text-[13px] text-muted-foreground"
          >
            <CircleDot className="mt-0.5 size-4 shrink-0 text-info" />
            <span>
              {replay.state === 'unavailable' && replay.reason === 'replay_off'
                ? 'Turn on the router_replay plugin for a recipe to see requests here; the test request still shows its route.'
                : replay.state === 'unavailable' && replay.reason === 'router_unreachable'
                  ? 'Start the router, then send a test request. Its route appears here.'
                  : 'Requests appear here as soon as the router records them.'}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
      {result?.answer ? (
        <div className="border-t border-border px-4 py-3 text-[13px] text-muted-foreground">
          <span className="mr-2 text-xs text-subtle-foreground">Answer</span>
          <span className="line-clamp-2">{result.answer}</span>
        </div>
      ) : null}
      {traceId ? (
        <div className="border-t border-border bg-surface-subtle/70 px-4 py-2.5 text-right">
          <Link
            to={getInsightsRecordPath(traceId)}
            className="inline-flex items-center gap-1 text-[13px] font-medium text-primary hover:underline"
          >
            Open the full trace
            <ArrowUpRight className="size-3.5" />
          </Link>
        </div>
      ) : null}
    </Card>
  )
}

const API_LABELS: Record<string, string> = {
  chat_completions: 'Chat Completions',
  openai: 'Chat Completions',
  responses: 'Responses',
  messages: 'Anthropic Messages',
  anthropic: 'Anthropic Messages',
  images: 'Images',
  systemone: 'System One',
}

function listenerApis(listeners: unknown, name?: string): string {
  const listener = Array.isArray(listeners)
    ? (listeners.find((item) => (item as { name?: unknown })?.name === name) as
        | { apis?: Record<string, unknown> }
        | undefined)
    : undefined
  const apis = listener?.apis ? Object.keys(listener.apis) : []
  const labels = apis.length
    ? apis.map((api) => API_LABELS[api] ?? api)
    : ['Chat Completions', 'Responses', 'Anthropic Messages']
  return [...new Set(labels)].join(' · ')
}

export default function GetStartedPage() {
  const data = useGetStartedData()
  const inventory = useDashboardObservation<ModelRuntimeInventory>(
    '/api/router/api/v1/inventory/model-runtime',
  )
  const [params, setParams] = useSearchParams()
  const [customBase, setCustomBase] = useState(readStoredBaseUrl)

  const publicModels = useMemo(() => listPublicModels(data.config), [data.config])
  const providers = useMemo(
    () => (data.config ? buildModelProviders(data.config as ConfigDocument) : {}),
    [data.config],
  )
  const chatModels = useMemo<PublicModel[]>(() => {
    const configured = publicModels.filter((model) => model.api === 'chat')
    if (configured.length || data.config) return configured
    return (data.liveModels ?? []).map((model, index) => ({
      name: model.id,
      recipe: model.recipe ?? 'default',
      api: 'chat',
      isDefault: index === 0,
      description: model.description || undefined,
      decisions: 0,
      signals: 0,
      providers: [],
    }))
  }, [data.config, data.liveModels, publicModels])
  const systemOneModels = publicModels.filter((model) => model.api === 'systemone')

  const requested = params.get('model')
  const selected =
    chatModels.find((model) => model.name === requested) ??
    chatModels.find((model) => model.isDefault) ??
    chatModels[0]
  const modelName = selected?.name ?? DEFAULT_PUBLIC_MODEL
  const agentParam = params.get('agent')
  const agent: AgentHarnessId =
    agentParam === 'codex' || agentParam === 'opencode' || agentParam === 'deepseek_harness'
      ? agentParam
      : 'claude_code'

  const endpoint = resolveGatewayEndpoint({
    listeners: data.config?.listeners,
    fallbackPort: data.setupPort,
    hostname: window.location.hostname,
    override: customBase,
    publicBaseUrl: data.publicBaseUrl,
    publicRequiresKey: data.publicRequiresKey,
  })
  const target = {
    origin: endpoint.origin,
    apiBase: endpoint.apiBase,
    model: modelName,
    requiresKey: endpoint.requiresKey,
  }
  const requestSnippets = buildRequestSnippets(target)
  const agentSnippets = buildAgentSnippets(target)
  const steps = buildStartChecklist({
    status: data.status,
    statusFailed: data.statusFailed,
    config: data.config,
    replay: data.replay,
    decisionModel: data.config
      ? decisionModelHealth(data.config, inventory.data, data.status?.models)
      : null,
  }).map((step) => (step.id === 'first-request' ? { ...step, to: '/observe/requests' } : step))
  const firstRequestSeen = data.replay.state === 'ready' && data.replay.count > 0

  const updateParam = (key: string, value: string | null) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (value) next.set(key, value)
        else next.delete(key)
        return next
      },
      { replace: true },
    )

  const changeBase = (value: string) => {
    writeStoredBaseUrl(value)
    setCustomBase(value)
  }

  return (
    <PageContainer>
      <PageHeader
        icon={<IconTile icon={Rocket} tone="blue" size="lg" />}
        title="Get started"
        description="Point any OpenAI-compatible client or coding agent at the router and pick a public model name."
        actions={
          <Button variant="outline" asChild>
            <a href={GUIDE_URL} target="_blank" rel="noreferrer">
              Read the guide
              <ArrowUpRight />
            </a>
          </Button>
        }
      />

      {data.configError && !data.loading ? (
        <Alert tone="warning" className="mb-6" title="The router configuration could not load">
          {chatModels.length
            ? `The models below are the ones the router advertises${
                endpoint.source === 'listener'
                  ? '; the endpoint uses the listener port setup reports'
                  : ''
              }.`
            : `The snippets use ${DEFAULT_PUBLIC_MODEL}${
                endpoint.source === 'listener' ? ' and the default listener' : ''
              }. ${data.configError}`}
        </Alert>
      ) : null}

      <div className="grid grid-cols-1 gap-8 xl:grid-cols-[minmax(0,1fr)_320px]">
        <ol className="flex min-w-0 flex-col">
          <Step
            index={1}
            title="Choose a public model"
            description="Apps send this name in the model field. Each one resolves to a recipe."
            done={Boolean(selected)}
          >
            {data.loading && chatModels.length === 0 ? (
              <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2 2xl:grid-cols-3">
                {[0, 1, 2].map((index) => (
                  <Skeleton key={index} className="h-[108px] rounded-xl" />
                ))}
              </div>
            ) : chatModels.length ? (
              <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2 2xl:grid-cols-3">
                {chatModels.map((model) => (
                  <ModelChoice
                    key={model.name}
                    model={model}
                    active={model.name === modelName}
                    onSelect={() => updateParam('model', model.isDefault ? null : model.name)}
                  />
                ))}
              </div>
            ) : (
              <p className="rounded-lg border border-dashed border-border-strong px-4 py-3 text-[13px] text-muted-foreground">
                No chat entrypoint is configured, so the snippets use{' '}
                <InlineCode>{DEFAULT_PUBLIC_MODEL}</InlineCode>.{' '}
                <Link to="/routing/recipes" className="font-medium text-primary hover:underline">
                  Add an entrypoint
                </Link>
              </p>
            )}
            {systemOneModels.length ? (
              <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                <BrainCircuit className="size-3.5 text-chart-3" />
                {systemOneModels.map((model) => (
                  <InlineCode key={model.name}>{model.name}</InlineCode>
                ))}
                <span>
                  answer{systemOneModels.length === 1 ? 's' : ''} the System One decision API, not
                  chat.
                </span>
                <Link
                  to="/models/decision?tab=playground"
                  className="font-medium text-primary hover:underline"
                >
                  Try System One
                </Link>
              </p>
            ) : null}
          </Step>

          <Step
            index={2}
            title="Copy the endpoint"
            description="Clients call the router's gateway, not the dashboard address."
            done
          >
            <EndpointStep
              apiBase={endpoint.apiBase}
              model={modelName}
              listener={endpoint.listener}
              source={endpoint.source}
              publicDeclared={Boolean(data.publicBaseUrl)}
              ready={data.setupSettled}
              requiresKey={endpoint.requiresKey}
              onCustomBase={changeBase}
            />
          </Step>

          <Step
            index={3}
            title="Send a request"
            description={
              <>
                Every response carries headers that explain its route, such as{' '}
                <InlineCode>{X_VSR_SELECTED_MODEL}</InlineCode>.
              </>
            }
          >
            <CodeBlock tabs={requestSnippets} />
          </Step>

          <Step
            index={4}
            title="Connect a coding agent"
            description="Route every call an agent makes, its subagents included, through the router."
          >
            <AgentStep
              snippets={agentSnippets}
              agent={agent}
              onAgent={(next) => updateParam('agent', next === 'claude_code' ? null : next)}
            />
          </Step>

          <Step
            index={5}
            id="watch-it-route"
            title="Watch it route"
            description="Send a test request from here, or run a snippet above; it shows up as the router records it."
            done={firstRequestSeen}
            last
          >
            <WatchStep
              model={modelName}
              latest={data.latest}
              replay={data.replay}
              providers={providers}
            />
          </Step>
        </ol>

        <aside className="flex flex-col gap-4 xl:sticky xl:top-6 xl:self-start">
          <StartChecklistCard steps={steps} title="Setup checklist" />
          <Card>
            <CardHeader>
              <CardTitle>Connection</CardTitle>
              <CardDescription>What a client needs to know</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 pt-3 text-[13px]">
              {[
                {
                  label: 'Protocols',
                  value: listenerApis(data.config?.listeners, endpoint.listener),
                },
                {
                  label: 'Base URL',
                  value: endpoint.apiBase,
                  mono: true,
                  loading: !data.setupSettled,
                },
                { label: 'Streaming', value: 'Server-sent events' },
                { label: 'Recipe', value: selected?.recipe ?? 'default', mono: true },
                {
                  label: 'Client key',
                  value: endpoint.requiresKey ? 'Required, as a bearer token' : 'Not required',
                  loading: !data.setupSettled,
                },
              ].map((row) => (
                <div key={row.label} className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-xs text-subtle-foreground">{row.label}</span>
                  {row.loading ? (
                    <Skeleton className="my-0.5 h-4 w-44" />
                  ) : (
                    <span className={cn('break-words', row.mono && 'font-mono text-[12.5px]')}>
                      {row.value}
                    </span>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Response headers</CardTitle>
              <CardDescription>Read these to see how a request was routed</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-2.5 pt-3">
              {[
                [X_VSR_SELECTED_MODEL, 'The backend model that answered'],
                [X_VSR_SELECTED_DECISION, 'The decision that matched'],
                [X_VSR_SELECTED_RECIPE, 'The recipe behind the model name'],
                [X_VSR_COST, 'What the request cost, on buffered responses'],
                [X_VSR_REPLAY_ID, 'The record to open in Requests'],
              ].map(([header, meaning]) => (
                <div key={header} className="flex flex-col gap-0.5">
                  <InlineCode className="w-fit">{header}</InlineCode>
                  <span className="text-xs text-muted-foreground">{meaning}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </aside>
      </div>
    </PageContainer>
  )
}
