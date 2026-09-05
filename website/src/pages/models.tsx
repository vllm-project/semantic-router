import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import Layout from '@theme/Layout'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Ai2 from '@lobehub/icons/es/Ai2/components/Mono'
import Ai21 from '@lobehub/icons/es/Ai21/components/Mono'
import Bedrock from '@lobehub/icons/es/Bedrock/components/Mono'
import Baidu from '@lobehub/icons/es/Baidu/components/Mono'
import ByteDance from '@lobehub/icons/es/ByteDance/components/Mono'
import Cerebras from '@lobehub/icons/es/Cerebras/components/Mono'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import Cohere from '@lobehub/icons/es/Cohere/components/Mono'
import CometAPI from '@lobehub/icons/es/CometAPI/components/Mono'
import DeepInfra from '@lobehub/icons/es/DeepInfra/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Featherless from '@lobehub/icons/es/Featherless/components/Mono'
import Fireworks from '@lobehub/icons/es/Fireworks/components/Mono'
import Friendli from '@lobehub/icons/es/Friendli/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Grok from '@lobehub/icons/es/Grok/components/Mono'
import Groq from '@lobehub/icons/es/Groq/components/Mono'
import HuggingFace from '@lobehub/icons/es/HuggingFace/components/Mono'
import InternLM from '@lobehub/icons/es/InternLM/components/Mono'
import Kimi from '@lobehub/icons/es/Kimi/components/Mono'
import LG from '@lobehub/icons/es/LG/components/Mono'
import LmStudio from '@lobehub/icons/es/LmStudio/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Mono'
import Microsoft from '@lobehub/icons/es/Microsoft/components/Mono'
import Minimax from '@lobehub/icons/es/Minimax/components/Mono'
import Mistral from '@lobehub/icons/es/Mistral/components/Mono'
import Nova from '@lobehub/icons/es/Nova/components/Mono'
import Nebius from '@lobehub/icons/es/Nebius/components/Mono'
import Novita from '@lobehub/icons/es/Novita/components/Mono'
import Nvidia from '@lobehub/icons/es/Nvidia/components/Mono'
import Ollama from '@lobehub/icons/es/Ollama/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import OpenRouter from '@lobehub/icons/es/OpenRouter/components/Mono'
import Perplexity from '@lobehub/icons/es/Perplexity/components/Mono'
import Qwen from '@lobehub/icons/es/Qwen/components/Mono'
import Snowflake from '@lobehub/icons/es/Snowflake/components/Mono'
import SambaNova from '@lobehub/icons/es/SambaNova/components/Mono'
import Stepfun from '@lobehub/icons/es/Stepfun/components/Mono'
import TII from '@lobehub/icons/es/TII/components/Mono'
import Tencent from '@lobehub/icons/es/Tencent/components/Mono'
import Together from '@lobehub/icons/es/Together/components/Mono'
import Upstage from '@lobehub/icons/es/Upstage/components/Mono'
import Vercel from '@lobehub/icons/es/Vercel/components/Mono'
import Vllm from '@lobehub/icons/es/Vllm/components/Mono'
import XiaomiMiMo from '@lobehub/icons/es/XiaomiMiMo/components/Mono'
import Yi from '@lobehub/icons/es/Yi/components/Mono'
import Xinference from '@lobehub/icons/es/Xinference/components/Mono'
import Zhipu from '@lobehub/icons/es/Zhipu/components/Mono'
import catalogDocument from '../../static/model-catalog/catalog.json'
import {
  availableModelHubBenchmarkMetrics,
  availableModelHubBenchmarkProfiles,
  modelHubBenchmarkSelectionCounts,
  modelHubChartHues,
  preferredModelHubBenchmarkSelection,
} from '../data/modelHubBenchmarkSupport'
import {
  modelHubEvaluationConditionLabel,
  modelHubEvaluationDateLabel,
} from '../data/modelHubEvaluationLabel'
import styles from './models.module.css'

type SupportTier = 'native' | 'compatible' | 'runtime'
type ModelView = 'cards' | 'table'
type ModelKind = 'physical' | 'virtual'
type Distribution = 'proprietary_api' | 'open_weights' | 'router_recipe'
type ModelRelationship =
  'first_party' | 'managed_cloud' | 'gateway' | 'self_hosted'

interface CatalogProtocol {
  id: string
  display_name: string
  operations: Array<{ id: string, method: string, path: string }>
}

interface CatalogPresentation {
  logo: string
  monogram: string
  monochrome: boolean
}

interface CatalogModelBinding {
  catalog: string
  relationship: ModelRelationship
  id: string
  protocols: string[]
  lifecycle: string
}

interface CatalogProvider {
  id: string
  display_name: string
  description: string
  category: 'start_here' | 'model_api' | 'private_runtime'
  support_tier: SupportTier
  protocols: string[]
  default_protocol: string
  supported_operations: string[]
  path_overrides?: Record<string, string>
  auth: { strategy: string }
  presentation: CatalogPresentation
  conformance: { status: string, verified_at?: string }
  models?: CatalogModelBinding[]
}

interface VirtualRole {
  name: string
  required: boolean
  minimum_candidates: number
  recommended_pool: string[]
  traits: string[]
}

interface CatalogModel {
  id: string
  display_name: string
  description: string
  kind: ModelKind
  publisher: string
  presentation: CatalogPresentation
  distribution: { type: Distribution, source: string, license?: string }
  family: string
  parameter_size?: string
  released_at?: string
  lifecycle: 'experimental' | 'active' | 'deprecated' | 'removed'
  limits?: { context_window_size?: number, max_output_tokens?: number }
  capabilities: string[]
  modalities: { input: string[], output: string[] }
  reasoning_family?: string
  verification: { status: string, verified_at: string, source?: string }
  entrypoint?: string
  recipe?: string
  policy_version?: string
  roles?: VirtualRole[]
  traits?: string[]
}

interface BenchmarkMetric {
  id: string
  direction: 'higher_is_better' | 'lower_is_better'
  range: [number, number]
  unit: string
}

interface CatalogBenchmark {
  id: string
  display_name: string
  domain: string
  default_profile: string
  source?: string
  profiles: Array<{ id: string, display_name: string, description?: string }>
  metrics: BenchmarkMetric[]
}

interface CatalogEvaluation {
  id: string
  model: string
  benchmark: string
  benchmark_profile: string
  reasoning_effort: string
  status: 'available' | 'missing'
  measured_at?: string
  observed_at?: string
  metrics?: Record<string, number | null>
  subject: Record<string, unknown>
  evidence: {
    provenance:
      'vendor_claimed' | 'third_party' | 'vllm_sr_reproduced' | 'operator'
    verification: string
    source?: string
  }
}

interface CatalogSnapshot {
  protocols: CatalogProtocol[]
  providers: CatalogProvider[]
  reasoning_families: Array<{ id: string, levels: string[], default: string }>
  models: CatalogModel[]
  benchmarks: CatalogBenchmark[]
  evaluations: CatalogEvaluation[]
}

interface BenchmarkRow {
  evaluation: CatalogEvaluation
  model: CatalogModel
  value: number
}

const catalog = catalogDocument as CatalogSnapshot
const MODEL_PAGE_SIZE = 12
const BENCHMARK_PAGE_SIZE = 12
const PROVIDER_PAGE_SIZE = 12

const tierLabel: Record<SupportTier, string> = {
  native: 'Native',
  compatible: 'Compatible',
  runtime: 'Runtime',
}

const categoryLabel: Record<CatalogProvider['category'], string> = {
  start_here: 'Start here',
  model_api: 'Model APIs',
  private_runtime: 'Private runtimes',
}

const distributionLabel: Record<Distribution, string> = {
  open_weights: 'Open weights',
  proprietary_api: 'Proprietary API',
  router_recipe: 'Router recipe',
}

const relationshipLabel: Record<ModelRelationship, string> = {
  first_party: 'First-party',
  managed_cloud: 'Managed cloud',
  gateway: 'Gateway',
  self_hosted: 'Self-hosted',
}

const packageIcons: Record<string, typeof OpenAI> = {
  ai2: Ai2,
  ai21: Ai21,
  anthropic: Claude,
  baidu: Baidu,
  bedrock: Bedrock,
  bytedance: ByteDance,
  cerebras: Cerebras,
  cohere: Cohere,
  cometapi: CometAPI,
  deepinfra: DeepInfra,
  deepseek: DeepSeek,
  featherless: Featherless,
  fireworks: Fireworks,
  friendli: Friendli,
  gemini: Gemini,
  google: Gemini,
  groq: Groq,
  huggingface: HuggingFace,
  internlm: InternLM,
  exaone: LG,
  lg: LG,
  lmstudio: LmStudio,
  meta: Meta,
  microsoft: Microsoft,
  minimax: Minimax,
  mistral: Mistral,
  moonshot: Kimi,
  nebius: Nebius,
  novita: Novita,
  nova: Nova,
  nvidia: Nvidia,
  ollama: Ollama,
  openai: OpenAI,
  openrouter: OpenRouter,
  perplexity: Perplexity,
  qwen: Qwen,
  sambanova: SambaNova,
  snowflake: Snowflake,
  stepfun: Stepfun,
  tencent: Tencent,
  together: Together,
  tii: TII,
  upstage: Upstage,
  vercel: Vercel,
  vllm: Vllm,
  xai: Grok,
  xiaomimimo: XiaomiMiMo,
  xinference: Xinference,
  yi: Yi,
  zai: Zhipu,
}

const readable = (value: string) => value.replace(/_/g, ' ')

const formatTokens = (value?: number) => {
  if (!value) return '—'
  if (value >= 1_000_000) {
    const millions = value / 1_000_000
    return `${Number.isInteger(millions) ? millions : millions.toFixed(2)}M`
  }
  if (value >= 1_000) {
    const thousands = value / 1_000
    return `${Number.isInteger(thousands) ? thousands : thousands.toFixed(1)}K`
  }
  return String(value)
}

const formatDate = (value?: string) => {
  if (!value) return 'Not published'
  return new Intl.DateTimeFormat('en', {
    year: 'numeric',
    month: 'short',
  }).format(new Date(`${value}T00:00:00Z`))
}

const formatMetric = (value: number, metric: BenchmarkMetric) => {
  if (metric.unit === 'proportion') return `${(value * 100).toFixed(1)}%`
  if (metric.unit === 'elo') return Math.round(value).toLocaleString()
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

const chartDomain = (
  values: number[],
  metric: BenchmarkMetric,
): [number, number] => {
  if (metric.unit === 'proportion' || metric.unit === 'fraction')
    return metric.range
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum))
    return metric.range
  return minimum === maximum ? [minimum - 1, maximum + 1] : [minimum, maximum]
}

function CatalogMark({
  presentation,
  large = false,
}: {
  presentation: CatalogPresentation
  large?: boolean
}) {
  const logo = presentation.logo ?? ''
  const packageID = logo.startsWith('package:')
    ? logo.slice('package:'.length)
    : ''
  const publicLogo = logo.startsWith('public:')
    ? logo.slice('public:'.length)
    : ''
  const resolvedPublicLogo = useBaseUrl(publicLogo)
  const directLogo = publicLogo
    ? resolvedPublicLogo
    : logo.startsWith('url:')
      ? logo.slice('url:'.length)
      : ''
  const [logoFailed, setLogoFailed] = useState(false)
  useEffect(() => setLogoFailed(false), [directLogo])
  const Icon = packageIcons[packageID]
  return (
    <span
      className={`${styles.catalogMark} ${large ? styles.catalogMarkLarge : ''}`}
      aria-hidden="true"
    >
      {Icon
        ? (
            <Icon size={large ? 29 : 21} />
          )
        : directLogo && !logoFailed
          ? (
              <img src={directLogo} alt="" onError={() => setLogoFailed(true)} />
            )
          : (
              presentation.monogram
            )}
    </span>
  )
}

export default function ModelsPage() {
  const [modelSearch, setModelSearch] = useState('')
  const [modelKind, setModelKind] = useState<'all' | ModelKind>('all')
  const [modelDistribution, setModelDistribution] = useState<
    'all' | Distribution
  >('all')
  const [modelPublisher, setModelPublisher] = useState('all')
  const [modelLifecycle, setModelLifecycle] = useState<
    'supported' | 'all' | CatalogModel['lifecycle']
  >('supported')
  const [modelSort, setModelSort] = useState<'newest' | 'name' | 'context'>(
    'newest',
  )
  const [modelView, setModelView] = useState<ModelView>('cards')
  const [modelPage, setModelPage] = useState(1)
  const [selectedModelID, setSelectedModelID] = useState<string | null>(null)
  const [providerSearch, setProviderSearch] = useState('')
  const [providerTier, setProviderTier] = useState<'all' | SupportTier>('all')
  const [providerPage, setProviderPage] = useState(1)

  const protocols = useMemo(
    () => new Map(catalog.protocols.map(protocol => [protocol.id, protocol])),
    [],
  )
  const modelByID = useMemo(
    () => new Map(catalog.models.map(model => [model.id, model])),
    [],
  )
  const providersByModel = useMemo(() => {
    const bindings = new Map<
      string,
      Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
    >()
    for (const provider of catalog.providers) {
      for (const binding of provider.models ?? []) {
        bindings.set(binding.catalog, [
          ...(bindings.get(binding.catalog) ?? []),
          { provider, binding },
        ])
      }
    }
    return bindings
  }, [])
  const reasoningFamilies = useMemo(
    () =>
      new Map(catalog.reasoning_families.map(family => [family.id, family])),
    [],
  )
  const evaluationsByModel = useMemo(() => {
    const groups = new Map<string, CatalogEvaluation[]>()
    for (const evaluation of catalog.evaluations) {
      if (evaluation.status !== 'available') continue
      groups.set(evaluation.model, [
        ...(groups.get(evaluation.model) ?? []),
        evaluation,
      ])
    }
    return groups
  }, [])
  const publishers = useMemo(
    () =>
      Array.from(
        new Set(
          catalog.models
            .filter(model => model.kind === 'physical')
            .map(model => model.publisher),
        ),
      ).sort((left, right) => left.localeCompare(right)),
    [],
  )

  const models = useMemo(() => {
    const query = modelSearch.trim().toLocaleLowerCase()
    return catalog.models
      .filter((model) => {
        const matchesLifecycle
          = modelLifecycle === 'all'
            || (modelLifecycle === 'supported'
              ? model.lifecycle === 'active' || model.lifecycle === 'experimental'
              : model.lifecycle === modelLifecycle)
        return (
          (modelKind === 'all' || model.kind === modelKind)
          && (modelDistribution === 'all'
            || model.distribution.type === modelDistribution)
          && (modelPublisher === 'all' || model.publisher === modelPublisher)
          && matchesLifecycle
          && (!query
            || `${model.display_name} ${model.id} ${model.publisher} ${model.family} ${model.capabilities.join(' ')}`
              .toLocaleLowerCase()
              .includes(query))
        )
      })
      .sort((left, right) => {
        if (modelSort === 'name')
          return left.display_name.localeCompare(right.display_name)
        if (modelSort === 'context') {
          return (
            (right.limits?.context_window_size ?? 0)
            - (left.limits?.context_window_size ?? 0)
            || left.display_name.localeCompare(right.display_name)
          )
        }
        return (
          (right.released_at ?? '').localeCompare(left.released_at ?? '')
          || left.display_name.localeCompare(right.display_name)
        )
      })
  }, [
    modelDistribution,
    modelKind,
    modelLifecycle,
    modelPublisher,
    modelSearch,
    modelSort,
  ])

  useEffect(
    () => setModelPage(1),
    [
      modelDistribution,
      modelKind,
      modelLifecycle,
      modelPublisher,
      modelSearch,
      modelSort,
    ],
  )
  const modelPageCount = Math.max(
    1,
    Math.ceil(models.length / MODEL_PAGE_SIZE),
  )
  const pageModels = models.slice(
    (modelPage - 1) * MODEL_PAGE_SIZE,
    modelPage * MODEL_PAGE_SIZE,
  )
  const selectedModel = catalog.models.find(
    model => model.id === selectedModelID,
  )
  const selectedProviders = selectedModel
    ? (providersByModel.get(selectedModel.id) ?? [])
    : []
  const selectedEvaluations = selectedModel
    ? (evaluationsByModel.get(selectedModel.id) ?? []).sort((left, right) =>
        left.benchmark.localeCompare(right.benchmark),
      )
    : []
  const selectedFamily = selectedModel?.reasoning_family
    ? reasoningFamilies.get(selectedModel.reasoning_family)
    : undefined

  const closeModelDetail = useCallback(() => setSelectedModelID(null), [])

  const benchmarkCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const evaluation of catalog.evaluations) {
      if (evaluation.status === 'available') {
        counts.set(
          evaluation.benchmark,
          (counts.get(evaluation.benchmark) ?? 0) + 1,
        )
      }
    }
    return counts
  }, [])
  const benchmarkSelectionCounts = useMemo(
    () => modelHubBenchmarkSelectionCounts(catalog.evaluations),
    [],
  )
  const benchmarksWithData = useMemo(
    () =>
      catalog.benchmarks
        .filter(benchmark => (benchmarkCounts.get(benchmark.id) ?? 0) > 0)
        .sort(
          (left, right) =>
            (benchmarkCounts.get(right.id) ?? 0)
            - (benchmarkCounts.get(left.id) ?? 0),
        ),
    [benchmarkCounts],
  )
  const defaultBenchmarkSelection = preferredModelHubBenchmarkSelection(
    benchmarkSelectionCounts,
  )
  const [benchmarkID, setBenchmarkID] = useState(
    () =>
      defaultBenchmarkSelection?.benchmark ?? benchmarksWithData[0]?.id ?? '',
  )
  const selectedBenchmark
    = benchmarksWithData.find(benchmark => benchmark.id === benchmarkID)
      ?? benchmarksWithData[0]
  const initialBenchmarkSelection = preferredModelHubBenchmarkSelection(
    benchmarkSelectionCounts,
    selectedBenchmark?.id ?? '',
  )
  const [benchmarkProfile, setBenchmarkProfile] = useState(
    initialBenchmarkSelection?.profile
    ?? selectedBenchmark?.default_profile
    ?? '',
  )
  const [benchmarkMetric, setBenchmarkMetric] = useState(
    initialBenchmarkSelection?.metric
    ?? selectedBenchmark?.metrics[0]?.id
    ?? '',
  )
  const [benchmarkPage, setBenchmarkPage] = useState(1)
  const availableProfiles = selectedBenchmark
    ? availableModelHubBenchmarkProfiles(
        benchmarkSelectionCounts,
        selectedBenchmark.id,
      )
    : new Set<string>()
  const activeProfile = availableProfiles.has(benchmarkProfile)
    ? benchmarkProfile
    : (initialBenchmarkSelection?.profile ?? '')
  const availableMetrics = selectedBenchmark
    ? availableModelHubBenchmarkMetrics(
        benchmarkSelectionCounts,
        selectedBenchmark.id,
        activeProfile,
      )
    : new Set<string>()
  const metric
    = selectedBenchmark?.metrics.find(
      candidate =>
        candidate.id === benchmarkMetric && availableMetrics.has(candidate.id),
    )
    ?? selectedBenchmark?.metrics.find(candidate =>
      availableMetrics.has(candidate.id),
    )
  const benchmarkRows = useMemo(() => {
    if (!selectedBenchmark || !metric) return []
    const uniqueRows = new Map<string, BenchmarkRow>()
    for (const evaluation of catalog.evaluations) {
      const value = evaluation.metrics?.[metric.id]
      const model = modelByID.get(evaluation.model)
      if (
        evaluation.status !== 'available'
        || evaluation.benchmark !== selectedBenchmark.id
        || evaluation.benchmark_profile !== activeProfile
        || typeof value !== 'number'
        || !model
      )
        continue
      uniqueRows.set(`${model.id}:${evaluation.reasoning_effort}`, {
        evaluation,
        model,
        value,
      })
    }
    return Array.from(uniqueRows.values()).sort((left, right) =>
      metric.direction === 'lower_is_better'
        ? left.value - right.value
        : right.value - left.value,
    )
  }, [activeProfile, metric, modelByID, selectedBenchmark])
  const benchmarkPageCount = Math.max(
    1,
    Math.ceil(benchmarkRows.length / BENCHMARK_PAGE_SIZE),
  )
  const pageBenchmarkRows = benchmarkRows.slice(
    (benchmarkPage - 1) * BENCHMARK_PAGE_SIZE,
    benchmarkPage * BENCHMARK_PAGE_SIZE,
  )
  const benchmarkChartDomain = metric
    ? chartDomain(
        benchmarkRows.map(row => row.value),
        metric,
      )
    : ([0, 1] as [number, number])
  const benchmarkChartHues = useMemo(
    () => modelHubChartHues(benchmarkRows.map(row => row.model.id)),
    [benchmarkRows],
  )

  const providers = useMemo(() => {
    const query = providerSearch.trim().toLocaleLowerCase()
    return catalog.providers.filter(
      provider =>
        (providerTier === 'all' || provider.support_tier === providerTier)
        && (!query
          || `${provider.display_name} ${provider.id} ${provider.description}`
            .toLocaleLowerCase()
            .includes(query)),
    )
  }, [providerSearch, providerTier])
  useEffect(() => setProviderPage(1), [providerSearch, providerTier])
  const providerPageCount = Math.max(
    1,
    Math.ceil(providers.length / PROVIDER_PAGE_SIZE),
  )
  const pageProviders = providers.slice(
    (providerPage - 1) * PROVIDER_PAGE_SIZE,
    providerPage * PROVIDER_PAGE_SIZE,
  )

  const physicalModels = catalog.models.filter(
    model => model.kind === 'physical',
  ).length
  const virtualModels = catalog.models.length - physicalModels
  const physicalPublishers = new Set(
    catalog.models
      .filter(model => model.kind === 'physical')
      .map(model => model.publisher),
  ).size

  const chooseBenchmark = (id: string) => {
    const benchmark = benchmarksWithData.find(
      candidate => candidate.id === id,
    )
    const preferred = preferredModelHubBenchmarkSelection(
      benchmarkSelectionCounts,
      id,
    )
    setBenchmarkID(id)
    setBenchmarkProfile(
      preferred?.profile
      ?? benchmark?.default_profile
      ?? benchmark?.profiles[0]?.id
      ?? '',
    )
    setBenchmarkMetric(preferred?.metric ?? benchmark?.metrics[0]?.id ?? '')
    setBenchmarkPage(1)
  }

  return (
    <Layout
      title="Model Hub"
      description="Discover the built-in models, virtual recipes, providers, protocols, and benchmark evidence available in vLLM Semantic Router."
    >
      <main className={styles.page}>
        <header className={styles.hero}>
          <div className={styles.heroGlow} aria-hidden="true" />
          <div className={styles.heroCopy}>
            <span className={styles.eyebrow}>
              vLLM Semantic Router · Model Hub
            </span>
            <h1>Discover the right models for your routing stack.</h1>
            <p>
              Browse
              {' '}
              {physicalModels}
              {' '}
              single models from
              {' '}
              {physicalPublishers}
              {' '}
              mainstream creators, centered on recent generations and
              representative lines. Provider contracts, reasoning controls,
              protocol support, and benchmark evidence all come from the same
              validated catalog used by the Router and Dashboard.
            </p>
            <nav className={styles.jumpNav} aria-label="Model Hub sections">
              <a href="#models">Browse models</a>
              <a href="#benchmarks">Explore benchmarks</a>
              <a href="#providers">Provider registry</a>
            </nav>
          </div>
          <div className={styles.stats} aria-label="Catalog summary">
            <Stat value={physicalModels} label="single models" />
            <Stat value={virtualModels} label="virtual recipes" />
            <Stat value={physicalPublishers} label="model creators" />
            <Stat value={catalog.providers.length} label="provider contracts" />
          </div>
        </header>

        <section
          id="models"
          className={styles.section}
          aria-labelledby="models-heading"
        >
          <SectionHeading
            id="models-heading"
            eyebrow="Catalog"
            title="Built-in models"
            description="This is a discovery hub, not a global ranking. Find a model by its architecture and runtime fit, then inspect the evidence that exists for it."
          />
          <div className={styles.controlSurface}>
            <label className={styles.searchControl}>
              <span className={styles.srOnly}>Search models</span>
              <span className={styles.searchIcon} aria-hidden="true">
                ⌕
              </span>
              <input
                type="search"
                value={modelSearch}
                onChange={event => setModelSearch(event.target.value)}
                placeholder="Search model, creator, family, or capability"
              />
            </label>
            <div className={styles.filterGrid}>
              <SelectControl
                label="Type"
                value={modelKind}
                options={[
                  ['all', 'All models'],
                  ['physical', 'Single models'],
                  ['virtual', 'Virtual recipes'],
                ]}
                onChange={value => setModelKind(value as 'all' | ModelKind)}
              />
              <SelectControl
                label="Distribution"
                value={modelDistribution}
                options={[
                  ['all', 'All distributions'],
                  ['open_weights', 'Open weights'],
                  ['proprietary_api', 'Proprietary API'],
                  ['router_recipe', 'Router recipe'],
                ]}
                onChange={value =>
                  setModelDistribution(value as 'all' | Distribution)}
              />
              <SelectControl
                label="Model creator"
                value={modelPublisher}
                options={[
                  ['all', 'All creators'],
                  ...publishers.map(
                    publisher => [publisher, publisher] as [string, string],
                  ),
                ]}
                onChange={setModelPublisher}
              />
              <SelectControl
                label="Lifecycle"
                value={modelLifecycle}
                options={[
                  ['supported', 'Supported'],
                  ['active', 'Active'],
                  ['experimental', 'Experimental'],
                  ['deprecated', 'Deprecated'],
                  ['removed', 'Removed'],
                  ['all', 'All states'],
                ]}
                onChange={value =>
                  setModelLifecycle(value as typeof modelLifecycle)}
              />
              <SelectControl
                label="Sort by"
                value={modelSort}
                options={[
                  ['newest', 'Newest release'],
                  ['name', 'Name A–Z'],
                  ['context', 'Context window'],
                ]}
                onChange={value => setModelSort(value as typeof modelSort)}
              />
            </div>
          </div>

          <div className={styles.resultsBar}>
            <span>
              <strong>{models.length}</strong>
              {' '}
              matching models
            </span>
            <ViewToggle value={modelView} onChange={setModelView} />
          </div>

          {modelView === 'cards'
            ? (
                <div className={styles.modelGrid}>
                  {pageModels.map(model => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      evaluationCount={
                        evaluationsByModel.get(model.id)?.length ?? 0
                      }
                      providerCount={providersByModel.get(model.id)?.length ?? 0}
                      onSelect={() => setSelectedModelID(model.id)}
                    />
                  ))}
                </div>
              )
            : (
                <ModelTable
                  models={pageModels}
                  providersByModel={providersByModel}
                  evaluationsByModel={evaluationsByModel}
                  onSelect={setSelectedModelID}
                />
              )}
          {models.length === 0
            ? (
                <EmptyState
                  title="No models found"
                  body="Try removing a filter or using a broader search."
                />
              )
            : null}
          <Pagination
            page={modelPage}
            pageCount={modelPageCount}
            total={models.length}
            pageSize={MODEL_PAGE_SIZE}
            label="models"
            onChange={setModelPage}
          />
        </section>

        <section
          id="benchmarks"
          className={styles.section}
          aria-labelledby="benchmarks-heading"
        >
          <SectionHeading
            id="benchmarks-heading"
            eyebrow="Benchmark explorer"
            title="Compare one measurement at a time"
            description="Select a benchmark, profile, and metric. Only exact model and evaluation-condition measurements are shown; missing evidence is never treated as zero."
          />
          {selectedBenchmark && metric
            ? (
                <div className={styles.benchmarkShell}>
                  <div className={styles.benchmarkControls}>
                    <SelectControl
                      label="Benchmark"
                      value={selectedBenchmark.id}
                      options={benchmarksWithData.map(benchmark => [
                        benchmark.id,
                        `${benchmark.display_name} (${benchmarkCounts.get(benchmark.id) ?? 0} records across setups)`,
                      ])}
                      onChange={chooseBenchmark}
                    />
                    <SelectControl
                      label="Profile"
                      value={activeProfile}
                      options={selectedBenchmark.profiles
                        .filter(profile => availableProfiles.has(profile.id))
                        .map(profile => [profile.id, profile.display_name])}
                      onChange={(value) => {
                        const preferred = preferredModelHubBenchmarkSelection(
                          benchmarkSelectionCounts,
                          selectedBenchmark.id,
                          value,
                        )
                        setBenchmarkProfile(value)
                        setBenchmarkMetric(preferred?.metric ?? '')
                        setBenchmarkPage(1)
                      }}
                    />
                    <SelectControl
                      label="Metric"
                      value={metric.id}
                      options={selectedBenchmark.metrics
                        .filter(item => availableMetrics.has(item.id))
                        .map(item => [item.id, readable(item.id)])}
                      onChange={(value) => {
                        setBenchmarkMetric(value)
                        setBenchmarkPage(1)
                      }}
                    />
                  </div>
                  <div className={styles.benchmarkMeta}>
                    <div>
                      <span>{readable(selectedBenchmark.domain)}</span>
                      <h3>{selectedBenchmark.display_name}</h3>
                      <p>
                        {
                          selectedBenchmark.profiles.find(
                            profile => profile.id === activeProfile,
                          )?.description
                        }
                      </p>
                    </div>
                    {selectedBenchmark.source
                      ? (
                          <a
                            href={selectedBenchmark.source}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Benchmark source ↗
                          </a>
                        )
                      : null}
                  </div>
                  <div className={styles.chartHeader}>
                    <span>Rank · model · evaluation condition</span>
                    <span>{`${readable(metric.id)} · ${readable(metric.unit)}`}</span>
                  </div>
                  <div
                    className={styles.barChart}
                    aria-label={`${selectedBenchmark.display_name} results`}
                  >
                    {pageBenchmarkRows.map((row, index) => (
                      <BenchmarkBar
                        key={`${row.model.id}:${row.evaluation.reasoning_effort}`}
                        row={row}
                        metric={metric}
                        domain={benchmarkChartDomain}
                        hue={benchmarkChartHues.get(row.model.id) ?? 0}
                        rank={(benchmarkPage - 1) * BENCHMARK_PAGE_SIZE + index + 1}
                      />
                    ))}
                  </div>
                  {benchmarkRows.length === 0
                    ? (
                        <EmptyState
                          title="No matching measurements"
                          body="Choose another profile or metric to see available model evidence."
                        />
                      )
                    : null}
                  <Pagination
                    page={benchmarkPage}
                    pageCount={benchmarkPageCount}
                    total={benchmarkRows.length}
                    pageSize={BENCHMARK_PAGE_SIZE}
                    label="measurements"
                    onChange={setBenchmarkPage}
                  />
                </div>
              )
            : (
                <EmptyState
                  title="No benchmark evidence yet"
                  body="Measurements will appear here as they are added to the catalog."
                />
              )}
        </section>

        <section
          id="providers"
          className={styles.section}
          aria-labelledby="providers-heading"
        >
          <SectionHeading
            id="providers-heading"
            eyebrow="Transport support matrix"
            title="Provider registry"
            description="Runtime endpoint contracts power Add Model and custom deployments. Curated model mappings are shown separately, so a provider with zero mappings is not presented as built-in model support."
          />
          <div className={styles.providerTools}>
            <label className={styles.searchControl}>
              <span className={styles.srOnly}>Search providers</span>
              <span className={styles.searchIcon} aria-hidden="true">
                ⌕
              </span>
              <input
                type="search"
                value={providerSearch}
                onChange={event => setProviderSearch(event.target.value)}
                placeholder="Search provider or protocol"
              />
            </label>
            <FilterButtons
              value={providerTier}
              options={[
                ['all', 'All'],
                ['native', 'Native'],
                ['compatible', 'Compatible'],
                ['runtime', 'Runtime'],
              ]}
              onChange={value =>
                setProviderTier(value as 'all' | SupportTier)}
            />
          </div>
          <div className={styles.providerGrid}>
            {pageProviders.map(provider => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                protocols={protocols}
              />
            ))}
          </div>
          {providers.length === 0
            ? (
                <EmptyState
                  title="No providers found"
                  body="Try a different name or support tier."
                />
              )
            : null}
          <Pagination
            page={providerPage}
            pageCount={providerPageCount}
            total={providers.length}
            pageSize={PROVIDER_PAGE_SIZE}
            label="providers"
            onChange={setProviderPage}
          />
        </section>

        {selectedModel
          ? (
              <ModelDetail
                model={selectedModel}
                providers={selectedProviders}
                evaluations={selectedEvaluations}
                benchmarks={catalog.benchmarks}
                family={selectedFamily}
                modelByID={modelByID}
                onClose={closeModelDetail}
              />
            )
          : null}
      </main>
    </Layout>
  )
}

function ModelCard({
  model,
  evaluationCount,
  providerCount,
  onSelect,
}: {
  model: CatalogModel
  evaluationCount: number
  providerCount: number
  onSelect: () => void
}) {
  return (
    <article className={styles.modelCard}>
      <button
        type="button"
        className={styles.cardButton}
        onClick={onSelect}
        aria-label={`Open ${model.display_name} details`}
      >
        <span className={styles.cardTopline}>
          <CatalogMark presentation={model.presentation} />
          <span className={styles.lifecycleDot} data-status={model.lifecycle}>
            {readable(model.lifecycle)}
          </span>
        </span>
        <span className={styles.cardTitle}>
          <small>{model.publisher}</small>
          <strong>{model.display_name}</strong>
          <code>{model.id}</code>
        </span>
        <span className={styles.cardDescription}>{model.description}</span>
        <span className={styles.cardMeta}>
          <span>
            <small>Context</small>
            <strong>{formatTokens(model.limits?.context_window_size)}</strong>
          </span>
          <span>
            <small>Benchmarks</small>
            <strong>{evaluationCount || '—'}</strong>
          </span>
          <span>
            <small>{model.kind === 'virtual' ? 'Roles' : 'Providers'}</small>
            <strong>
              {model.kind === 'virtual'
                ? (model.roles?.length ?? 0)
                : providerCount || '—'}
            </strong>
          </span>
        </span>
        <span className={styles.cardFooter}>
          <Badge
            value={model.distribution.type}
            label={distributionLabel[model.distribution.type]}
          />
          <span>
            {formatDate(model.released_at)}
            {' '}
            <b aria-hidden="true">↗</b>
          </span>
        </span>
      </button>
    </article>
  )
}

function ModelTable({
  models,
  providersByModel,
  evaluationsByModel,
  onSelect,
}: {
  models: CatalogModel[]
  providersByModel: Map<
    string,
    Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
  >
  evaluationsByModel: Map<string, CatalogEvaluation[]>
  onSelect: (id: string) => void
}) {
  return (
    <div className={styles.tableFrame}>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Distribution</th>
            <th>Context</th>
            <th>Providers</th>
            <th>Evidence</th>
            <th>Released</th>
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr key={model.id} onClick={() => onSelect(model.id)}>
              <td>
                <button
                  type="button"
                  className={`${styles.tableModel} ${styles.tableModelButton}`}
                  aria-label={`View details for ${model.display_name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    onSelect(model.id)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === ' ') event.preventDefault()
                  }}
                  onKeyUp={(event) => {
                    if (event.key === ' ') onSelect(model.id)
                  }}
                >
                  <CatalogMark presentation={model.presentation} />
                  <span>
                    <strong>{model.display_name}</strong>
                    <small>
                      {model.publisher}
                      {' '}
                      ·
                      {model.id}
                    </small>
                  </span>
                </button>
              </td>
              <td>
                <Badge
                  value={model.distribution.type}
                  label={distributionLabel[model.distribution.type]}
                />
              </td>
              <td>{formatTokens(model.limits?.context_window_size)}</td>
              <td>
                {model.kind === 'virtual'
                  ? `${model.roles?.length ?? 0} roles`
                  : (providersByModel.get(model.id)?.length ?? '—')}
              </td>
              <td>{evaluationsByModel.get(model.id)?.length ?? '—'}</td>
              <td>{formatDate(model.released_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function BenchmarkBar({
  row,
  metric,
  domain,
  hue,
  rank,
}: {
  row: BenchmarkRow
  metric: BenchmarkMetric
  domain: [number, number]
  hue: number
  rank: number
}) {
  const [minimum, maximum] = domain
  const rawPosition = (row.value - minimum) / (maximum - minimum)
  const performance
    = metric.direction === 'lower_is_better' ? 1 - rawPosition : rawPosition
  const percentage = Math.max(8, Math.min(100, 8 + performance * 92))
  const conditionLabel = modelHubEvaluationConditionLabel(
    row.model,
    row.evaluation.reasoning_effort,
  )
  const barStyle = {
    '--bar-color': `hsl(${hue} 72% 34%)`,
    '--bar-soft': `hsl(${hue} 72% 34% / 0.16)`,
    '--bar-size': `${percentage}%`,
  } as React.CSSProperties
  return (
    <div
      className={styles.barRow}
      style={barStyle}
      aria-label={`Rank ${rank}, ${row.model.display_name}, ${conditionLabel}, ${formatMetric(row.value, metric)}`}
    >
      <span className={styles.barIdentity}>
        <b className={styles.barRank}>{rank}</b>
        <CatalogMark presentation={row.model.presentation} />
        <span>
          <strong>{row.model.display_name}</strong>
          <small>
            {row.model.publisher}
            {' '}
            ·
            {conditionLabel}
          </small>
        </span>
      </span>
      <span className={styles.barTrack}>
        <span className={styles.barFill} />
      </span>
      <strong className={styles.barValue}>
        {formatMetric(row.value, metric)}
      </strong>
    </div>
  )
}

function ProviderCard({
  provider,
  protocols,
}: {
  provider: CatalogProvider
  protocols: Map<string, CatalogProtocol>
}) {
  return (
    <article className={styles.providerCard}>
      <div className={styles.providerTitle}>
        <CatalogMark presentation={provider.presentation} />
        <span>
          <strong>{provider.display_name}</strong>
          <code>{provider.id}</code>
        </span>
        <Badge
          value={provider.support_tier}
          label={tierLabel[provider.support_tier]}
        />
      </div>
      <p>{provider.description}</p>
      <div className={styles.providerFacts}>
        <span>
          <small>Category</small>
          <strong>{categoryLabel[provider.category]}</strong>
        </span>
        <span>
          <small>Curated mappings</small>
          <strong>
            {provider.models?.length
              ? provider.models.length
              : 'Custom models only'}
          </strong>
        </span>
        <span>
          <small>Auth</small>
          <strong>{readable(provider.auth.strategy)}</strong>
        </span>
      </div>
      <div className={styles.protocolList}>
        {provider.protocols.map((protocolID) => {
          const protocol = protocols.get(protocolID)
          const operationCount
            = protocol?.operations.filter(operation =>
              provider.supported_operations.includes(
                `${protocolID}#${operation.id}`,
              ),
            ).length ?? 0
          return (
            <span key={protocolID}>
              {protocol?.display_name ?? protocolID}
              <small>
                {operationCount}
                {' '}
                operations
              </small>
            </span>
          )
        })}
      </div>
      <div className={styles.providerFooter}>
        <Badge value={provider.conformance.status} />
        <small>
          {provider.conformance.verified_at
            ? `Verified ${provider.conformance.verified_at}`
            : 'Verification pending'}
        </small>
      </div>
    </article>
  )
}

function ModelDetail({
  model,
  providers,
  evaluations,
  benchmarks,
  family,
  modelByID,
  onClose,
}: {
  model: CatalogModel
  providers: Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
  evaluations: CatalogEvaluation[]
  benchmarks: CatalogBenchmark[]
  family?: { id: string, levels: string[], default: string }
  modelByID: Map<string, CatalogModel>
  onClose: () => void
}) {
  const dialogRef = useRef<HTMLElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const benchmarkByID = new Map(
    benchmarks.map(benchmark => [benchmark.id, benchmark]),
  )

  useEffect(() => {
    const dialog = dialogRef.current
    const previousFocus = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    closeRef.current?.focus()

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key !== 'Tab' || !dialog) return
      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter(element => !element.hidden)
      if (!focusable.length) {
        event.preventDefault()
        dialog.focus()
        return
      }
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (!dialog.contains(document.activeElement)) {
        event.preventDefault()
        const destination = event.shiftKey ? last : first
        destination.focus()
      }
      else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      }
      else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', handleKeyDown)
      previousFocus?.focus()
    }
  }, [onClose])

  return (
    <div
      className={styles.detailBackdrop}
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <aside
        ref={dialogRef}
        className={styles.modelDetail}
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-detail-title"
        aria-describedby="model-detail-description"
        tabIndex={-1}
      >
        <button
          ref={closeRef}
          type="button"
          className={styles.closeButton}
          onClick={onClose}
          aria-label="Close model details"
        >
          ×
        </button>
        <div className={styles.detailHero}>
          <CatalogMark presentation={model.presentation} large />
          <span>
            <small>{model.publisher}</small>
            <h2 id="model-detail-title">{model.display_name}</h2>
            <code>{model.id}</code>
          </span>
        </div>
        <p id="model-detail-description" className={styles.detailDescription}>
          {model.description}
        </p>
        <div className={styles.detailBadges}>
          <Badge value={model.kind} />
          <Badge
            value={model.distribution.type}
            label={distributionLabel[model.distribution.type]}
          />
          <Badge value={model.lifecycle} />
          {model.parameter_size
            ? (
                <Badge value="size" label={model.parameter_size} />
              )
            : null}
          {model.distribution.license
            ? (
                <Badge value="license" label={model.distribution.license} />
              )
            : null}
        </div>
        <div className={styles.detailFacts}>
          <span>
            <small>Context window</small>
            <strong>{formatTokens(model.limits?.context_window_size)}</strong>
          </span>
          <span>
            <small>Max output</small>
            <strong>{formatTokens(model.limits?.max_output_tokens)}</strong>
          </span>
          <span>
            <small>Released</small>
            <strong>{formatDate(model.released_at)}</strong>
          </span>
          <span>
            <small>Verified</small>
            <strong>{model.verification.verified_at}</strong>
          </span>
        </div>
        <DetailSection title="Capabilities & modalities">
          <Tags values={model.capabilities} />
          <p>
            Input:
            {model.modalities.input.map(readable).join(', ')}
            {' '}
            · Output:
            {model.modalities.output.map(readable).join(', ')}
          </p>
        </DetailSection>
        {family
          ? (
              <DetailSection title="Reasoning controls">
                <p>
                  Built-in family
                  <code>{family.id}</code>
                  ; default effort is
                  <strong>{readable(family.default)}</strong>
                  .
                </p>
                <Tags values={family.levels} />
              </DetailSection>
            )
          : null}
        {model.kind === 'virtual'
          ? (
              <DetailSection title="Backend model pool">
                <p>
                  This recipe composes candidates by routing role. Pool entries may
                  resolve to built-in or operator-defined models.
                </p>
                <div className={styles.recipeSummary}>
                  <span>
                    <small>Entrypoint</small>
                    <code>{model.entrypoint ?? model.id}</code>
                  </span>
                  <span>
                    <small>Recipe</small>
                    <strong>{model.recipe ?? 'Built-in'}</strong>
                  </span>
                  <span>
                    <small>Roles</small>
                    <strong>{model.roles?.length ?? 0}</strong>
                  </span>
                </div>
                <div className={styles.poolEntrypoint}>
                  <CatalogMark presentation={model.presentation} />
                  <span>
                    <strong>{model.display_name}</strong>
                    <small>Request entrypoint</small>
                  </span>
                </div>
                <div className={styles.roleGrid}>
                  {(model.roles ?? []).map(role => (
                    <article key={role.name} className={styles.roleCard}>
                      <div>
                        <strong>{readable(role.name)}</strong>
                        <Badge value={role.required ? 'required' : 'optional'} />
                      </div>
                      <small>{`Minimum ${role.minimum_candidates} candidate${role.minimum_candidates === 1 ? '' : 's'}`}</small>
                      <ul>
                        {role.recommended_pool.map((candidate) => {
                          const candidateModel = modelByID.get(candidate)
                          return (
                            <li key={candidate}>
                              {candidateModel
                                ? (
                                    <CatalogMark
                                      presentation={candidateModel.presentation}
                                    />
                                  )
                                : (
                                    <span className={styles.customModelMark}>C</span>
                                  )}
                              <span>
                                <strong>
                                  {candidateModel?.display_name ?? candidate}
                                </strong>
                                <small>
                                  {candidateModel
                                    ? candidateModel.publisher
                                    : 'Custom model slot'}
                                </small>
                              </span>
                            </li>
                          )
                        })}
                      </ul>
                      <Tags values={role.traits} />
                    </article>
                  ))}
                </div>
              </DetailSection>
            )
          : (
              <DetailSection title="Providers & API protocols">
                {providers.length
                  ? (
                      <div className={styles.detailProviderList}>
                        {providers.map(({ provider, binding }) => (
                          <article key={`${provider.id}:${binding.id}`}>
                            <span>
                              <CatalogMark presentation={provider.presentation} />
                              <span>
                                <strong>{provider.display_name}</strong>
                                <code>{binding.id}</code>
                              </span>
                            </span>
                            <span className={styles.bindingLabels}>
                              <Badge
                                value={binding.relationship}
                                label={relationshipLabel[binding.relationship]}
                              />
                              <Tags values={binding.protocols} />
                            </span>
                          </article>
                        ))}
                      </div>
                    )
                  : (
                      <p>No built-in provider binding is currently published.</p>
                    )}
              </DetailSection>
            )}
        <DetailSection title={`Evaluation evidence (${evaluations.length})`}>
          {evaluations.length
            ? (
                <div className={styles.evalList}>
                  {evaluations.map((evaluation) => {
                    const benchmark = benchmarkByID.get(evaluation.benchmark)
                    const dateLabel = modelHubEvaluationDateLabel(evaluation)
                    return (
                      <article key={evaluation.id}>
                        <div>
                          <span>
                            <strong>
                              {benchmark?.display_name ?? evaluation.benchmark}
                            </strong>
                            <small>
                              {modelHubEvaluationConditionLabel(
                                model,
                                evaluation.reasoning_effort,
                              )}
                              {' '}
                              ·
                              {' '}
                              {readable(evaluation.benchmark_profile)}
                            </small>
                          </span>
                          <span className={styles.metricChips}>
                            {Object.entries(evaluation.metrics ?? {}).map(
                              ([id, value]) => {
                                const definition = benchmark?.metrics.find(
                                  item => item.id === id,
                                )
                                return (
                                  <span key={id}>
                                    <small>{readable(id)}</small>
                                    <b>
                                      {definition && typeof value === 'number'
                                        ? formatMetric(value, definition)
                                        : value}
                                    </b>
                                  </span>
                                )
                              },
                            )}
                          </span>
                        </div>
                        {dateLabel || evaluation.evidence.source
                          ? (
                              <footer className={styles.evidenceMeta}>
                                {dateLabel ? <small>{dateLabel}</small> : null}
                                {evaluation.evidence.source
                                  ? (
                                      <a
                                        href={evaluation.evidence.source}
                                        target="_blank"
                                        rel="noreferrer"
                                      >
                                        Evidence source ↗
                                      </a>
                                    )
                                  : null}
                              </footer>
                            )
                          : null}
                      </article>
                    )
                  })}
                </div>
              )
            : (
                <p>
                  No benchmark measurements have been published for this model yet.
                </p>
              )}
        </DetailSection>
        <a
          className={styles.sourceLink}
          href={model.distribution.source}
          target="_blank"
          rel="noreferrer"
        >
          Open official model source ↗
        </a>
      </aside>
    </div>
  )
}

function Stat({ value, label }: { value: number, label: string }) {
  return (
    <div>
      <strong>{value.toLocaleString()}</strong>
      <span>{label}</span>
    </div>
  )
}
function SectionHeading({
  id,
  eyebrow,
  title,
  description,
}: {
  id: string
  eyebrow: string
  title: string
  description: string
}) {
  return (
    <div className={styles.sectionHeading}>
      <span>{eyebrow}</span>
      <h2 id={id}>{title}</h2>
      <p>{description}</p>
    </div>
  )
}
function FilterButtons({
  value,
  options,
  onChange,
}: {
  value: string
  options: Array<[string, string]>
  onChange: (value: string) => void
}) {
  return (
    <div className={styles.segmented} role="group">
      {options.map(([id, label]) => (
        <button
          key={id}
          type="button"
          className={value === id ? styles.segmentActive : undefined}
          onClick={() => onChange(id)}
          aria-pressed={value === id}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
function ViewToggle({
  value,
  onChange,
}: {
  value: ModelView
  onChange: (value: ModelView) => void
}) {
  return (
    <div className={styles.viewToggle} role="group" aria-label="Model view">
      <button
        type="button"
        aria-label="Show models as cards"
        title="Cards"
        aria-pressed={value === 'cards'}
        onClick={() => onChange('cards')}
      >
        ▦
        <span>Cards</span>
      </button>
      <button
        type="button"
        aria-label="Show models as a table"
        title="Table"
        aria-pressed={value === 'table'}
        onClick={() => onChange('table')}
      >
        ☷
        <span>Table</span>
      </button>
    </div>
  )
}
function SelectControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: Array<[string, string]>
  onChange: (value: string) => void
}) {
  return (
    <label className={styles.selectControl}>
      <span>{label}</span>
      <span className={styles.selectShell}>
        <select
          value={value}
          onChange={event => onChange(event.target.value)}
        >
          {options.map(([optionValue, optionLabel]) => (
            <option key={optionValue} value={optionValue}>
              {optionLabel}
            </option>
          ))}
        </select>
        <i aria-hidden="true">⌄</i>
      </span>
    </label>
  )
}

function Pagination({
  page,
  pageCount,
  total,
  pageSize,
  label,
  onChange,
}: {
  page: number
  pageCount: number
  total: number
  pageSize: number
  label: string
  onChange: (page: number) => void
}) {
  if (total === 0) return null
  const first = (page - 1) * pageSize + 1
  const last = Math.min(page * pageSize, total)
  const pages = Array.from(
    new Set([1, page - 1, page, page + 1, pageCount]),
  ).filter(candidate => candidate >= 1 && candidate <= pageCount)
  return (
    <nav className={styles.pagination} aria-label={`${label} pagination`}>
      <span>
        {first}
        –
        {last}
        {' '}
        of
        {total}
        {' '}
        {label}
      </span>
      <div>
        <button
          type="button"
          disabled={page === 1}
          onClick={() => onChange(page - 1)}
          aria-label="Previous page"
        >
          ←
        </button>
        {pages.map((candidate, index) => (
          <React.Fragment key={candidate}>
            {index > 0 && candidate - pages[index - 1] > 1
              ? (
                  <span>…</span>
                )
              : null}
            <button
              type="button"
              className={candidate === page ? styles.currentPage : undefined}
              aria-current={candidate === page ? 'page' : undefined}
              onClick={() => onChange(candidate)}
            >
              {candidate}
            </button>
          </React.Fragment>
        ))}
        <button
          type="button"
          disabled={page === pageCount}
          onClick={() => onChange(page + 1)}
          aria-label="Next page"
        >
          →
        </button>
      </div>
    </nav>
  )
}

function DetailSection({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <section className={styles.detailSection}>
      <h3>{title}</h3>
      {children}
    </section>
  )
}
function EmptyState({ title, body }: { title: string, body: string }) {
  return (
    <div className={styles.empty}>
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  )
}
function Badge({ value, label }: { value: string, label?: string }) {
  return (
    <span className={`${styles.badge} ${styles[`badge_${value}`] ?? ''}`}>
      {label ?? readable(value)}
    </span>
  )
}
function Tags({ values }: { values: string[] }) {
  return (
    <span className={styles.tags}>
      {values.map(value => (
        <span key={value}>{readable(value)}</span>
      ))}
    </span>
  )
}
