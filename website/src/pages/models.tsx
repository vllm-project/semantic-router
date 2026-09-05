import React, { useMemo, useState } from 'react'
import Layout from '@theme/Layout'
import Ai2 from '@lobehub/icons/es/Ai2/components/Mono'
import Ai21 from '@lobehub/icons/es/Ai21/components/Mono'
import Bedrock from '@lobehub/icons/es/Bedrock/components/Mono'
import ByteDance from '@lobehub/icons/es/ByteDance/components/Mono'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import Cohere from '@lobehub/icons/es/Cohere/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Grok from '@lobehub/icons/es/Grok/components/Mono'
import IBM from '@lobehub/icons/es/IBM/components/Mono'
import InternLM from '@lobehub/icons/es/InternLM/components/Mono'
import Kimi from '@lobehub/icons/es/Kimi/components/Mono'
import LG from '@lobehub/icons/es/LG/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Mono'
import Microsoft from '@lobehub/icons/es/Microsoft/components/Mono'
import Minimax from '@lobehub/icons/es/Minimax/components/Mono'
import Mistral from '@lobehub/icons/es/Mistral/components/Mono'
import Nvidia from '@lobehub/icons/es/Nvidia/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import Qwen from '@lobehub/icons/es/Qwen/components/Mono'
import Snowflake from '@lobehub/icons/es/Snowflake/components/Mono'
import Stepfun from '@lobehub/icons/es/Stepfun/components/Mono'
import TII from '@lobehub/icons/es/TII/components/Mono'
import Tencent from '@lobehub/icons/es/Tencent/components/Mono'
import Upstage from '@lobehub/icons/es/Upstage/components/Mono'
import XiaomiMiMo from '@lobehub/icons/es/XiaomiMiMo/components/Mono'
import Yi from '@lobehub/icons/es/Yi/components/Mono'
import Zhipu from '@lobehub/icons/es/Zhipu/components/Mono'
import catalogDocument from '../../static/model-catalog/catalog.json'
import styles from './models.module.css'

type SupportTier = 'native' | 'compatible' | 'runtime'

interface CatalogProtocol {
  id: string
  display_name: string
  operations: Array<{ id: string, method: string, path: string }>
}

interface CatalogHeader {
  channel: string
  default_intelligence_index: string
}

interface CatalogProvider {
  id: string
  display_name: string
  description: string
  category: 'start_here' | 'model_api' | 'private_runtime'
  support_tier: SupportTier
  default_base_url?: string
  protocols: string[]
  default_protocol: string
  supported_operations: string[]
  path_overrides?: Record<string, string>
  reasoning_transport?:
    | 'chat_template_kwargs'
    | 'top_level_effort'
    | 'top_level_boolean'
    | 'reasoning_object'
    | 'thinking_object'
    | 'deepseek_thinking'
  auth: { strategy: string }
  presentation: { logo: string, monogram: string, monochrome: boolean }
  conformance: { status: string, verified_at?: string }
  models?: CatalogModelBinding[]
}

interface CatalogModelBinding {
  catalog: string
  id: string
  protocols: string[]
  lifecycle: string
}

interface CatalogModel {
  id: string
  display_name: string
  description: string
  kind: 'physical' | 'virtual'
  publisher: string
  presentation: { logo: string, monogram: string, monochrome: boolean }
  distribution: {
    type: 'proprietary_api' | 'open_weights' | 'router_recipe'
    source: string
    license?: string
  }
  family: string
  parameter_size?: string
  lifecycle: 'experimental' | 'active' | 'deprecated' | 'removed'
  limits?: { context_window_size?: number, max_output_tokens?: number }
  capabilities: string[]
  modalities: { input: string[], output: string[] }
  reasoning_family?: string
  verification: { status: string, verified_at: string, source?: string }
}

interface CatalogEvaluation {
  id: string
  subject: Record<string, unknown>
  evidence: {
    provenance: 'vendor_claimed' | 'third_party' | 'vllm_sr_reproduced' | 'operator'
    verification: string
    source?: string
  }
}

interface CatalogIndex {
  id: string
  display_name: string
  description?: string
  methodology?: string
  aggregation: string
  scale: [number, number]
  missing: { policy: string, minimum?: number }
  components: Array<{
    benchmark?: string
    metric?: string
    benchmark_profile?: string
    index?: string
    weight: number
  }>
}

interface CatalogIndexResult {
  model: string
  reasoning_effort: string
  index: string
  status: string
  score: number | null
  coverage: number
  components: Array<{
    benchmark?: string
    metric?: string
    benchmark_profile?: string
    index?: string
    status: string
    value?: number | null
  }>
  provenance: string[]
}

interface CatalogSnapshot {
  catalogs: CatalogHeader[]
  protocols: CatalogProtocol[]
  providers: CatalogProvider[]
  reasoning_families: Array<{ id: string, levels: string[], default: string }>
  models: CatalogModel[]
  benchmarks: Array<{ id: string, display_name: string, domain: string }>
  evaluations: CatalogEvaluation[]
  indices: CatalogIndex[]
  index_results: CatalogIndexResult[]
}

const catalog = catalogDocument as CatalogSnapshot

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

const distributionLabel: Record<CatalogModel['distribution']['type'], string> = {
  open_weights: 'Open weights',
  proprietary_api: 'Proprietary API',
  router_recipe: 'Router recipe',
}

const packageIcons: Record<string, typeof OpenAI> = {
  ai2: Ai2,
  ai21: Ai21,
  anthropic: Claude,
  bytedance: ByteDance,
  cohere: Cohere,
  deepseek: DeepSeek,
  gemini: Gemini,
  internlm: InternLM,
  exaone: LG,
  lg: LG,
  meta: Meta,
  microsoft: Microsoft,
  minimax: Minimax,
  mistral: Mistral,
  moonshot: Kimi,
  nvidia: Nvidia,
  openai: OpenAI,
  qwen: Qwen,
  snowflake: Snowflake,
  stepfun: Stepfun,
  tencent: Tencent,
  tii: TII,
  upstage: Upstage,
  xai: Grok,
  xiaomimimo: XiaomiMiMo,
  yi: Yi,
  zai: Zhipu,
}

const publisherIcons: Record<string, typeof OpenAI> = {
  'Ai2': Ai2,
  'AI21 Labs': Ai21,
  'Alibaba Cloud': Qwen,
  'Amazon': Bedrock,
  'Anthropic': Claude,
  'ByteDance Seed': ByteDance,
  'Cohere': Cohere,
  'DeepSeek': DeepSeek,
  'Google': Gemini,
  'IBM': IBM,
  'LG AI Research': LG,
  'Shanghai AI Laboratory': InternLM,
  'Meta': Meta,
  'Microsoft': Microsoft,
  'Microsoft AI': Microsoft,
  'MiniMax': Minimax,
  'Mistral AI': Mistral,
  'Moonshot AI': Kimi,
  'NVIDIA': Nvidia,
  'OpenAI': OpenAI,
  'StepFun': Stepfun,
  'Snowflake': Snowflake,
  'Technology Innovation Institute': TII,
  'Tencent': Tencent,
  'Upstage': Upstage,
  'Xiaomi': XiaomiMiMo,
  '01.AI': Yi,
  'xAI': Grok,
  'Z.ai': Zhipu,
}

const readable = (value: string) => value.replace(/_/g, ' ')

const formatTokens = (value?: number) => {
  if (!value) return '—'
  if (value >= 1_000_000) return `${value / 1_000_000}M`
  if (value >= 1_000) return `${value / 1_000}K`
  return String(value)
}

const scoreStatus = (result?: CatalogIndexResult) => {
  if (!result || result.score === null) return 'Not yet measured'
  return result.score.toFixed(1)
}

const benchmarkName = (
  benchmarkID: string | undefined,
  metricID: string,
  benchmarks: CatalogSnapshot['benchmarks'],
) => {
  const benchmark = benchmarks.find(candidate => candidate.id === benchmarkID)
  return benchmark ? `${benchmark.display_name} · ${metricID}` : metricID
}

const benchmarkValue = (value?: number | null) => (
  typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'Missing'
)

const subjectSummary = (subject: Record<string, unknown>) => Object.entries(subject)
  .filter(([key, value]) => key !== 'source_kind' && value !== null && value !== '')
  .map(([key, value]) => `${readable(key)}: ${String(value)}`)
  .join(' · ')

function CatalogMark({
  presentation,
  publisher,
  large = false,
}: {
  presentation: { logo: string, monogram: string, monochrome: boolean }
  publisher?: string
  large?: boolean
}) {
  const packageID = presentation.logo.startsWith('package:')
    ? presentation.logo.slice('package:'.length)
    : ''
  const Icon = packageIcons[packageID] ?? (publisher ? publisherIcons[publisher] : undefined)
  return (
    <span className={`${styles.catalogMark} ${large ? styles.catalogMarkLarge : ''}`} aria-hidden="true">
      {Icon ? <Icon size={large ? 30 : 21} /> : presentation.monogram}
    </span>
  )
}

export default function ModelsPage() {
  const [providerSearch, setProviderSearch] = useState('')
  const [providerTier, setProviderTier] = useState<'all' | SupportTier>('all')
  const [modelSearch, setModelSearch] = useState('')
  const [modelKind, setModelKind] = useState<'all' | CatalogModel['kind']>('physical')
  const [modelDistribution, setModelDistribution] = useState<
    'all' | CatalogModel['distribution']['type']
  >('all')
  const [modelPublisher, setModelPublisher] = useState('all')
  const [modelLifecycle, setModelLifecycle] = useState<
    'supported' | 'all' | CatalogModel['lifecycle']
  >('supported')
  const [modelSort, setModelSort] = useState<'intelligence' | 'name'>('intelligence')
  const [selectedModelID, setSelectedModelID] = useState<string | null>(null)
  const [selectedEffort, setSelectedEffort] = useState<string | null>(null)

  const protocols = useMemo(
    () => new Map(catalog.protocols.map(protocol => [protocol.id, protocol])),
    [],
  )
  const providersByModel = useMemo(() => {
    const bindings = new Map<string, Array<{ provider: CatalogProvider, model: CatalogModelBinding }>>()
    for (const provider of catalog.providers) {
      for (const model of provider.models ?? []) {
        bindings.set(model.catalog, [
          ...(bindings.get(model.catalog) ?? []),
          { provider, model },
        ])
      }
    }
    return bindings
  }, [])
  const defaultIndexID = catalog.catalogs.find(header => header.channel === 'latest')
    ?.default_intelligence_index
    ?? catalog.catalogs[0]?.default_intelligence_index
  const defaultIndex = catalog.indices.find(index => index.id === defaultIndexID)
  const resultsByModel = useMemo(() => {
    const groups = new Map<string, CatalogIndexResult[]>()
    for (const result of catalog.index_results) {
      const key = `${result.index}:${result.model}`
      groups.set(key, [...(groups.get(key) ?? []), result])
    }
    return groups
  }, [])
  const preferredResults = useMemo(() => {
    const families = new Map(catalog.reasoning_families.map(family => [family.id, family]))
    return new Map(catalog.models.map((model) => {
      const key = `${defaultIndexID}:${model.id}`
      const candidates = resultsByModel.get(key) ?? []
      const defaultEffort = model.reasoning_family
        ? families.get(model.reasoning_family)?.default
        : 'default'
      const preferred = candidates.find(result => (
        result.reasoning_effort === defaultEffort && result.status === 'available'
      )) ?? candidates
        .filter(result => result.status === 'available')
        .sort((left, right) => right.coverage - left.coverage)[0]
        ?? candidates.find(result => result.reasoning_effort === defaultEffort)
        ?? candidates[0]
      return [key, preferred] as const
    }))
  }, [defaultIndexID, resultsByModel])
  const evaluations = useMemo(
    () => new Map(catalog.evaluations.map(evaluation => [evaluation.id, evaluation])),
    [],
  )
  const publishers = useMemo(
    () => [...new Set(catalog.models.map(model => model.publisher))]
      .sort((left, right) => left.localeCompare(right)),
    [],
  )

  const providers = useMemo(() => {
    const query = providerSearch.trim().toLocaleLowerCase()
    return catalog.providers.filter((provider) => {
      const matchesTier = providerTier === 'all' || provider.support_tier === providerTier
      const matchesQuery = !query || `${provider.display_name} ${provider.id} ${provider.description}`
        .toLocaleLowerCase()
        .includes(query)
      return matchesTier && matchesQuery
    })
  }, [providerSearch, providerTier])

  const models = useMemo(() => {
    const query = modelSearch.trim().toLocaleLowerCase()
    return catalog.models
      .filter((model) => {
        const matchesKind = modelKind === 'all' || model.kind === modelKind
        const matchesDistribution = modelDistribution === 'all'
          || model.distribution.type === modelDistribution
        const matchesPublisher = modelPublisher === 'all' || model.publisher === modelPublisher
        const matchesLifecycle = modelLifecycle === 'all'
          || (modelLifecycle === 'supported'
            ? model.lifecycle === 'active' || model.lifecycle === 'experimental'
            : model.lifecycle === modelLifecycle)
        const matchesQuery = !query || `${model.display_name} ${model.id} ${model.publisher} ${model.family} ${model.capabilities.join(' ')}`
          .toLocaleLowerCase()
          .includes(query)
        return matchesKind && matchesDistribution && matchesPublisher && matchesLifecycle && matchesQuery
      })
      .sort((left, right) => {
        if (modelSort === 'name') return left.display_name.localeCompare(right.display_name)
        const leftScore = preferredResults.get(`${defaultIndex?.id}:${left.id}`)?.score
        const rightScore = preferredResults.get(`${defaultIndex?.id}:${right.id}`)?.score
        if (leftScore !== null && leftScore !== undefined && rightScore !== null && rightScore !== undefined) {
          return rightScore - leftScore
        }
        if (leftScore !== null && leftScore !== undefined) return -1
        if (rightScore !== null && rightScore !== undefined) return 1
        return left.display_name.localeCompare(right.display_name)
      })
  }, [
    defaultIndex?.id,
    modelDistribution,
    modelKind,
    modelLifecycle,
    modelPublisher,
    modelSearch,
    modelSort,
    preferredResults,
  ])
  const selectedModel = models.find(model => model.id === selectedModelID) ?? models[0]
  const selectedResults = selectedModel && defaultIndex
    ? resultsByModel.get(`${defaultIndex.id}:${selectedModel.id}`) ?? []
    : []
  const preferredSelectedResult = selectedModel && defaultIndex
    ? preferredResults.get(`${defaultIndex.id}:${selectedModel.id}`)
    : undefined
  const selectedResult = selectedResults.find(result => (
    result.reasoning_effort === selectedEffort
  )) ?? preferredSelectedResult
  const selectedProviders = selectedModel
    ? (providersByModel.get(selectedModel.id) ?? [])
    : []
  const physicalModels = catalog.models.filter(model => model.kind === 'physical').length
  const virtualModels = catalog.models.length - physicalModels
  const scoredModels = defaultIndex
    ? new Set(catalog.index_results.filter(result => (
      result.index === defaultIndex.id && result.status === 'available'
    )).map(result => result.model)).size
    : 0

  return (
    <Layout
      title="Built-in models and providers"
      description="Browse the provider, protocol, model, reasoning, benchmark, and quality-index support compiled into vLLM Semantic Router."
    >
      <main className={styles.page}>
        <header className={styles.hero}>
          <div className={styles.heroCopy}>
            <span className={styles.eyebrow}>Repository-owned compatibility catalog</span>
            <h1>Models, providers, and evidence in one view</h1>
            <p>
              This page is generated from the same validated catalog used by the Router, CLI,
              and Dashboard. “Compatible” describes a wire contract; it is not presented as a
              native-adapter claim. Missing benchmark evidence is never converted into a zero.
            </p>
          </div>
          <div className={styles.stats} aria-label="Catalog summary">
            <Stat value={physicalModels} label="single models" />
            <Stat value={virtualModels} label="virtual recipes" />
            <Stat value={catalog.providers.length} label="providers" />
            <Stat value={publishers.length} label="publishers" />
            <Stat value={scoredModels} label="comparable scores" />
          </div>
        </header>

        <section className={styles.section} aria-labelledby="models-heading">
          <SectionHeading
            id="models-heading"
            eyebrow="Model cards and leaderboard"
            title="Built-in models"
            description="Scores are computed from versioned benchmark records. A model remains visible when its score is unavailable."
          />
          <div className={styles.modelControls}>
            <label className={styles.searchControl}>
              <span>Search</span>
              <input
                type="search"
                value={modelSearch}
                onChange={event => setModelSearch(event.target.value)}
                placeholder="Model, publisher, family, capability…"
              />
            </label>
            <SelectControl
              label="Kind"
              value={modelKind}
              options={[
                ['physical', 'Single models'],
                ['virtual', 'Virtual recipes'],
                ['all', 'All kinds'],
              ]}
              onChange={value => setModelKind(value as 'all' | CatalogModel['kind'])}
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
              onChange={value => setModelDistribution(
                value as 'all' | CatalogModel['distribution']['type'],
              )}
            />
            <SelectControl
              label="Publisher"
              value={modelPublisher}
              options={[
                ['all', 'All publishers'],
                ...publishers.map(publisher => [publisher, publisher] as [string, string]),
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
                ['all', 'All lifecycle states'],
              ]}
              onChange={value => setModelLifecycle(
                value as 'supported' | 'all' | CatalogModel['lifecycle'],
              )}
            />
            <SelectControl
              label="Sort"
              value={modelSort}
              options={[
                ['intelligence', 'Intelligence'],
                ['name', 'Name'],
              ]}
              onChange={value => setModelSort(value as 'intelligence' | 'name')}
            />
          </div>

          <div className={styles.modelWorkspace}>
            <div className={styles.modelCatalogPanel}>
              <div className={styles.resultHeader}>
                <strong>{`${models.length} models`}</strong>
                <span>Composite scores require 60% benchmark coverage.</span>
              </div>
              <div className={styles.modelList} role="listbox" aria-label="Model catalog results">
                {models.map((model) => {
                  const result = defaultIndex
                    ? preferredResults.get(`${defaultIndex.id}:${model.id}`)
                    : undefined
                  const active = model.id === selectedModel?.id
                  return (
                    <button
                      key={model.id}
                      type="button"
                      role="option"
                      aria-selected={active}
                      className={`${styles.modelRow} ${active ? styles.modelRowActive : ''}`}
                      onClick={() => {
                        setSelectedModelID(model.id)
                        setSelectedEffort(null)
                      }}
                    >
                      <span className={styles.modelIdentity}>
                        <CatalogMark presentation={model.presentation} publisher={model.publisher} />
                        <span>
                          <strong>{model.display_name}</strong>
                          <small>{model.id}</small>
                        </span>
                      </span>
                      <span>{model.publisher}</span>
                      <span>{distributionLabel[model.distribution.type]}</span>
                      <span>{formatTokens(model.limits?.context_window_size)}</span>
                      <span className={result?.score == null ? styles.unavailable : styles.score}>
                        {scoreStatus(result)}
                        {result?.score != null
                          ? <small>{`${Math.round(result.coverage * 100)}% coverage`}</small>
                          : null}
                      </span>
                    </button>
                  )
                })}
                {models.length === 0
                  ? <p className={styles.empty}>No built-in models match these filters.</p>
                  : null}
              </div>
            </div>

            <aside className={styles.modelDetail} aria-live="polite">
              {selectedModel
                ? (
                    <>
                      <div className={styles.detailIdentity}>
                        <CatalogMark
                          presentation={selectedModel.presentation}
                          publisher={selectedModel.publisher}
                          large
                        />
                        <span>
                          <small>{selectedModel.publisher}</small>
                          <h3>{selectedModel.display_name}</h3>
                          <code>{selectedModel.id}</code>
                        </span>
                      </div>
                      <p>{selectedModel.description}</p>
                      <div className={styles.detailBadges}>
                        <Badge value={selectedModel.kind} />
                        <Badge value={selectedModel.distribution.type} label={distributionLabel[selectedModel.distribution.type]} />
                        <Badge value={selectedModel.lifecycle} />
                        {selectedModel.parameter_size
                          ? <Badge value="parameter_size" label={selectedModel.parameter_size} />
                          : null}
                        {selectedModel.distribution.license
                          ? <Badge value="license" label={selectedModel.distribution.license} />
                          : null}
                      </div>

                      <DetailSection title="Intelligence evidence" value={scoreStatus(selectedResult)}>
                        {selectedResults.length > 1
                          ? (
                              <label className={styles.effortSelect}>
                                <span>Reasoning effort</span>
                                <select
                                  value={selectedResult?.reasoning_effort ?? ''}
                                  onChange={event => setSelectedEffort(event.target.value)}
                                >
                                  {selectedResults.map(result => (
                                    <option key={result.reasoning_effort} value={result.reasoning_effort}>
                                      {readable(result.reasoning_effort)}
                                    </option>
                                  ))}
                                </select>
                              </label>
                            )
                          : null}
                        <p>
                          {selectedResult?.score != null
                            ? `${Math.round(selectedResult.coverage * 100)}% of the default index is backed by published evidence for this effort.`
                            : 'No composite is shown until three of the five default benchmarks are available for this effort.'}
                        </p>
                        <div className={styles.benchmarkGrid}>
                          {selectedResult?.components.map(component => (
                            <div
                              key={component.index ?? `${component.benchmark}#${component.metric}@${component.benchmark_profile}`}
                            >
                              <span>
                                {component.metric
                                  ? benchmarkName(component.benchmark, component.metric, catalog.benchmarks)
                                  : (component.index ?? 'Index component')}
                              </span>
                              <strong>{benchmarkValue(component.value)}</strong>
                            </div>
                          ))}
                        </div>
                        {selectedResult?.provenance.length
                          ? (
                              <div className={styles.evidenceLinks}>
                                {selectedResult.provenance.map((recordID) => {
                                  const record = evaluations.get(recordID)
                                  return record
                                    ? (
                                        <div className={styles.evidenceRecord} key={recordID}>
                                          <span>
                                            {`${readable(record.evidence.provenance)} · ${record.evidence.verification}`}
                                          </span>
                                          <small>{subjectSummary(record.subject)}</small>
                                          {record.evidence.source
                                            ? (
                                                <a
                                                  href={record.evidence.source}
                                                  target="_blank"
                                                  rel="noreferrer"
                                                >
                                                  Official evidence ↗
                                                </a>
                                              )
                                            : null}
                                        </div>
                                      )
                                    : null
                                })}
                              </div>
                            )
                          : null}
                      </DetailSection>

                      <DetailSection title="Capabilities">
                        <Tags values={selectedModel.capabilities} />
                        {selectedModel.reasoning_family
                          ? <small>{`Reasoning family: ${selectedModel.reasoning_family}`}</small>
                          : null}
                      </DetailSection>

                      <DetailSection title="Available through">
                        {selectedProviders.length
                          ? (
                              <ul className={styles.providerList}>
                                {selectedProviders.map(({ provider, model }) => (
                                  <li key={`${provider.id}/${model.id}`}>
                                    <span>
                                      <strong>{provider.display_name}</strong>
                                      <code>{model.id}</code>
                                    </span>
                                    <small>{model.lifecycle}</small>
                                  </li>
                                ))}
                              </ul>
                            )
                          : <p>Materialized from its built-in router recipe.</p>}
                      </DetailSection>

                      <a
                        className={styles.sourceLink}
                        href={selectedModel.distribution.source}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open official model source ↗
                      </a>
                    </>
                  )
                : <p className={styles.empty}>Select a model to inspect it.</p>}
            </aside>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="providers-heading">
          <SectionHeading
            id="providers-heading"
            eyebrow="Transport support matrix"
            title="Built-in providers"
            description="Provider identity, auth defaults, endpoint behavior, protocol operations, support tier, and presentation metadata share one schema."
          />
          <div className={styles.legend}>
            <span>
              <Badge value="native" />
              {' '}
              provider-specific runtime behavior
            </span>
            <span>
              <Badge value="compatible" />
              {' '}
              compatibility protocol
            </span>
            <span>
              <Badge value="runtime" />
              {' '}
              private serving runtime
            </span>
          </div>
          <div className={styles.toolbar}>
            <input
              type="search"
              value={providerSearch}
              onChange={event => setProviderSearch(event.target.value)}
              placeholder="Search providers"
              aria-label="Search built-in providers"
            />
            <FilterButtons
              value={providerTier}
              options={[
                ['all', 'All'],
                ['native', 'Native'],
                ['compatible', 'Compatible'],
                ['runtime', 'Runtime'],
              ]}
              onChange={value => setProviderTier(value as 'all' | SupportTier)}
            />
          </div>
          <div className={styles.tableFrame}>
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Tier</th>
                  <th>Category</th>
                  <th>Protocol and API operations</th>
                  <th>Auth</th>
                  <th>Conformance</th>
                </tr>
              </thead>
              <tbody>
                {providers.map(provider => (
                  <tr key={provider.id}>
                    <td>
                      <span className={styles.providerIdentity}>
                        <CatalogMark presentation={provider.presentation} />
                        <span>
                          <strong>{provider.display_name}</strong>
                          <code>{provider.id}</code>
                          <small>{provider.description}</small>
                        </span>
                      </span>
                    </td>
                    <td><Badge value={provider.support_tier} label={tierLabel[provider.support_tier]} /></td>
                    <td>{categoryLabel[provider.category]}</td>
                    <td>
                      <div className={styles.protocols}>
                        {provider.protocols.map((protocolID) => {
                          const protocol = protocols.get(protocolID)
                          const operations = protocol?.operations.filter(operation =>
                            provider.supported_operations.includes(`${protocolID}#${operation.id}`),
                          ) ?? []
                          return (
                            <span key={protocolID}>
                              <strong>{protocol?.display_name ?? protocolID}</strong>
                              <small>
                                {operations.map((operation) => {
                                  const operationKey = `${protocolID}#${operation.id}`
                                  return `${operation.method} ${provider.path_overrides?.[operationKey] ?? operation.path}`
                                }).join(' · ') || '—'}
                              </small>
                            </span>
                          )
                        })}
                      </div>
                    </td>
                    <td>{readable(provider.auth.strategy)}</td>
                    <td>
                      <Badge value={provider.conformance.status} />
                      {provider.conformance.verified_at ? <small>{provider.conformance.verified_at}</small> : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {providers.length === 0 ? <p className={styles.empty}>No providers match these filters.</p> : null}
        </section>

        {defaultIndex
          ? (
              <section className={styles.section} aria-labelledby="method-heading">
                <SectionHeading
                  id="method-heading"
                  eyebrow="Transparent aggregation"
                  title={defaultIndex.display_name}
                  description={defaultIndex.description ?? ''}
                />
                <div className={styles.methodGrid}>
                  <div className={styles.methodCard}>
                    <span>Aggregation</span>
                    <strong>{readable(defaultIndex.aggregation)}</strong>
                    <small>
                      Scale
                      {defaultIndex.scale[0]}
                      –
                      {defaultIndex.scale[1]}
                    </small>
                  </div>
                  <div className={styles.methodCard}>
                    <span>Missing data</span>
                    <strong>{readable(defaultIndex.missing.policy)}</strong>
                    <small>Unavailable evidence stays unavailable.</small>
                  </div>
                  <div className={styles.methodCard}>
                    <span>Evidence records</span>
                    <strong>{catalog.evaluations.length}</strong>
                    <small>Versioned, model-specific measurements.</small>
                  </div>
                  <div className={styles.methodCard}>
                    <span>Methodology</span>
                    {defaultIndex.methodology
                      ? (
                          <a href={defaultIndex.methodology} target="_blank" rel="noreferrer">Reference ↗</a>
                        )
                      : <strong>Repository-defined</strong>}
                    <small>The index definition is versioned with the catalog.</small>
                  </div>
                </div>
                <div className={styles.componentGrid}>
                  {defaultIndex.components.map(component => (
                    <div
                      key={component.index ?? `${component.benchmark}#${component.metric}@${component.benchmark_profile}`}
                      className={styles.component}
                    >
                      <span>
                        {Math.round(component.weight * 100)}
                        %
                      </span>
                      <code>
                        {component.benchmark
                          ? `${component.benchmark}#${component.metric}`
                          : component.index}
                      </code>
                    </div>
                  ))}
                </div>
              </section>
            )
          : null}
      </main>
    </Layout>
  )
}

function Stat({ value, label }: { value: number, label: string }) {
  return (
    <div>
      <strong>{value}</strong>
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
    <div className={styles.filters} role="group">
      {options.map(([id, label]) => (
        <button
          key={id}
          type="button"
          className={value === id ? styles.activeFilter : undefined}
          onClick={() => onChange(id)}
          aria-pressed={value === id}
        >
          {label}
        </button>
      ))}
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
    <label>
      <span>{label}</span>
      <select value={value} onChange={event => onChange(event.target.value)}>
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
        ))}
      </select>
    </label>
  )
}

function DetailSection({
  title,
  value,
  children,
}: {
  title: string
  value?: string
  children: React.ReactNode
}) {
  return (
    <section className={styles.detailSection}>
      <div className={styles.detailSectionHeading}>
        <h4>{title}</h4>
        {value ? <strong>{value}</strong> : null}
      </div>
      {children}
    </section>
  )
}

function Badge({ value, label }: { value: string, label?: string }) {
  return <span className={`${styles.badge} ${styles[`badge_${value}`] ?? ''}`}>{label ?? readable(value)}</span>
}

function Tags({ values }: { values: string[] }) {
  return <span className={styles.tags}>{values.map(value => <span key={value}>{readable(value)}</span>)}</span>
}
