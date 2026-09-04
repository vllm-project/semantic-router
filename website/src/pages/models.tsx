import React, { useMemo, useState } from 'react'
import Layout from '@theme/Layout'
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
  reasoning_transport?: 'chat_template_kwargs' | 'top_level_effort' | 'deepseek_thinking'
  auth: { strategy: string }
  presentation: { monogram: string }
  conformance: { status: string, verified_at?: string }
}

interface CatalogModel {
  id: string
  display_name: string
  description: string
  kind: 'physical' | 'virtual'
  family: string
  parameter_size?: string
  lifecycle: string
  limits?: { context_window_size?: number, max_output_tokens?: number }
  capabilities: string[]
  modalities: { input: string[], output: string[] }
  reasoning_family?: string
  protocols: string[]
  verification: { status: string, verified_at: string }
}

interface CatalogIndex {
  id: string
  display_name: string
  description?: string
  methodology?: string
  aggregation: string
  scale: [number, number]
  missing: { policy: string, minimum?: number }
  components: Array<{ metric?: string, index?: string, weight: number }>
}

interface CatalogIndexResult {
  model: string
  index: string
  status: string
  score: number | null
  coverage: number
}

interface CatalogSnapshot {
  catalogs: CatalogHeader[]
  protocols: CatalogProtocol[]
  providers: CatalogProvider[]
  models: CatalogModel[]
  offerings: Array<{ provider: string, model: string, lifecycle?: string }>
  benchmarks: Array<{ id: string, display_name: string, domain: string }>
  evaluations: unknown[]
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

export default function ModelsPage() {
  const [providerSearch, setProviderSearch] = useState('')
  const [providerTier, setProviderTier] = useState<'all' | SupportTier>('all')
  const [modelSearch, setModelSearch] = useState('')
  const [modelKind, setModelKind] = useState<'all' | CatalogModel['kind']>('all')

  const protocols = useMemo(
    () => new Map(catalog.protocols.map(protocol => [protocol.id, protocol])),
    [],
  )
  const offeringCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const offering of catalog.offerings) {
      if (offering.lifecycle === 'removed') continue
      counts.set(offering.model, (counts.get(offering.model) ?? 0) + 1)
    }
    return counts
  }, [])
  const defaultIndexID = catalog.catalogs.find(header => header.channel === 'latest')
    ?.default_intelligence_index
    ?? catalog.catalogs[0]?.default_intelligence_index
  const defaultIndex = catalog.indices.find(index => index.id === defaultIndexID)
  const results = useMemo(
    () => new Map(catalog.index_results.map(result => [`${result.index}:${result.model}`, result])),
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
        const matchesQuery = !query || `${model.display_name} ${model.id} ${model.family} ${model.capabilities.join(' ')}`
          .toLocaleLowerCase()
          .includes(query)
        return matchesKind && matchesQuery
      })
      .sort((left, right) => {
        const leftScore = results.get(`${defaultIndex?.id}:${left.id}`)?.score
        const rightScore = results.get(`${defaultIndex?.id}:${right.id}`)?.score
        if (leftScore !== null && leftScore !== undefined && rightScore !== null && rightScore !== undefined) {
          return rightScore - leftScore
        }
        if (leftScore !== null && leftScore !== undefined) return -1
        if (rightScore !== null && rightScore !== undefined) return 1
        return left.display_name.localeCompare(right.display_name)
      })
  }, [defaultIndex?.id, modelKind, modelSearch, results])

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
            <Stat value={catalog.providers.length} label="providers" />
            <Stat value={catalog.models.length} label="built-in models" />
            <Stat value={catalog.protocols.length} label="protocols" />
            <Stat value={catalog.benchmarks.length} label="benchmarks" />
          </div>
        </header>

        <section className={styles.section} aria-labelledby="models-heading">
          <SectionHeading
            id="models-heading"
            eyebrow="Model cards and leaderboard"
            title="Built-in models"
            description="Scores are computed from versioned benchmark records. A model remains visible when its score is unavailable."
          />
          <div className={styles.toolbar}>
            <input
              type="search"
              value={modelSearch}
              onChange={event => setModelSearch(event.target.value)}
              placeholder="Search models, families, or capabilities"
              aria-label="Search built-in models"
            />
            <FilterButtons
              value={modelKind}
              options={[
                ['all', 'All'],
                ['physical', 'Physical'],
                ['virtual', 'Virtual'],
              ]}
              onChange={value => setModelKind(value as 'all' | CatalogModel['kind'])}
            />
          </div>
          <div className={styles.tableFrame}>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Kind</th>
                  <th>Context</th>
                  <th>Capabilities</th>
                  <th>Reasoning</th>
                  <th>Offerings</th>
                  <th>{defaultIndex?.display_name ?? 'Quality index'}</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => {
                  const result = defaultIndex
                    ? results.get(`${defaultIndex.id}:${model.id}`)
                    : undefined
                  return (
                    <tr key={model.id}>
                      <td>
                        <strong>{model.display_name}</strong>
                        <code>{model.id}</code>
                        <small>{model.description}</small>
                      </td>
                      <td><Badge value={model.kind} /></td>
                      <td>{formatTokens(model.limits?.context_window_size)}</td>
                      <td><Tags values={model.capabilities} /></td>
                      <td>{model.reasoning_family ?? '—'}</td>
                      <td>{offeringCounts.get(model.id) ?? 0}</td>
                      <td>
                        <span className={result?.score == null ? styles.unavailable : styles.score}>
                          {scoreStatus(result)}
                        </span>
                        {result
                          ? (
                              <small>
                                {Math.round(result.coverage * 100)}
                                % coverage
                              </small>
                            )
                          : null}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {models.length === 0 ? <p className={styles.empty}>No built-in models match these filters.</p> : null}
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
                        <span className={styles.monogram} aria-hidden="true">{provider.presentation.monogram}</span>
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
                    <div key={component.metric ?? component.index} className={styles.component}>
                      <span>
                        {Math.round(component.weight * 100)}
                        %
                      </span>
                      <code>{component.metric ?? component.index}</code>
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

function Badge({ value, label }: { value: string, label?: string }) {
  return <span className={`${styles.badge} ${styles[`badge_${value}`] ?? ''}`}>{label ?? readable(value)}</span>
}

function Tags({ values }: { values: string[] }) {
  return <span className={styles.tags}>{values.map(value => <span key={value}>{readable(value)}</span>)}</span>
}
