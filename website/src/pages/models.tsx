import React, { useCallback, useMemo, useState } from 'react'
import Layout from '@theme/Layout'

import catalogDocument from '../../static/model-catalog/catalog.json'
import { ModelHubBenchmark } from '../components/model-hub/ModelHubBenchmark'
import {
  ModelHubDirectory,
} from '../components/model-hub/ModelHubDirectory'
import { ModelHubDetail } from '../components/model-hub/ModelHubDetail'
import { ModelHubProviders } from '../components/model-hub/ModelHubProviders'
import {
  availableModelHubBenchmarkMetrics,
  availableModelHubBenchmarkProfiles,
  MODEL_HUB_MIN_BENCHMARK_MODELS,
  modelHubBenchmarkSelectionCounts,
  modelHubBenchmarkDomain,
  modelHubChartColors,
  modelHubPublicEvaluations,
  preferredModelHubBenchmarkSelection,
} from '../data/modelHubBenchmarkSupport'
import type {
  BenchmarkRow,
  CatalogEvaluation,
  CatalogModel,
  CatalogModelBinding,
  CatalogProvider,
  CatalogSnapshot,
  ModelView,
} from '../data/modelHubCatalogTypes'
import {
  modelHubDirectoryDefaults,
  type ModelHubDirectoryFilters,
} from '../data/modelHubDirectorySupport'
import styles from './models.module.css'

const catalog = catalogDocument as unknown as CatalogSnapshot
const MODEL_PAGE_SIZE = 10

const supportedLifecycle = (model: CatalogModel, lifecycle: ModelHubDirectoryFilters['lifecycle']) =>
  lifecycle === 'all'
  || (lifecycle === 'supported'
    ? model.lifecycle === 'active' || model.lifecycle === 'experimental'
    : model.lifecycle === lifecycle)

function modelBindings(): Map<
  string,
  Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
> {
  const bindings = new Map<
    string,
    Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
  >()
  catalog.providers.forEach(provider =>
    (provider.models ?? []).forEach(binding =>
      bindings.set(binding.catalog, [
        ...(bindings.get(binding.catalog) ?? []),
        { provider, binding },
      ]),
    ),
  )
  return bindings
}

function availableEvaluations(): Map<string, CatalogEvaluation[]> {
  const grouped = new Map<string, CatalogEvaluation[]>()
  modelHubPublicEvaluations(catalog.evaluations).forEach((evaluation) => {
    grouped.set(evaluation.model, [...(grouped.get(evaluation.model) ?? []), evaluation])
  })
  return grouped
}

function useCatalogDirectory() {
  const [filters, setFilters] = useState(modelHubDirectoryDefaults)
  const [view, setView] = useState<ModelView>('list')
  const [page, setPage] = useState(1)
  const providersByModel = useMemo(modelBindings, [])
  const evaluationsByModel = useMemo(availableEvaluations, [])
  const publishers = useMemo(
    () => Array.from(
      new Set(
        catalog.models
          .filter(model => model.kind === 'physical')
          .map(model => model.publisher),
      ),
    ).sort((left, right) => left.localeCompare(right)),
    [],
  )
  const providers = useMemo(
    () => catalog.providers.filter(provider => provider.models?.length),
    [],
  )
  const capabilities = useMemo(
    () => Array.from(new Set(catalog.models.flatMap(model => model.capabilities))).sort(),
    [],
  )
  const models = useMemo(() => {
    const query = filters.search.trim().toLocaleLowerCase()
    return catalog.models
      .filter((model) => {
        const providersForModel = providersByModel.get(model.id) ?? []
        const matchesQuery = !query
          || `${model.display_name} ${model.id} ${model.publisher} ${model.family} ${model.capabilities.join(' ')}`
            .toLocaleLowerCase()
            .includes(query)
        return (
          matchesQuery
          && (filters.kind === 'all' || model.kind === filters.kind)
          && (filters.distribution === 'all' || model.distribution.type === filters.distribution)
          && (filters.publisher === 'all' || model.publisher === filters.publisher)
          && (filters.provider === 'all' || providersForModel.some(({ provider }) => provider.id === filters.provider))
          && (filters.capability === 'all' || model.capabilities.includes(filters.capability))
          && supportedLifecycle(model, filters.lifecycle)
        )
      })
      .sort((left, right) => {
        if (filters.sort === 'name') return left.display_name.localeCompare(right.display_name)
        if (filters.sort === 'context') {
          return (right.limits?.context_window_size ?? 0) - (left.limits?.context_window_size ?? 0) || left.display_name.localeCompare(right.display_name)
        }
        return (right.released_at ?? '').localeCompare(left.released_at ?? '') || left.display_name.localeCompare(right.display_name)
      })
  }, [filters, providersByModel])
  const updateFilters = (patch: Partial<ModelHubDirectoryFilters>) => {
    setFilters(current => ({ ...current, ...patch }))
    setPage(1)
  }
  const pageCount = Math.max(1, Math.ceil(models.length / MODEL_PAGE_SIZE))
  const pageModels = models.slice((page - 1) * MODEL_PAGE_SIZE, page * MODEL_PAGE_SIZE)

  return {
    filters,
    updateFilters,
    resetFilters: () => {
      setFilters(modelHubDirectoryDefaults)
      setPage(1)
    },
    view,
    setView,
    page,
    setPage,
    pageCount,
    pageModels,
    models,
    publishers,
    providers,
    capabilities,
    providersByModel,
    evaluationsByModel,
  }
}

function useBenchmarkExplorer(modelByID: Map<string, CatalogModel>) {
  const counts = useMemo(
    () => modelHubBenchmarkSelectionCounts(
      catalog.evaluations,
      MODEL_HUB_MIN_BENCHMARK_MODELS,
    ),
    [],
  )
  const benchmarkCounts = useMemo(() => {
    const values = new Map<string, number>()
    counts.forEach((count, key) => {
      const benchmarkID = key.split('\u0000')[0]
      values.set(benchmarkID, Math.max(values.get(benchmarkID) ?? 0, count))
    })
    return values
  }, [counts])
  const benchmarks = useMemo(
    () => catalog.benchmarks
      .filter(benchmark => (benchmarkCounts.get(benchmark.id) ?? 0) > 0)
      .sort((left, right) => (benchmarkCounts.get(right.id) ?? 0) - (benchmarkCounts.get(left.id) ?? 0)),
    [benchmarkCounts],
  )
  const fallback = preferredModelHubBenchmarkSelection(counts)
  const [benchmarkID, setBenchmarkID] = useState(fallback?.benchmark ?? benchmarks[0]?.id ?? '')
  const benchmark = benchmarks.find(item => item.id === benchmarkID) ?? benchmarks[0]
  const preferred = preferredModelHubBenchmarkSelection(counts, benchmark?.id ?? '')
  const [profile, setProfile] = useState(preferred?.profile ?? benchmark?.default_profile ?? '')
  const [metricID, setMetricID] = useState(preferred?.metric ?? benchmark?.metrics[0]?.id ?? '')
  const [query, setQuery] = useState('')
  const [publisher, setPublisher] = useState('all')
  const availableProfiles = benchmark
    ? availableModelHubBenchmarkProfiles(counts, benchmark.id)
    : new Set<string>()
  const activeProfile = availableProfiles.has(profile) ? profile : preferred?.profile ?? ''
  const availableMetrics = benchmark
    ? availableModelHubBenchmarkMetrics(counts, benchmark.id, activeProfile)
    : new Set<string>()
  const metric = benchmark?.metrics.find(item => item.id === metricID && availableMetrics.has(item.id))
    ?? benchmark?.metrics.find(item => availableMetrics.has(item.id))
  const allRows = useMemo(() => {
    if (!benchmark || !metric) return []
    const unique = new Map<string, BenchmarkRow>()
    catalog.evaluations.forEach((evaluation) => {
      const value = evaluation.metrics?.[metric.id]
      const model = modelByID.get(evaluation.model)
      if (
        evaluation.status !== 'available'
        || evaluation.benchmark !== benchmark.id
        || evaluation.benchmark_profile !== activeProfile
        || typeof value !== 'number'
        || !model
      ) return
      unique.set(`${model.id}:${evaluation.reasoning_effort}`, { evaluation, model, value })
    })
    return Array.from(unique.values()).sort((left, right) =>
      metric.direction === 'lower_is_better'
        ? left.value - right.value
        : right.value - left.value,
    )
  }, [activeProfile, benchmark, metric, modelByID])
  const publishers = useMemo(
    () => Array.from(new Set(allRows.map(row => row.model.publisher))).sort((left, right) => left.localeCompare(right)),
    [allRows],
  )
  const activePublisher = publisher === 'all' || publishers.includes(publisher) ? publisher : 'all'
  const rows = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return allRows.filter(row => (
      (activePublisher === 'all' || row.model.publisher === activePublisher)
      && (!needle || `${row.model.display_name} ${row.model.id} ${row.model.publisher}`.toLocaleLowerCase().includes(needle))
    ))
  }, [activePublisher, allRows, query])
  const domain = metric
    ? modelHubBenchmarkDomain(rows.map(row => row.value), metric)
    : [0, 1] as [number, number]
  const colors = useMemo(
    () => modelHubChartColors(
      rows.map(row => row.model.id),
      catalog.models.map(model => model.id),
    ),
    [rows],
  )
  const chooseBenchmark = (id: string) => {
    const next = benchmarks.find(item => item.id === id)
    const nextPreferred = preferredModelHubBenchmarkSelection(counts, id)
    setBenchmarkID(id)
    setProfile(nextPreferred?.profile ?? next?.default_profile ?? '')
    setMetricID(nextPreferred?.metric ?? next?.metrics[0]?.id ?? '')
  }
  const chooseProfile = (id: string) => {
    const next = preferredModelHubBenchmarkSelection(counts, benchmark?.id, id)
    setProfile(id)
    setMetricID(next?.metric ?? '')
  }

  return {
    benchmark,
    metric,
    profile: activeProfile,
    benchmarks,
    profiles: benchmark?.profiles.filter(item => availableProfiles.has(item.id)).map(item => [item.id, item.display_name] as [string, string]) ?? [],
    metrics: benchmark?.metrics.filter(item => availableMetrics.has(item.id)).map(item => [item.id, item.id.replace(/_/g, ' ')] as [string, string]) ?? [],
    rows,
    totalResults: allRows.length,
    query,
    setQuery,
    publisher: activePublisher,
    setPublisher,
    publishers,
    domain,
    colors,
    chooseBenchmark,
    chooseProfile,
    chooseMetric: setMetricID,
  }
}

export default function ModelsPage() {
  const directory = useCatalogDirectory()
  const [selectedModelID, setSelectedModelID] = useState<string | null>(null)
  const modelByID = useMemo(() => new Map(catalog.models.map(model => [model.id, model])), [])
  const reasoningFamilies = useMemo(
    () => new Map(catalog.reasoning_families.map(family => [family.id, family])),
    [],
  )
  const benchmark = useBenchmarkExplorer(modelByID)
  const selectedModel = selectedModelID ? modelByID.get(selectedModelID) : undefined
  const selectedProviders = selectedModel ? directory.providersByModel.get(selectedModel.id) ?? [] : []
  const selectedEvaluations = selectedModel
    ? [...(directory.evaluationsByModel.get(selectedModel.id) ?? [])].sort((left, right) => left.benchmark.localeCompare(right.benchmark))
    : []
  const selectedFamily = selectedModel?.reasoning_family
    ? reasoningFamilies.get(selectedModel.reasoning_family)
    : undefined
  const closeModelDetail = useCallback(() => setSelectedModelID(null), [])
  const physicalModels = catalog.models.filter(model => model.kind === 'physical').length
  const virtualModels = catalog.models.length - physicalModels
  const creators = new Set(catalog.models.filter(model => model.kind === 'physical').map(model => model.publisher)).size
  const mappedProviders = catalog.providers.filter(provider => provider.models?.length).length
  const evaluations = modelHubPublicEvaluations(catalog.evaluations).length

  return (
    <Layout title="Model Hub" description="Built-in models and exact benchmark evidence for vLLM Semantic Router.">
      <main className={styles.page}>
        <header className={styles.pageHeader}>
          <div>
            <h1>Model Hub</h1>
            <p>Models, ready to route.</p>
          </div>
          <dl>
            <div>
              <dd>{catalog.models.length}</dd>
              <dt>models</dt>
            </div>
            <div>
              <dd>{creators}</dd>
              <dt>creators</dt>
            </div>
            <div>
              <dd>{mappedProviders}</dd>
              <dt>mapped providers</dt>
            </div>
            <div>
              <dd>{evaluations}</dd>
              <dt>evaluations</dt>
            </div>
          </dl>
        </header>

        <section id="models" className={styles.section} aria-labelledby="models-heading">
          <header className={styles.sectionHeading}>
            <h2 id="models-heading">Models</h2>
            <span>
              {physicalModels}
              {' '}
              single ·
              {' '}
              {virtualModels}
              {' '}
              virtual
            </span>
          </header>
          <ModelHubDirectory
            {...directory}
            pageSize={MODEL_PAGE_SIZE}
            selectModel={setSelectedModelID}
          />
        </section>

        <section id="benchmarks" className={styles.section} aria-labelledby="benchmarks-heading">
          <header className={styles.sectionHeading}>
            <h2 id="benchmarks-heading">Benchmarks</h2>
            <span>Exact published results</span>
          </header>
          <ModelHubBenchmark
            {...benchmark}
            selectModel={setSelectedModelID}
          />
        </section>

        <section id="providers" className={styles.section} aria-labelledby="providers-heading">
          <header className={styles.sectionHeading}>
            <h2 id="providers-heading">Providers</h2>
            <span>
              {mappedProviders}
              {' '}
              mapped ·
              {' '}
              {catalog.providers.length}
              {' '}
              runtime contracts
            </span>
          </header>
          <ModelHubProviders providers={catalog.providers} protocols={catalog.protocols} />
        </section>

        {selectedModel
          ? (
              <ModelHubDetail
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
