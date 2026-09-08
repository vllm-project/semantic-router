import React, { useCallback, useMemo, useState } from 'react'
import Layout from '@theme/Layout'

import catalogDocument from '../../static/model-catalog/catalog.json'
import { ModelHubBenchmark } from '../components/model-hub/ModelHubBenchmark'
import { ModelHubDirectory } from '../components/model-hub/ModelHubDirectory'
import { ModelHubDetail } from '../components/model-hub/ModelHubDetail'
import { ModelHubProviders } from '../components/model-hub/ModelHubProviders'
import {
  modelHubBenchmarkCharts,
  modelHubBenchmarkDomain,
  modelHubChartColors,
  modelHubPublicEvaluations,
} from '../data/modelHubBenchmarkSupport'
import type {
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

const supportedLifecycle = (
  model: CatalogModel,
  lifecycle: ModelHubDirectoryFilters['lifecycle'],
) =>
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
        const matchesQuery
          = !query
            || `${model.display_name} ${model.id} ${model.publisher} ${model.family} ${model.capabilities.join(' ')}`
              .toLocaleLowerCase()
              .includes(query)
        return (
          matchesQuery
          && (filters.kind === 'all' || model.kind === filters.kind)
          && (filters.distribution === 'all' || model.distribution.type === filters.distribution)
          && (filters.publisher === 'all' || model.publisher === filters.publisher)
          && (filters.provider === 'all'
            || providersForModel.some(({ provider }) => provider.id === filters.provider))
          && (filters.capability === 'all' || model.capabilities.includes(filters.capability))
          && supportedLifecycle(model, filters.lifecycle)
        )
      })
      .sort((left, right) => {
        if (filters.sort === 'name') return left.display_name.localeCompare(right.display_name)
        if (filters.sort === 'context') {
          return (
            (right.limits?.context_window_size ?? 0) - (left.limits?.context_window_size ?? 0)
            || left.display_name.localeCompare(right.display_name)
          )
        }
        return (
          (right.released_at ?? '').localeCompare(left.released_at ?? '')
          || left.display_name.localeCompare(right.display_name)
        )
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

function useBenchmarkExplorer() {
  const allCharts = useMemo(() => modelHubBenchmarkCharts(catalog), [])
  const colors = useMemo(
    () =>
      modelHubChartColors(
        catalog.models.map(model => model.id),
        catalog.models.map(model => model.id),
      ),
    [],
  )
  const [benchmarkFilter, setBenchmarkFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [publisher, setPublisher] = useState('all')
  const benchmarkFilters = useMemo(() => {
    const tagCounts = new Map<string, number>()
    const domainCounts = new Map<string, number>()
    allCharts.forEach((chart) => {
      domainCounts.set(
        chart.benchmark.domain,
        (domainCounts.get(chart.benchmark.domain) ?? 0) + 1,
      )
      chart.benchmark.tags?.forEach((tag) => {
        tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1)
      })
    })
    const tags = Array.from(tagCounts, ([tag, count]) => ({
      id: `tag:${tag}`,
      label: tag === 'core' ? 'Core' : tag,
      count,
    })).sort((left, right) => {
      if (left.id === 'tag:core') return -1
      if (right.id === 'tag:core') return 1
      return left.label.localeCompare(right.label)
    })
    const domains = Array.from(domainCounts, ([domain, count]) => ({
      id: `domain:${domain}`,
      label: domain,
      count,
    })).sort((left, right) => left.label.localeCompare(right.label))
    return [...tags, ...domains]
  }, [allCharts])
  const publishers = useMemo(
    () =>
      Array.from(
        new Set(allCharts.flatMap(chart => chart.rows.map(row => row.model.publisher))),
      ).sort((left, right) => left.localeCompare(right)),
    [allCharts],
  )
  const activePublisher = publisher === 'all' || publishers.includes(publisher) ? publisher : 'all'
  const activeBenchmarkFilter
    = benchmarkFilter === 'all' || benchmarkFilters.some(item => item.id === benchmarkFilter)
      ? benchmarkFilter
      : 'all'
  const charts = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return allCharts
      .filter((chart) => {
        if (activeBenchmarkFilter === 'all') return true
        if (activeBenchmarkFilter.startsWith('tag:')) {
          return chart.benchmark.tags?.includes(activeBenchmarkFilter.slice(4)) ?? false
        }
        return chart.benchmark.domain === activeBenchmarkFilter.slice(7)
      })
      .map((chart) => {
        const rows = chart.rows.filter(
          row =>
            (activePublisher === 'all' || row.model.publisher === activePublisher)
            && (!needle
              || `${row.model.display_name} ${row.model.id} ${row.model.publisher}`
                .toLocaleLowerCase()
                .includes(needle)),
        )
        return {
          ...chart,
          rows,
          totalResults: chart.rows.length,
          domain: modelHubBenchmarkDomain(
            rows.map(row => row.value),
            chart.metric,
          ),
          colors,
        }
      })
      .filter(chart => chart.rows.length)
  }, [activeBenchmarkFilter, activePublisher, allCharts, colors, query])

  return {
    charts,
    chartCount: allCharts.length,
    filters: benchmarkFilters,
    filter: activeBenchmarkFilter,
    setFilter: setBenchmarkFilter,
    query,
    setQuery,
    publisher: activePublisher,
    setPublisher,
    publishers,
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
  const benchmark = useBenchmarkExplorer()
  const selectedModel = selectedModelID ? modelByID.get(selectedModelID) : undefined
  const selectedProviders = selectedModel
    ? (directory.providersByModel.get(selectedModel.id) ?? [])
    : []
  const selectedEvaluations = selectedModel
    ? [...(directory.evaluationsByModel.get(selectedModel.id) ?? [])].sort((left, right) =>
        left.benchmark.localeCompare(right.benchmark),
      )
    : []
  const selectedFamily = selectedModel?.reasoning_family
    ? reasoningFamilies.get(selectedModel.reasoning_family)
    : undefined
  const closeModelDetail = useCallback(() => setSelectedModelID(null), [])
  const physicalModels = catalog.models.filter(model => model.kind === 'physical').length
  const virtualModels = catalog.models.length - physicalModels
  const creators = new Set(
    catalog.models.filter(model => model.kind === 'physical').map(model => model.publisher),
  ).size
  const mappedProviders = catalog.providers.filter(provider => provider.models?.length).length
  const evaluations = modelHubPublicEvaluations(catalog.evaluations).length

  return (
    <Layout
      title="Model Hub"
      description="Built-in models and exact benchmark evidence for vLLM Semantic Router."
    >
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
          <ModelHubBenchmark {...benchmark} selectModel={setSelectedModelID} />
        </section>

        <section id="providers" className={styles.section} aria-labelledby="providers-heading">
          <header className={styles.sectionHeading}>
            <h2 id="providers-heading">Providers</h2>
            <span>
              {mappedProviders}
              {' '}
              mapped ·
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
