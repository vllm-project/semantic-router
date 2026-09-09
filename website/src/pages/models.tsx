import React, { useCallback, useMemo } from 'react'
import { useHistory, useLocation } from '@docusaurus/router'
import Translate from '@docusaurus/Translate'
import Layout from '@theme/Layout'
import BrowseLayout from '@site/src/components/site/BrowseLayout'

import catalogDocument from '../../static/model-catalog/catalog.json'
import { ModelHubArena } from '../components/model-hub/ModelHubArena'
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
import {
  parseModelHubUrlState,
  serializeModelHubUrlState,
  type ModelHubBenchmarkUrlState,
  type ModelHubUrlState,
} from '../data/modelHubUrlState'
import styles from './models.module.css'

const catalog = catalogDocument as unknown as CatalogSnapshot
const MODEL_PAGE_SIZE = 10
const MODEL_HUB_SECTIONS = [
  { key: 'arena', label: 'Arena' },
  { key: 'models', label: 'Models' },
  { key: 'benchmarks', label: 'Benchmarks' },
  { key: 'providers', label: 'Providers' },
] as const

type UpdateModelHubUrlState = (
  updater: (current: ModelHubUrlState) => ModelHubUrlState,
  push?: boolean,
) => void

function modelHubActiveSection(hash: string): string {
  const id = hash.replace('#', '')
  return MODEL_HUB_SECTIONS.some(section => section.key === id) ? id : 'arena'
}

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

function useCatalogDirectory(
  urlState: ModelHubUrlState,
  updateUrlState: UpdateModelHubUrlState,
) {
  const { filters, view } = urlState
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
  const pageCount = Math.max(1, Math.ceil(models.length / MODEL_PAGE_SIZE))
  const page = Math.min(urlState.page, pageCount)
  const pageModels = models.slice((page - 1) * MODEL_PAGE_SIZE, page * MODEL_PAGE_SIZE)
  const updateFilters = (patch: Partial<ModelHubDirectoryFilters>) =>
    updateUrlState(current => ({
      ...current,
      filters: { ...current.filters, ...patch },
      page: 1,
    }))

  return {
    filters,
    updateFilters,
    resetFilters: () => updateUrlState(current => ({
      ...current,
      filters: modelHubDirectoryDefaults,
      page: 1,
    })),
    view,
    setView: (nextView: ModelView) => updateUrlState(current => ({
      ...current,
      view: nextView,
    })),
    page,
    setPage: (nextPage: number) => updateUrlState(current => ({
      ...current,
      page: nextPage,
    })),
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

function useBenchmarkExplorer(
  benchmarkState: ModelHubBenchmarkUrlState,
  updateUrlState: UpdateModelHubUrlState,
) {
  const allCharts = useMemo(() => modelHubBenchmarkCharts(catalog), [])
  const colors = useMemo(
    () =>
      modelHubChartColors(
        catalog.models.map(model => model.id),
        catalog.models.map(model => model.id),
      ),
    [],
  )
  const { filter: benchmarkFilter, query, publisher } = benchmarkState
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
    setFilter: (filter: string) => updateUrlState(current => ({
      ...current,
      benchmark: { ...current.benchmark, filter },
    })),
    query,
    setQuery: (nextQuery: string) => updateUrlState(current => ({
      ...current,
      benchmark: { ...current.benchmark, query: nextQuery },
    })),
    publisher: activePublisher,
    setPublisher: (nextPublisher: string) => updateUrlState(current => ({
      ...current,
      benchmark: { ...current.benchmark, publisher: nextPublisher },
    })),
    publishers,
  }
}

export default function ModelsPage() {
  const history = useHistory()
  const location = useLocation()
  const urlState = useMemo(
    () => parseModelHubUrlState(location.search),
    [location.search],
  )
  const updateUrlState = useCallback<UpdateModelHubUrlState>((updater, push = false) => {
    const current = parseModelHubUrlState(location.search)
    const nextSearch = serializeModelHubUrlState(updater(current), location.search)
    const nextLocation = `${location.pathname}${nextSearch}${location.hash}`
    if (push) history.push(nextLocation)
    else history.replace(nextLocation)
  }, [history, location.hash, location.pathname, location.search])
  const directory = useCatalogDirectory(urlState, updateUrlState)
  const modelByID = useMemo(() => new Map(catalog.models.map(model => [model.id, model])), [])
  const reasoningFamilies = useMemo(
    () => new Map(catalog.reasoning_families.map(family => [family.id, family])),
    [],
  )
  const benchmark = useBenchmarkExplorer(urlState.benchmark, updateUrlState)
  const selectedModel = urlState.selectedModelID
    ? modelByID.get(urlState.selectedModelID)
    : undefined
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
  const selectModel = useCallback((selectedModelID: string) => {
    updateUrlState(current => ({ ...current, selectedModelID }), true)
  }, [updateUrlState])
  const closeModelDetail = useCallback(() => {
    updateUrlState(current => ({ ...current, selectedModelID: null }))
  }, [updateUrlState])
  const sections = useMemo(
    () => MODEL_HUB_SECTIONS.map(section => ({
      ...section,
      to: `${location.pathname}${location.search}#${section.key}`,
    })),
    [location.pathname, location.search],
  )
  const physicalModels = catalog.models.filter(model => model.kind === 'physical').length
  const virtualModels = catalog.models.length - physicalModels
  const creators = new Set(
    catalog.models.filter(model => model.kind === 'physical').map(model => model.publisher),
  ).size
  const evaluations = modelHubPublicEvaluations(catalog.evaluations).length

  return (
    <Layout
      title="Model Hub"
      description="Built-in models and exact benchmark evidence for vLLM Semantic Router."
    >
      <div className={styles.page}>
        <BrowseLayout
          activeKey={modelHubActiveSection(location.hash)}
          description="Models, ready to route."
          eyebrow={<Translate id="models.layout.eyebrow">Catalog</Translate>}
          groups={[
            {
              key: 'explore',
              label: 'Explore',
              items: sections,
            },
          ]}
          sidebarLabel="Model Hub sections"
          title="Model Hub"
          actions={(
            <dl className={styles.stats}>
              <div>
                <dd>{catalog.models.length}</dd>
                <dt>models</dt>
              </div>
              <div>
                <dd>{creators}</dd>
                <dt>creators</dt>
              </div>
              <div>
                <dd>{catalog.providers.length}</dd>
                <dt>providers</dt>
              </div>
              <div>
                <dd>{evaluations}</dd>
                <dt>evaluations</dt>
              </div>
            </dl>
          )}
        >
          <section id="arena" className={styles.section} aria-labelledby="arena-heading">
            <header className={styles.sectionHeading}>
              <h2 id="arena-heading">Arena</h2>
              <span>One index · same rules for single and virtual models</span>
            </header>
            <ModelHubArena
              catalog={catalog}
              scope={urlState.arenaScope}
              setScope={arenaScope => updateUrlState(current => ({ ...current, arenaScope }))}
              layer={urlState.arenaLayer}
              setLayer={arenaLayer => updateUrlState(current => ({ ...current, arenaLayer }))}
              capability={urlState.arenaCapability}
              setCapability={arenaCapability => updateUrlState(current => ({
                ...current,
                arenaCapability,
                arenaLayer: 'capabilities',
              }))}
              benchmark={urlState.arenaBenchmark}
              setBenchmark={arenaBenchmark => updateUrlState(current => ({
                ...current,
                arenaBenchmark,
                arenaLayer: 'benchmarks',
              }))}
              selectModel={selectModel}
            />
          </section>

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
              selectModel={selectModel}
            />
          </section>

          <section id="benchmarks" className={styles.section} aria-labelledby="benchmarks-heading">
            <header className={styles.sectionHeading}>
              <h2 id="benchmarks-heading">Benchmarks</h2>
              <span>Exact published results</span>
            </header>
            <ModelHubBenchmark {...benchmark} selectModel={selectModel} />
          </section>

          <section id="providers" className={styles.section} aria-labelledby="providers-heading">
            <header className={styles.sectionHeading}>
              <h2 id="providers-heading">Providers</h2>
              <span>
                {catalog.providers.length}
                {' '}
                available providers
              </span>
            </header>
            <ModelHubProviders providers={catalog.providers} protocols={catalog.protocols} />
          </section>
        </BrowseLayout>

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
      </div>
    </Layout>
  )
}
