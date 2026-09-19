import React, { useEffect, useMemo, useState } from 'react'
import styles from './RouterModelInventory.module.css'
import {
  getLoadedModelCount,
  getPreviewRouterModels,
  getRouterModelAnchor,
  getRouterModelConsumers,
  getRouterModelResources,
  getRouterModelState,
  getRouterModelStateLabel,
  getTotalKnownModelCount,
  sortRouterModels,
  type RouterModelInfo,
  type RouterModelsInfo,
} from '../utils/routerRuntime'
import {
  formatRouterModelLabel,
  formatRouterModelTokens,
  getRouterModelInputLimits,
  getRouterModelKind,
  getRouterModelDevice,
  getRouterModelDisplayName,
  getRouterModelPreviewName,
} from './routerModelPresentation'
import { buildRouterModelDetailSections, type RouterModelDetailRow } from './routerModelDetails'
import {
  clampInventoryPage,
  filterAndSortRouterModels,
  type ModelInventorySort,
  type ModelInventoryStateFilter,
} from './routerModelInventorySupport'

interface RouterModelInventoryProps {
  modelsInfo?: RouterModelsInfo | null
  mode?: 'preview' | 'full' | 'detail'
  previewLimit?: number
  showSummary?: boolean
  emptyMessage?: string
  onSelectModel?: (model: RouterModelInfo) => void
}

function getStateChipClass(model: RouterModelInfo): string {
  switch (getRouterModelState(model)) {
    case 'ready':
      return styles.stateReady
    case 'downloading':
      return styles.stateDownloading
    case 'pending':
    case 'initializing':
      return styles.stateWarm
    default:
      return styles.stateMuted
  }
}

function getCardToneClass(model: RouterModelInfo): string {
  switch (getRouterModelState(model)) {
    case 'ready':
      return styles.cardReady
    case 'downloading':
    case 'pending':
    case 'initializing':
      return styles.cardWarm
    default:
      return styles.cardMuted
  }
}

function renderDetailRows(rows: RouterModelDetailRow[]): JSX.Element {
  return (
    <dl className={styles.detailList}>
      {rows.map((row) => (
        <div key={`${row.label}-${row.value}`} className={styles.detailRow}>
          <dt className={styles.detailLabel}>{row.label}</dt>
          <dd className={styles.detailValue}>{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

const DeviceMark: React.FC<{ model: RouterModelInfo }> = ({ model }) => {
  const device = getRouterModelDevice(model)
  return (
    <span className={styles.deviceMark}>
      {device.isAmd && <img src="/amd-logo.png" alt="AMD GPU" className={styles.platformLogo} />}
      <span>{device.label}</span>
    </span>
  )
}

const PreviewCardBody: React.FC<{ model: RouterModelInfo; consumers: RouterModelInfo[] }> = ({
  model,
  consumers,
}) => {
  const limits = getRouterModelInputLimits(model)
  const name = getRouterModelPreviewName(model)
  const budgetVaries = consumers.some((consumer) => {
    const other = getRouterModelInputLimits(consumer)
    return (other.document ?? other.input) !== (limits.document ?? limits.input)
  })
  return (
    <>
      <div className={styles.previewTopRow}>
        <span className={styles.previewPurpose}>{getRouterModelKind(model)}</span>
        <span className={`${styles.previewState} ${getStateChipClass(model)}`}>
          {getRouterModelStateLabel(model)}
        </span>
      </div>
      <div className={styles.previewIdentity}>
        <h3 className={styles.previewModelId}>{name.title}</h3>
        {name.subtitle && <p className={styles.previewSubtitle}>{name.subtitle}</p>}
      </div>
      <div className={styles.previewFooter}>
        {limits.window !== undefined && (
          <p className={styles.previewContext}>Window: {formatRouterModelTokens(limits.window)}</p>
        )}
        {limits.input !== undefined ? (
          <p className={styles.previewContext}>
            {limits.overflow === 'window' ? 'Document budget' : 'Input budget'}:{' '}
            {budgetVaries
              ? 'Varies by consumer'
              : formatRouterModelTokens(limits.document ?? limits.input)}
          </p>
        ) : (
          <p className={styles.previewContext}>
            {limits.publishedContext !== undefined ? 'Published model context' : 'Input budget'}:{' '}
            {formatRouterModelTokens(limits.publishedContext)}
          </p>
        )}
        {consumers.length > 1 && (
          <p className={styles.previewContext}>Shared by {consumers.length} consumers</p>
        )}
      </div>
    </>
  )
}

const FullCardBody: React.FC<{ model: RouterModelInfo; consumers: RouterModelInfo[] }> = ({
  model,
  consumers,
}) => {
  const sections = buildRouterModelDetailSections(model, consumers)

  return (
    <>
      <div className={styles.titleBlock}>
        <div className={styles.titleRow}>
          <h3 className={styles.modelName}>{getRouterModelDisplayName(model)}</h3>
          <span className={`${styles.stateChip} ${getStateChipClass(model)}`}>
            {getRouterModelStateLabel(model)}
          </span>
        </div>
        <p className={styles.modelSubtitle}>{getRouterModelKind(model)}</p>
      </div>

      <div className={styles.detailSections}>
        {sections.map((section) =>
          section.collapsible ? (
            <details
              key={section.title}
              className={`${styles.detailSection} ${styles.detailSectionWide} ${styles.technicalDetails}`}
            >
              <summary className={styles.detailSectionTitle}>{section.title}</summary>
              {renderDetailRows(section.rows)}
            </details>
          ) : (
            <section
              key={section.title}
              className={`${styles.detailSection} ${section.wide ? styles.detailSectionWide : ''}`}
            >
              <h4 className={styles.detailSectionTitle}>{section.title}</h4>
              {renderDetailRows(section.rows)}
            </section>
          ),
        )}
      </div>

      <div
        className={`${styles.detailFooter} ${!model.registry?.model_card_url ? styles.detailFooterEnd : ''}`}
      >
        {model.registry?.model_card_url ? (
          <a
            className={styles.cardLink}
            href={model.registry.model_card_url}
            target="_blank"
            rel="noreferrer"
          >
            Open model card
          </a>
        ) : null}

        <DeviceMark model={model} />
      </div>
    </>
  )
}

const RouterModelInventory: React.FC<RouterModelInventoryProps> = ({
  modelsInfo,
  mode = 'full',
  previewLimit,
  showSummary = true,
  emptyMessage = 'No router model metadata is available yet.',
  onSelectModel,
}) => {
  const [query, setQuery] = useState('')
  const [stateFilter, setStateFilter] = useState<ModelInventoryStateFilter>('all')
  const [sort, setSort] = useState<ModelInventorySort>('state')
  const [page, setPage] = useState(1)
  const pageSize = 8
  const resources = useMemo(
    () => getRouterModelResources(sortRouterModels(modelsInfo?.models ?? [])),
    [modelsInfo?.models],
  )
  const allModels = useMemo(() => resources.map(({ model }) => model), [resources])
  const filteredModels = useMemo(
    () =>
      filterAndSortRouterModels(
        resources
          .filter(
            ({ consumers }) =>
              filterAndSortRouterModels(consumers, query, 'all', 'state').length > 0,
          )
          .map(({ model }) => model),
        '',
        stateFilter,
        sort,
      ),
    [resources, query, sort, stateFilter],
  )
  const totalPages = Math.max(1, Math.ceil(filteredModels.length / pageSize))
  const currentPage = clampInventoryPage(page, filteredModels.length, pageSize)
  const models =
    mode === 'preview'
      ? getPreviewRouterModels(modelsInfo, previewLimit)
      : filteredModels.slice((currentPage - 1) * pageSize, currentPage * pageSize)

  const loadedCount = getLoadedModelCount(modelsInfo)
  const totalCount = getTotalKnownModelCount(modelsInfo)
  const phase = modelsInfo?.summary?.phase
  const summaryMessage = modelsInfo?.summary?.message

  useEffect(() => {
    setPage(1)
  }, [query, sort, stateFilter])

  useEffect(() => {
    if (mode !== 'full' || !window.location.hash) return
    const anchor = decodeURIComponent(window.location.hash.slice(1))
    const modelIndex = filteredModels.findIndex((model) => getRouterModelAnchor(model) === anchor)
    if (modelIndex >= 0) setPage(Math.floor(modelIndex / pageSize) + 1)
  }, [filteredModels, mode])

  if (allModels.length === 0) {
    return <div className={styles.empty}>{emptyMessage}</div>
  }

  return (
    <div className={styles.inventory}>
      {mode === 'full' && (
        <>
          {showSummary && (
            <div className={styles.summaryRow}>
              <div className={styles.summaryStat}>
                <span className={styles.summaryLabel}>Runtimes ready</span>
                <span className={styles.summaryValue}>
                  {loadedCount}/{totalCount}
                </span>
              </div>
              {phase && (
                <div className={styles.summaryStat}>
                  <span className={styles.summaryLabel}>Phase</span>
                  <span className={styles.summaryValue}>{formatRouterModelLabel(phase)}</span>
                </div>
              )}
              {summaryMessage && <p className={styles.summaryMessage}>{summaryMessage}</p>}
            </div>
          )}

          <div className={styles.inventoryToolbar}>
            <label className={styles.searchField}>
              <span className={styles.srOnly}>Search models</span>
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search name, type, repository, or tag"
              />
            </label>
            <label className={styles.selectField}>
              <span>Status</span>
              <select
                value={stateFilter}
                onChange={(event) =>
                  setStateFilter(event.target.value as ModelInventoryStateFilter)
                }
              >
                <option value="all">All statuses</option>
                <option value="ready">Ready</option>
                <option value="loading">Loading</option>
                <option value="not_loaded">Not loaded</option>
              </select>
            </label>
            <label className={styles.selectField}>
              <span>Sort</span>
              <select
                value={sort}
                onChange={(event) => setSort(event.target.value as ModelInventorySort)}
              >
                <option value="state">Readiness</option>
                <option value="name">Name</option>
                <option value="type">Type</option>
              </select>
            </label>
            <span className={styles.resultCount} aria-live="polite">
              {filteredModels.length} of {allModels.length} runtimes
            </span>
          </div>
        </>
      )}

      {mode === 'full' && filteredModels.length === 0 ? (
        <div className={styles.empty}>
          No models match the current search and status filters.
          <button
            type="button"
            className={styles.clearFilters}
            onClick={() => {
              setQuery('')
              setStateFilter('all')
              setSort('state')
            }}
          >
            Clear filters
          </button>
        </div>
      ) : (
        <>
          <div
            className={mode === 'preview' ? styles.previewGrid : styles.fullGrid}
            data-testid={`router-model-grid-${mode}`}
          >
            {models.map((model) => {
              const consumers = getRouterModelConsumers(modelsInfo?.models ?? [], model)
              const className = [
                styles.card,
                mode === 'preview' ? styles.previewCard : styles.detailCard,
                getCardToneClass(model),
                onSelectModel && mode === 'preview' ? styles.cardInteractive : '',
              ]
                .filter(Boolean)
                .join(' ')

              const cardContent =
                mode === 'preview' ? (
                  <PreviewCardBody model={model} consumers={consumers} />
                ) : (
                  <FullCardBody model={model} consumers={consumers} />
                )

              if (onSelectModel && mode === 'preview') {
                return (
                  <button
                    key={getRouterModelAnchor(model)}
                    type="button"
                    className={className}
                    data-testid={`router-model-${mode}-${model.name}`}
                    onClick={() => onSelectModel(model)}
                  >
                    {cardContent}
                  </button>
                )
              }

              return (
                <article
                  key={getRouterModelAnchor(model)}
                  id={mode === 'full' ? getRouterModelAnchor(model) : undefined}
                  className={className}
                  data-testid={`router-model-${mode}-${model.name}`}
                >
                  {cardContent}
                </article>
              )
            })}
          </div>

          {mode === 'full' && totalPages > 1 && (
            <nav className={styles.pagination} aria-label="Model inventory pages">
              <button
                type="button"
                disabled={currentPage === 1}
                onClick={() => setPage((value) => Math.max(1, value - 1))}
              >
                Previous
              </button>
              <span>
                Page {currentPage} of {totalPages}
              </span>
              <button
                type="button"
                disabled={currentPage === totalPages}
                onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
              >
                Next
              </button>
            </nav>
          )}
        </>
      )}
    </div>
  )
}

export default RouterModelInventory
