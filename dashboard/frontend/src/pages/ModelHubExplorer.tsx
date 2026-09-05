import React from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { HubPagination } from './ModelHubComponents'
import { ModelDetail } from './ModelHubDetail'
import { HubFilters } from './ModelHubFilters'
import { BenchmarkExplorer, EmptyResults, ModelCards, ModelTable } from './ModelHubViews'
import type { useModelHubPageController } from './modelHubPageController'
import styles from './ModelHubPage.module.css'

type ModelHubPageController = ReturnType<typeof useModelHubPageController>

const ModelHubCatalogPanel: React.FC<{
  catalog: BuiltInModelCatalog
  hub: ModelHubPageController
}> = ({ catalog, hub }) => (
  <div className={styles.catalogPanel}>
    {hub.rows.length === 0 ? <EmptyResults /> : null}
    {hub.rows.length && hub.view === 'table' ? (
      <ModelTable rows={hub.pagination.items} selected={hub.selected} select={hub.selectModel} />
    ) : null}
    {hub.rows.length && hub.view === 'cards' ? (
      <ModelCards rows={hub.pagination.items} selected={hub.selected} select={hub.selectModel} />
    ) : null}
    {hub.rows.length && hub.view === 'benchmarks' ? (
      <BenchmarkExplorer
        catalog={catalog}
        rows={hub.rows}
        selected={hub.selected}
        select={hub.selectModel}
        compact={hub.compact}
      />
    ) : null}
    {hub.rows.length && hub.view !== 'benchmarks' ? (
      <HubPagination
        pagination={hub.pagination}
        setPage={hub.setPage}
        setPageSize={hub.updatePageSize}
      />
    ) : null}
  </div>
)

const ModelHubDetailLayer: React.FC<{
  catalog: BuiltInModelCatalog
  hub: ModelHubPageController
}> = ({ catalog, hub }) => (
  <div
    ref={hub.detailLayerRef}
    className={`${styles.detailLayer} ${hub.detailOpen ? styles.detailLayerOpen : ''}`}
    role={hub.compact && hub.detailOpen ? 'presentation' : undefined}
  >
    <button
      className={styles.detailDismiss}
      type="button"
      aria-hidden="true"
      tabIndex={-1}
      onClick={hub.closeDetail}
    />
    <ModelDetail
      key={hub.selected?.model.id ?? 'empty'}
      row={hub.selected}
      catalog={catalog}
      modal={hub.compact && hub.detailOpen}
      closeButtonRef={hub.detailCloseRef}
      onClose={hub.closeDetail}
    />
  </div>
)

export const ModelHubExplorer: React.FC<{
  catalog: BuiltInModelCatalog
  hub: ModelHubPageController
}> = ({ catalog, hub }) => (
  <section className={styles.explorer} id="catalog-explorer" aria-label="Built-in models">
    <div className={styles.explorerHeading}>
      <div>
        <span>Catalog explorer</span>
        <h2>Built-in models, one source of truth</h2>
      </div>
      <p>Filter the inventory, inspect one model, or compare a single benchmark setup.</p>
    </div>
    <HubFilters
      filters={hub.filters}
      creators={hub.creators}
      providers={hub.providers}
      capabilities={hub.capabilities}
      view={hub.view}
      update={hub.updateFilters}
      setView={hub.setView}
      reset={hub.resetFilters}
    />
    <div className={styles.resultHeader}>
      <div>
        <strong>{hub.rows.length.toLocaleString()}</strong>
        <span> matching models</span>
      </div>
      <span>
        Benchmark results are shown only when an exact published evaluation record exists.
      </span>
    </div>
    <div className={styles.workspace}>
      <ModelHubCatalogPanel catalog={catalog} hub={hub} />
      <ModelHubDetailLayer catalog={catalog} hub={hub} />
    </div>
  </section>
)
