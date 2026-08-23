import ProductIcon from '../components/ProductIcon'
import type { DiscoveredProviderModel, ProviderCatalogItem } from '../utils/providerCatalogApi'
import ModelProviderLogo from './ModelProviderLogo'
import styles from './ConfigPageAddModelsDialog.module.css'

interface Props {
  models: DiscoveredProviderModel[]
  provider: ProviderCatalogItem
  selected: Set<string>
  onSelected: (value: Set<string>) => void
  existing: Set<string>
  prefix: string
  search: string
  loading: boolean
  hasMore: boolean
  onSearch: (value: string) => void
  onSubmitSearch: () => void
  onLoadMore: () => void
}

export default function ConfigPageModelDiscoveryResults({
  models,
  provider,
  selected,
  onSelected,
  existing,
  prefix,
  search,
  loading,
  hasMore,
  onSearch,
  onSubmitSearch,
  onLoadMore,
}: Props) {
  const available = models.filter((model) => !existing.has(prefix + model.providerModelId))
  const allSelected =
    available.length > 0 && available.every((model) => selected.has(model.catalogItemId))

  const toggleAll = () => {
    const next = new Set(selected)
    available.forEach((model) =>
      allSelected ? next.delete(model.catalogItemId) : next.add(model.catalogItemId),
    )
    onSelected(next)
  }

  const toggle = (catalogItemId: string) => {
    const next = new Set(selected)
    if (next.has(catalogItemId)) next.delete(catalogItemId)
    else next.add(catalogItemId)
    onSelected(next)
  }

  return (
    <section className={styles.modelSection} aria-label="Available models" aria-busy={loading}>
      <div className={styles.modelToolbar}>
        <div>
          <strong>{models.length} models found</strong>
          <span>{selected.size} selected</span>
        </div>
        <div className={styles.modelTools}>
          <input
            type="search"
            value={search}
            onChange={(event) => onSearch(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                onSubmitSearch()
              }
            }}
            placeholder="Search models"
            aria-label="Search discovered models"
          />
          <button type="button" onClick={onSubmitSearch} disabled={loading}>
            <ProductIcon name="search" aria-hidden="true" />
            Search
          </button>
          <button
            type="button"
            onClick={toggleAll}
            disabled={available.length === 0}
            aria-pressed={allSelected}
          >
            <ProductIcon name={allSelected ? 'close' : 'check'} aria-hidden="true" />
            {allSelected ? 'Clear' : 'Select all'}
          </button>
        </div>
      </div>
      <div className={styles.modelList}>
        {models.map((model) => {
          const logicalName = prefix + model.providerModelId
          const alreadyAdded = existing.has(logicalName)
          return (
            <label
              key={model.catalogItemId}
              className={`${styles.modelOption} ${alreadyAdded ? styles.disabledOption : ''}`}
            >
              <input
                type="checkbox"
                checked={selected.has(model.catalogItemId)}
                disabled={alreadyAdded}
                onChange={() => toggle(model.catalogItemId)}
              />
              <ProductIcon className={styles.checkmark} name="check" aria-hidden="true" />
              <ModelProviderLogo
                icon={provider.display.icon}
                name={provider.display.name}
                monogram={provider.display.monogram}
                accent={provider.display.accent}
                size="small"
              />
              <span className={styles.modelName}>
                <strong title={logicalName}>{logicalName}</strong>
                <small>{model.displayName}</small>
              </span>
              {alreadyAdded ? <span className={styles.addedBadge}>Added</span> : null}
            </label>
          )
        })}
        {!loading && models.length === 0 ? (
          <div className={styles.empty}>No models found.</div>
        ) : null}
        {hasMore ? (
          <button
            type="button"
            className={styles.loadMoreButton}
            onClick={onLoadMore}
            disabled={loading}
          >
            {!loading ? <ProductIcon name="chevron-down" aria-hidden="true" /> : null}
            {loading ? 'Loading…' : 'More models'}
          </button>
        ) : null}
      </div>
    </section>
  )
}
