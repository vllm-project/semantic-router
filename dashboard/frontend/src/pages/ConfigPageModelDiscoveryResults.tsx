import type { DiscoveredProviderModel } from '../utils/modelDiscoveryApi'
import ModelProviderLogo from './ModelProviderLogo'
import styles from './ConfigPageAddModelsDialog.module.css'

interface Props {
  models: DiscoveredProviderModel[]
  selected: Set<string>
  onSelected: (value: Set<string>) => void
  existing: Set<string>
  prefix: string
  providerId: string
  search: string
  onSearch: (value: string) => void
}

export default function ConfigPageModelDiscoveryResults({
  models,
  selected,
  onSelected,
  existing,
  prefix,
  providerId,
  search,
  onSearch,
}: Props) {
  if (models.length === 0) return null
  const query = search.trim().toLowerCase()
  const visibleModels = query
    ? models.filter((model) => model.id.toLowerCase().includes(query))
    : models
  const availableVisible = visibleModels.filter((model) => !existing.has(prefix + model.id))
  const allVisibleSelected =
    availableVisible.length > 0 && availableVisible.every((model) => selected.has(model.id))

  const toggleAll = () => {
    const next = new Set(selected)
    availableVisible.forEach((model) =>
      allVisibleSelected ? next.delete(model.id) : next.add(model.id),
    )
    onSelected(next)
  }

  const toggle = (modelId: string) => {
    const next = new Set(selected)
    if (next.has(modelId)) next.delete(modelId)
    else next.add(modelId)
    onSelected(next)
  }

  return (
    <section className={styles.modelSection} aria-label="Available models">
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
            placeholder="Search models"
            aria-label="Search discovered models"
          />
          <button type="button" onClick={toggleAll}>
            {allVisibleSelected ? 'Clear' : 'Select all'}
          </button>
        </div>
      </div>
      <div className={styles.modelList}>
        {visibleModels.map((model) => {
          const logicalName = prefix + model.id
          const alreadyAdded = existing.has(logicalName)
          return (
            <label
              key={model.id}
              className={`${styles.modelOption} ${alreadyAdded ? styles.disabledOption : ''}`}
            >
              <input
                type="checkbox"
                checked={selected.has(model.id)}
                disabled={alreadyAdded}
                onChange={() => toggle(model.id)}
              />
              <span className={styles.checkmark} aria-hidden="true">
                ✓
              </span>
              <ModelProviderLogo provider={providerId} size="small" />
              <span className={styles.modelName}>
                <strong>{logicalName}</strong>
                {model.ownedBy ? <small>{model.ownedBy}</small> : null}
              </span>
              {alreadyAdded ? <span className={styles.addedBadge}>Added</span> : null}
            </label>
          )
        })}
        {visibleModels.length === 0 ? <div className={styles.empty}>No matches</div> : null}
      </div>
    </section>
  )
}
