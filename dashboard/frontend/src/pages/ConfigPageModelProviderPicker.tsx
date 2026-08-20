import ModelProviderLogo from './ModelProviderLogo'
import {
  FEATURED_MODEL_PROVIDERS,
  MODEL_PROVIDERS,
  type ModelProviderDefinition,
} from './modelProviderCatalog'
import styles from './ConfigPageAddModelsDialog.module.css'

interface Props {
  search: string
  providers: ModelProviderDefinition[]
  titleId: string
  onSearch: (value: string) => void
  onChoose: (provider: ModelProviderDefinition) => void
}

const providerGroups = (providers: ModelProviderDefinition[]) =>
  ['Local runtimes', 'Model APIs', 'Private gateways']
    .map((category) => ({
      category,
      providers: providers.filter((provider) => provider.category === category),
    }))
    .filter((group) => group.providers.length > 0)

export default function ConfigPageModelProviderPicker({
  search,
  providers,
  titleId,
  onSearch,
  onChoose,
}: Props) {
  return (
    <div className={`${styles.body} ${styles.providerBody}`}>
      <div className={styles.providerSearch}>
        <span aria-hidden="true">⌕</span>
        <input
          type="search"
          value={search}
          onChange={(event) => onSearch(event.target.value)}
          placeholder="Search providers"
          aria-label="Search model providers"
          autoFocus
        />
        <small>{MODEL_PROVIDERS.length} ready</small>
      </div>

      {!search.trim() ? (
        <section className={styles.featuredProviders} aria-labelledby={`${titleId}-featured`}>
          <div className={styles.sectionLabel} id={`${titleId}-featured`}>
            Start here
          </div>
          <div className={styles.featuredGrid}>
            {FEATURED_MODEL_PROVIDERS.map((provider) => (
              <button
                type="button"
                className={styles.featuredProvider}
                key={provider.id}
                onClick={() => onChoose(provider)}
              >
                <ModelProviderLogo provider={provider.id} size="large" />
                <span>
                  <strong>{provider.name}</strong>
                  <small>{provider.description}</small>
                </span>
                <b aria-hidden="true">→</b>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <div className={styles.providerCatalog}>
        {providerGroups(providers).map((group) => (
          <section key={group.category} className={styles.providerGroup}>
            <div className={styles.sectionLabel}>{group.category}</div>
            <div className={styles.providerList}>
              {group.providers.map((provider) => (
                <button
                  type="button"
                  className={styles.providerOption}
                  key={provider.id}
                  onClick={() => onChoose(provider)}
                >
                  <ModelProviderLogo provider={provider.id} />
                  <span>
                    <strong>{provider.name}</strong>
                    <small>{provider.description}</small>
                  </span>
                  <b aria-hidden="true">›</b>
                </button>
              ))}
            </div>
          </section>
        ))}
        {providers.length === 0 && search.trim() ? (
          <div className={styles.empty}>No providers match “{search}”.</div>
        ) : null}
      </div>
    </div>
  )
}
