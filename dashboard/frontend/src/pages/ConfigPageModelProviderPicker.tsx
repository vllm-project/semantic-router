import type { ProviderCatalogItem } from '../utils/providerCatalogApi'
import ProductIcon from '../components/ProductIcon'
import ModelProviderLogo from './ModelProviderLogo'
import styles from './ConfigPageAddModelsDialog.module.css'

interface Props {
  search: string
  providers: ProviderCatalogItem[]
  categories: string[]
  loading: boolean
  hasMore: boolean
  titleId: string
  onSearch: (value: string) => void
  onChoose: (provider: ProviderCatalogItem) => void
  onLoadMore: () => void
}

const providerGroups = (providers: ProviderCatalogItem[], categories: string[]) => {
  const ordered = [...categories]
  for (const provider of providers) {
    if (!ordered.includes(provider.display.category)) ordered.push(provider.display.category)
  }
  return ordered
    .map((category) => ({
      category,
      providers: providers.filter((provider) => provider.display.category === category),
    }))
    .filter((group) => group.providers.length > 0)
}

const ProviderMark = ({
  provider,
  size = 'medium',
}: {
  provider: ProviderCatalogItem
  size?: 'medium' | 'large'
}) => (
  <ModelProviderLogo
    icon={provider.display.icon}
    name={provider.display.name}
    monogram={provider.display.monogram}
    accent={provider.display.accent}
    size={size}
  />
)

export default function ConfigPageModelProviderPicker({
  search,
  providers,
  categories,
  loading,
  hasMore,
  titleId,
  onSearch,
  onChoose,
  onLoadMore,
}: Props) {
  const featured = search.trim() ? [] : providers.slice(0, 4)
  const featuredIds = new Set(featured.map((provider) => provider.providerId))
  const catalog = providers.filter((provider) => !featuredIds.has(provider.providerId))

  return (
    <div className={`${styles.body} ${styles.providerBody}`}>
      <div className={styles.providerSearch}>
        <ProductIcon name="search" />
        <input
          type="search"
          value={search}
          onChange={(event) => onSearch(event.target.value)}
          placeholder="Search providers"
          aria-label="Search model providers"
          data-dialog-initial-focus
          autoFocus
        />
        <small>{loading ? 'Loading…' : `${providers.length} shown`}</small>
      </div>

      {featured.length > 0 ? (
        <section className={styles.featuredProviders} aria-labelledby={`${titleId}-featured`}>
          <div className={styles.sectionLabel} id={`${titleId}-featured`}>
            Start here
          </div>
          <div className={styles.featuredGrid}>
            {featured.map((provider) => (
              <button
                type="button"
                className={styles.featuredProvider}
                key={provider.providerId}
                onClick={() => onChoose(provider)}
              >
                <ProviderMark provider={provider} size="large" />
                <span>
                  <strong>{provider.display.name}</strong>
                  <small>{provider.display.description}</small>
                </span>
                <ProductIcon name="chevron-right" />
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <div className={styles.providerCatalog} aria-busy={loading}>
        {providerGroups(catalog, categories).map((group) => (
          <section key={group.category} className={styles.providerGroup}>
            <div className={styles.sectionLabel}>{group.category}</div>
            <div className={styles.providerList}>
              {group.providers.map((provider) => (
                <button
                  type="button"
                  className={styles.providerOption}
                  key={provider.providerId}
                  onClick={() => onChoose(provider)}
                >
                  <ProviderMark provider={provider} />
                  <span>
                    <strong>{provider.display.name}</strong>
                    <small>{provider.display.description}</small>
                  </span>
                  <ProductIcon name="chevron-right" />
                </button>
              ))}
            </div>
          </section>
        ))}
        {!loading && providers.length === 0 ? (
          <div className={styles.empty}>
            {search.trim() ? `No providers match “${search.trim()}”.` : 'No providers available.'}
          </div>
        ) : null}
        {hasMore ? (
          <button
            type="button"
            className={styles.loadMoreButton}
            onClick={onLoadMore}
            disabled={loading}
          >
            {loading ? 'Loading…' : 'Show more providers'}
          </button>
        ) : null}
      </div>
    </div>
  )
}
