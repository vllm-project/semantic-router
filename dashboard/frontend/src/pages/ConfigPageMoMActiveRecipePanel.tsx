import MarkdownRenderer from '../components/MarkdownRenderer'
import type { RecipeDescriptor, RecipeMetadataAuthor } from '../types/recipe'
import styles from './ConfigPageMoMWorkspace.module.css'

const SOURCE_LABELS = {
  metadata: 'metadata.yaml',
  config: 'config.yaml',
  probes: 'probes.yaml',
  dsl: 'recipe.dsl',
  readme: 'README.md',
} as const

interface ConfigPageMoMActiveRecipePanelProps {
  recipe: RecipeDescriptor | null
  loading: boolean
  error: string | null
  onRetry: () => void
}

function formatAuthor(author: RecipeMetadataAuthor | string): string {
  return typeof author === 'string' ? author : author.name
}

function shortDigest(digest?: string): string {
  if (!digest) return 'Not available'
  return digest.length > 18 ? `${digest.slice(0, 18)}…` : digest
}

function safeExternalUrl(value: string): string | null {
  try {
    const url = new URL(value)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.toString() : null
  } catch {
    return null
  }
}

export default function ConfigPageMoMActiveRecipePanel({
  recipe,
  loading,
  error,
  onRetry,
}: ConfigPageMoMActiveRecipePanelProps) {
  if (loading) {
    return (
      <div className={styles.asyncState} role="status">
        Loading active Recipe metadata…
      </div>
    )
  }

  if (error || !recipe) {
    return (
      <div className={`${styles.asyncState} ${styles.errorState}`} role="alert">
        <strong>Recipe metadata is unavailable</strong>
        <span>{error || 'The Dashboard did not return an active Recipe.'}</span>
        <button type="button" onClick={onRetry}>
          Retry active Recipe metadata
        </button>
      </div>
    )
  }

  if (!recipe.managed) {
    return (
      <div className={styles.unmanagedState} role="status">
        <span className={styles.statusBadge}>Unmanaged configuration</span>
        <h2>This config is not a managed Recipe</h2>
        <p>
          Routing remains available, but metadata, probes, and source attribution require the
          five-file Recipe directory contract.
        </p>
        {recipe.source_health.issues.length > 0 ? (
          <ul>
            {recipe.source_health.issues.map((issue, index) => (
              <li key={`${index}-${issue}`}>{issue}</li>
            ))}
          </ul>
        ) : null}
      </div>
    )
  }

  if (!recipe.metadata) {
    return (
      <div className={`${styles.unmanagedState} ${styles.errorState}`} role="alert">
        <span className={styles.statusBadge}>Invalid managed Recipe</span>
        <h2>metadata.yaml could not be loaded</h2>
        <p>Fix the active Recipe source before using its probes or distribution identity.</p>
        {recipe.source_health.issues.length > 0 ? (
          <ul>
            {recipe.source_health.issues.map((issue, index) => (
              <li key={`${index}-${issue}`}>{issue}</li>
            ))}
          </ul>
        ) : null}
      </div>
    )
  }

  const { metadata } = recipe
  const links = Object.entries(metadata.links ?? {}).flatMap(([label, value]) => {
    const href = safeExternalUrl(value)
    return href ? [{ label, href }] : []
  })

  return (
    <div className={styles.overviewGrid}>
      <section className={`${styles.surfaceCard} ${styles.identityCard}`}>
        <div className={styles.cardHeader}>
          <div>
            <span className={styles.eyebrow}>Managed Recipe</span>
            <h2>{metadata.name}</h2>
          </div>
          <span
            className={`${styles.statusBadge} ${recipe.source_health.status === 'ready' ? styles.readyBadge : styles.warningBadge}`}
          >
            {recipe.source_health.status}
          </span>
        </div>
        <p className={styles.description}>{metadata.description}</p>
        <dl className={styles.definitionGrid}>
          <div>
            <dt>ID</dt>
            <dd>{metadata.id}</dd>
          </div>
          <div>
            <dt>Version</dt>
            <dd>{metadata.version}</dd>
          </div>
          <div>
            <dt>License</dt>
            <dd>{metadata.license || 'Not specified'}</dd>
          </div>
          <div>
            <dt>Authors</dt>
            <dd>{metadata.authors?.map(formatAuthor).join(', ') || 'Not specified'}</dd>
          </div>
        </dl>
        {metadata.tags?.length ? (
          <div className={styles.chips} aria-label="Recipe tags">
            {metadata.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        ) : null}
        {links.length ? (
          <div className={styles.links}>
            {links.map(({ label, href }) => (
              <a key={label} href={href} target="_blank" rel="noreferrer">
                {label}
              </a>
            ))}
          </div>
        ) : null}
      </section>

      <section className={styles.surfaceCard}>
        <div className={styles.cardHeader}>
          <div>
            <span className={styles.eyebrow}>Inventory</span>
            <h2>Active composition</h2>
          </div>
        </div>
        <div className={styles.metricsGrid}>
          <div>
            <strong>{recipe.counts.unified_models}</strong>
            <span>Unified models</span>
          </div>
          <div>
            <strong>{recipe.counts.recipes}</strong>
            <span>Routing recipes</span>
          </div>
          <div>
            <strong>{recipe.counts.decisions}</strong>
            <span>Decisions</span>
          </div>
          <div>
            <strong>{recipe.counts.probes}</strong>
            <span>Offline probes</span>
          </div>
        </div>
      </section>

      <section className={`${styles.surfaceCard} ${styles.fullWidthCard}`}>
        <div className={styles.cardHeader}>
          <div>
            <span className={styles.eyebrow}>Source health</span>
            <h2>Five-file Recipe contract</h2>
          </div>
          <span
            className={`${styles.statusBadge} ${recipe.source_health.status === 'ready' ? styles.readyBadge : styles.warningBadge}`}
          >
            {recipe.source_health.status}
          </span>
        </div>
        <div className={styles.sourceGrid}>
          {Object.entries(SOURCE_LABELS).map(([key, label]) => {
            const health = recipe.source_health.files[key as keyof typeof SOURCE_LABELS] ?? {
              present: false,
            }
            const digest = recipe.digests[key as keyof typeof recipe.digests] || health.digest
            return (
              <div key={key} className={styles.sourceFile}>
                <span
                  className={health.present ? styles.presentDot : styles.missingDot}
                  aria-hidden="true"
                />
                <div>
                  <strong>{label}</strong>
                  <code title={digest}>{shortDigest(digest)}</code>
                </div>
              </div>
            )
          })}
        </div>
        {recipe.digests.recipe ? (
          <div className={styles.recipeDigest}>
            <span>Recipe digest</span>
            <code title={recipe.digests.recipe}>{recipe.digests.recipe}</code>
          </div>
        ) : null}
        {recipe.source_health.issues.length ? (
          <ul className={styles.issueList}>
            {recipe.source_health.issues.map((issue, index) => (
              <li key={`${index}-${issue}`}>{issue}</li>
            ))}
          </ul>
        ) : null}
      </section>

      <section className={`${styles.surfaceCard} ${styles.fullWidthCard}`}>
        <div className={styles.cardHeader}>
          <div>
            <span className={styles.eyebrow}>Documentation</span>
            <h2>README</h2>
          </div>
        </div>
        {recipe.readme ? (
          <div className={styles.readme}>
            <MarkdownRenderer content={recipe.readme} allowImages={false} />
          </div>
        ) : (
          <p className={styles.emptyCopy}>README.md has no content.</p>
        )}
      </section>
    </div>
  )
}
