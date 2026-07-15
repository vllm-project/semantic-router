import type { EmbeddingProviderRuntimeStatus } from '../utils/routerRuntime'
import {
  embeddingProviderHealthLabel,
  embeddingProviderTone,
  formatEmbeddingProviderBackend,
  formatProviderCheckedAt,
} from './statusPageSupport'
import styles from './EmbeddingProviderStatusPanel.module.css'

interface EmbeddingProviderStatusPanelProps {
  provider: EmbeddingProviderRuntimeStatus
}

function valueOrFallback(value: string | number | undefined): string {
  if (typeof value === 'number') return String(value)
  return value?.trim() || 'Not reported'
}

function apiKeyStatus(provider: EmbeddingProviderRuntimeStatus): string {
  if (!provider.api_key_env) return 'Not required'
  if (provider.api_key_env_set === true) return 'Available'
  if (provider.api_key_env_set === false) return 'Missing'
  return 'Not checked'
}

export default function EmbeddingProviderStatusPanel({
  provider,
}: EmbeddingProviderStatusPanelProps) {
  const tone = embeddingProviderTone(provider)
  const healthLabel = embeddingProviderHealthLabel(provider)

  return (
    <section
      className={styles.panel}
      data-testid="embedding-provider-status"
      aria-labelledby="embedding-provider-status-title"
    >
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Embedding infrastructure</span>
          <h2 id="embedding-provider-status-title" className={styles.title}>
            Remote embedding provider
          </h2>
          <p className={styles.description}>
            Runtime probe state for the provider shared by text embedding consumers.
          </p>
        </div>
        <span className={`${styles.health} ${styles[tone]}`} role="status" aria-live="polite">
          <span className={styles.healthDot} aria-hidden="true" />
          {healthLabel}
        </span>
      </div>

      <dl className={styles.facts}>
        <div>
          <dt>Backend</dt>
          <dd>{formatEmbeddingProviderBackend(provider.backend)}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd title={provider.model}>{valueOrFallback(provider.model)}</dd>
        </div>
        <div>
          <dt>Dimension</dt>
          <dd>{valueOrFallback(provider.dimension)}</dd>
        </div>
        <div>
          <dt>Credential env</dt>
          <dd title={provider.api_key_env}>{valueOrFallback(provider.api_key_env)}</dd>
        </div>
        <div>
          <dt>Credential</dt>
          <dd>{apiKeyStatus(provider)}</dd>
        </div>
        <div>
          <dt>Last probe</dt>
          <dd>{formatProviderCheckedAt(provider.last_checked_at)}</dd>
        </div>
      </dl>

      {provider.last_probe_error ? (
        <div className={styles.error} role="alert">
          <strong>Last probe failed</strong>
          <span>{provider.last_probe_error}</span>
        </div>
      ) : null}
    </section>
  )
}
