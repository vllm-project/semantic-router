import type { BuiltInModelCatalog, BuiltInModelMetadata } from '../types/modelCatalog'
import styles from './ConfigPageMoMCatalogPanel.module.css'
import { modelCatalogVersionKey, modelsForCatalogVersion } from './configPageModelCatalogSupport'

interface ConfigPageMoMCatalogPanelProps {
  catalog: BuiltInModelCatalog | null
  loading: boolean
  error: string | null
  onRetry: () => void
}

function ModelRequirements({ model }: { model: BuiltInModelMetadata }) {
  return (
    <article className={styles.modelCard} data-compatible={model.compatible}>
      <div className={styles.modelHeader}>
        <div>
          <span className={styles.modelName}>{model.display_name}</span>
          <code>{model.id}</code>
        </div>
        <div className={styles.badges}>
          {model.default ? <span className={styles.defaultBadge}>default</span> : null}
          {model.enabled_by_default ? <span className={styles.enabledBadge}>enabled</span> : null}
          <span className={model.compatible ? styles.verifiedBadge : styles.warningBadge}>
            {model.compatible ? `catalog ${model.verification.status}` : 'catalog incompatible'}
          </span>
        </div>
      </div>
      <p>{model.description}</p>
      <dl className={styles.modelFacts}>
        <div>
          <dt>Policy</dt>
          <dd>
            {model.family} v{model.generation} · {model.policy_version}
          </dd>
        </div>
        <div>
          <dt>Routing recipe</dt>
          <dd>{model.recipe}</dd>
        </div>
        <div>
          <dt>Protocols</dt>
          <dd>{model.protocols.join(', ')}</dd>
        </div>
        <div>
          <dt>Compatibility</dt>
          <dd>{model.compatibility_reason}</dd>
        </div>
        <div>
          <dt>Verified by</dt>
          <dd>{model.verification.authority}</dd>
        </div>
        <div>
          <dt>Asset digest</dt>
          <dd>
            <code title={model.verification.asset_sha256}>
              {model.verification.asset_sha256.slice(0, 19)}…
            </code>
          </dd>
        </div>
      </dl>
      <div className={styles.traits} aria-label={`${model.display_name} traits`}>
        {model.traits.map((trait) => (
          <span key={trait}>{trait}</span>
        ))}
      </div>
      <details className={styles.roles}>
        <summary>{model.roles.length} backend role requirements</summary>
        <div className={styles.roleGrid}>
          {model.roles.map((role) => (
            <section key={role.name} className={styles.roleCard}>
              <div>
                <strong>{role.name}</strong>
                <span>
                  {role.required ? 'required' : 'optional'} · minimum {role.minimum_candidates}
                </span>
              </div>
              <p>{role.traits.join(' · ')}</p>
              <span className={styles.poolLabel}>Recommended pool</span>
              <div className={styles.recommendedPool}>
                {role.recommended_pool.map((name) => (
                  <code key={name}>{name}</code>
                ))}
              </div>
            </section>
          ))}
        </div>
      </details>
    </article>
  )
}

export default function ConfigPageMoMCatalogPanel({
  catalog,
  loading,
  error,
  onRetry,
}: ConfigPageMoMCatalogPanelProps) {
  if (loading) {
    return (
      <section className={styles.panel} aria-busy="true">
        <div className={styles.asyncState}>Loading model catalog versions…</div>
      </section>
    )
  }

  if (error || !catalog) {
    return (
      <section className={styles.panel}>
        <div className={styles.asyncState} role="alert">
          <strong>Built-in model catalog is unavailable</strong>
          <span>{error || 'The Dashboard did not return an installed model catalog.'}</span>
          <button type="button" onClick={onRetry}>
            Retry catalog
          </button>
        </div>
      </section>
    )
  }

  return (
    <section className={styles.panel} aria-labelledby="built-in-model-catalog-title">
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Installed with vllm-sr</span>
          <h2 id="built-in-model-catalog-title">Built-in model catalogs</h2>
          <p>
            Browse the version channels and virtual models embedded in this vllm-sr build. The
            moving latest channel follows the build; immutable release channels appear only after
            their release snapshot is published.
          </p>
        </div>
        <span className={styles.total}>{catalog.catalogs.length} version packages</span>
      </div>

      <div className={styles.versionList}>
        {catalog.catalogs.map((version) => {
          const models = modelsForCatalogVersion(catalog, version)
          return (
            <article key={modelCatalogVersionKey(version)} className={styles.versionCard}>
              <div className={styles.versionHeader}>
                <div>
                  <span className={styles.channel}>{version.channel}</span>
                  <h3>{version.catalog_version}</h3>
                </div>
                <dl className={styles.versionFacts}>
                  <div>
                    <dt>Default</dt>
                    <dd>
                      <code>{version.default_model}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Enabled by default</dt>
                    <dd>{version.enabled_models.length}</dd>
                  </div>
                  <div>
                    <dt>Supported models</dt>
                    <dd>{models.length}</dd>
                  </div>
                </dl>
              </div>
              <div className={styles.modelGrid}>
                {models.map((model) => (
                  <ModelRequirements key={`${version.channel}:${model.id}`} model={model} />
                ))}
              </div>
            </article>
          )
        })}
      </div>

      <aside className={styles.forkHint}>
        <strong>Start from a built-in, keep control of the result.</strong>
        <span>
          Use <code>vllm-sr model fork</code> to materialize one or more compatible models into
          editable YAML. Custom Dashboard edits remain custom and are never presented as verified
          catalog assets.
        </span>
      </aside>
    </section>
  )
}
