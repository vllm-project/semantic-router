import type { BuiltInModelCatalog } from '../types/modelCatalog'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import {
  catalogSnapshotsForEntrypoint,
  preferredCatalogModelForEntrypoint,
} from './configPageModelCatalogSupport'
import { collectRecipeTargetModels, getRecipeByName } from './configPageEntrypointsRecipesSupport'
import type { ConfigData, EntrypointConfig } from './configPageSupport'

interface ConfigPageMoMEntrypointCatalogMetadataProps {
  catalog: BuiltInModelCatalog | null
  catalogLoading: boolean
  catalogError: string | null
  config: ConfigData
  entrypoint: EntrypointConfig
  onCatalogRetry: () => void
}

export default function ConfigPageMoMEntrypointCatalogMetadata({
  catalog,
  catalogLoading,
  catalogError,
  config,
  entrypoint,
  onCatalogRetry,
}: ConfigPageMoMEntrypointCatalogMetadataProps) {
  if (catalogLoading) {
    return <div className={pageStyles.catalogDisclosure}>Loading built-in model metadata…</div>
  }
  if (catalogError) {
    return (
      <div className={pageStyles.catalogDisclosure} role="alert">
        <strong>Catalog metadata is unavailable.</strong>
        <span>{catalogError}</span>
        <button type="button" onClick={onCatalogRetry}>
          Retry catalog
        </button>
      </div>
    )
  }

  const model = preferredCatalogModelForEntrypoint(catalog, entrypoint)
  if (!model) {
    const recipe = getRecipeByName(config, entrypoint.recipe)
    return (
      <div className={pageStyles.catalogDisclosure}>
        <div className={pageStyles.catalogDisclosureHeader}>
          <div>
            <span className={pageStyles.metricLabel}>Custom unified model</span>
            <strong>{entrypoint.model_names.join(', ')}</strong>
          </div>
          <span className={pageStyles.customBadge}>custom / unverified</span>
        </div>
        <p>
          This entrypoint is not an exact installed catalog model. It remains editable, but the
          Dashboard does not present custom routing as a verified built-in asset.
        </p>
        <dl className={pageStyles.catalogFacts}>
          <div>
            <dt>Routing recipe</dt>
            <dd>{entrypoint.recipe}</dd>
          </div>
          <div>
            <dt>Configured targets</dt>
            <dd>{collectRecipeTargetModels(recipe).length}</dd>
          </div>
        </dl>
      </div>
    )
  }

  const snapshots = catalogSnapshotsForEntrypoint(catalog, entrypoint)
  return (
    <div className={pageStyles.catalogDisclosure}>
      <div className={pageStyles.catalogDisclosureHeader}>
        <div>
          <span className={pageStyles.metricLabel}>Mixture-of-Models metadata</span>
          <strong>{model.display_name}</strong>
          <code>{model.id}</code>
        </div>
        <div className={pageStyles.catalogBadges}>
          <span>{model.catalog_version}</span>
          <span>{model.channel}</span>
          <span className={model.compatible ? pageStyles.verifiedBadge : pageStyles.warningBadge}>
            {model.compatible ? `catalog ${model.verification.status}` : 'catalog incompatible'}
          </span>
        </div>
      </div>
      <p>{model.description}</p>
      <p>
        This is the installed catalog reference for the matching public model ID. The currently
        deployed YAML is independently custom / unverified unless exact catalog provenance is
        established by the runtime.
      </p>
      <dl className={pageStyles.catalogFacts}>
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
          <dt>Installed snapshots</dt>
          <dd>
            {snapshots
              .map((snapshot) => `${snapshot.catalog_version} ${snapshot.channel}`)
              .join(', ')}
          </dd>
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
      <div className={pageStyles.catalogTraits} aria-label={`${model.display_name} traits`}>
        {model.traits.map((trait) => (
          <span key={trait}>{trait}</span>
        ))}
      </div>
      <div className={pageStyles.catalogRoleGrid}>
        {model.roles.map((role) => (
          <section key={role.name}>
            <div>
              <strong>{role.name}</strong>
              <span>
                {role.required ? 'required' : 'optional'} · min {role.minimum_candidates}
              </span>
            </div>
            <p>{role.traits.join(' · ')}</p>
            <div className={pageStyles.catalogPool}>
              {role.recommended_pool.map((candidate) => (
                <code key={candidate}>{candidate}</code>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
