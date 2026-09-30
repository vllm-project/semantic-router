import React, { useMemo, useState } from 'react'

import type { BuiltInModelCatalog, CatalogProtocol, CatalogProvider } from '../types/modelCatalog'
import {
  modelHubContextLabel,
  modelHubMaxOutputLabel,
  modelHubParameterLabel,
  modelHubPublicEvaluations,
  modelHubDistributionLabel as distributionLabel,
  modelHubRelationshipLabel as relationshipLabel,
  readableModelHubValue as readable,
  type ModelHubRow,
} from './modelHubSupport'
import { ModelMark, ProviderMark } from './ModelHubComponents'
import {
  ModelHubDetailTabs,
  type ModelHubDetailTab,
  type ModelHubDetailTabItem,
} from './ModelHubDetailTabs'
import { EvaluationCard } from './ModelHubEvaluationCard'
import styles from './ModelHubDetail.module.css'
import { providerProtocolOperations } from './modelHubProviderOperations'

export { EvaluationCard } from './ModelHubEvaluationCard'

const Overview: React.FC<{ row: ModelHubRow; catalog: BuiltInModelCatalog }> = ({
  row,
  catalog,
}) => {
  const family = catalog.reasoning_families.find(
    (candidate) => candidate.id === row.model.reasoning_family,
  )
  return (
    <div className={styles.detailStack}>
      <section className={styles.detailSection}>
        <div className={styles.detailSectionTitle}>
          <h3>Capabilities</h3>
          <span>{row.model.capabilities.length}</span>
        </div>
        <div className={styles.badges}>
          {row.model.capabilities.map((capability) => (
            <span key={capability}>{readable(capability)}</span>
          ))}
        </div>
      </section>
      <section className={styles.detailSection}>
        <h3>Runtime envelope</h3>
        <dl className={styles.factGrid}>
          <div>
            <dt>Context</dt>
            <dd>{modelHubContextLabel(row.model)}</dd>
          </div>
          <div>
            <dt>Max output</dt>
            <dd>{modelHubMaxOutputLabel(row.model)}</dd>
          </div>
          <div>
            <dt>Parameters</dt>
            <dd>{modelHubParameterLabel(row.model)}</dd>
          </div>
          <div>
            <dt>Released</dt>
            <dd>{row.model.released_at ?? 'Not published'}</dd>
          </div>
        </dl>
      </section>
      <section className={styles.detailSection}>
        <h3>Modalities</h3>
        <div className={styles.modalityRows}>
          <span>
            <small>Input</small>
            {row.model.modalities.input.map(readable).join(', ')}
          </span>
          <span>
            <small>Output</small>
            {row.model.modalities.output.map(readable).join(', ')}
          </span>
        </div>
      </section>
      {family ? (
        <section className={styles.detailSection}>
          <div className={styles.detailSectionTitle}>
            <h3>Reasoning profile</h3>
            <span>{family.id}</span>
          </div>
          <p>
            Default:{' '}
            <strong>{family.default ? readable(family.default) : 'model selected'}</strong>
            {family.default_mode ? ` · mode: ${readable(family.default_mode)}` : ''}
            {family.disabled ? ` · disabled: ${readable(family.disabled)}` : ''}
            {family.activation_parameter
              ? ` · activation: ${readable(family.activation_parameter)}`
              : ''}
          </p>
          <div className={styles.reasoningTrack}>
            {(family.levels ?? []).map((level) => (
              <span className={level === family.default ? styles.reasoningDefault : ''} key={level}>
                {readable(level)}
              </span>
            ))}
          </div>
          {family.modes?.length ? (
            <div className={styles.reasoningTrack}>
              {family.modes.map((mode) => (
                <span
                  className={mode === family.default_mode ? styles.reasoningDefault : ''}
                  key={mode}
                >
                  {readable(mode)}
                </span>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}

const Evaluations: React.FC<{ row: ModelHubRow; catalog: BuiltInModelCatalog }> = ({
  row,
  catalog,
}) => {
  const records = modelHubPublicEvaluations(catalog).filter(
    (evaluation) => evaluation.model === row.model.id,
  )
  return (
    <section className={styles.detailStack}>
      <div className={styles.evaluationIntro}>
        <strong>{row.benchmarkCount}</strong>
        <span>benchmarks</span>
        <strong>{row.evaluationCount}</strong>
        <span>evaluation records</span>
      </div>
      {records.length ? (
        records.map((evaluation) => (
          <EvaluationCard
            key={evaluation.id}
            model={row.model}
            evaluation={evaluation}
            benchmark={catalog.benchmarks.find(
              (benchmark) => benchmark.id === evaluation.benchmark,
            )}
          />
        ))
      ) : (
        <div className={styles.detailEmpty}>
          <strong>No published evaluations yet</strong>
          <span>
            The catalog keeps missing evidence explicit rather than substituting a synthetic score.
          </span>
        </div>
      )}
    </section>
  )
}

const ProtocolOperations: React.FC<{
  provider: CatalogProvider
  protocol: CatalogProtocol
}> = ({ provider, protocol }) => {
  const operations = providerProtocolOperations(provider, protocol)
  if (!operations.length) return null

  return (
    <div className={styles.protocolCard}>
      <div>
        <strong>{protocol.display_name}</strong>
        <code>{protocol.id}</code>
      </div>
      <div className={styles.operationList}>
        {operations.map((operation) => (
          <div className={styles.operationRow} key={operation.reference}>
            <i>{operation.method}</i>
            <strong>{readable(operation.id)}</strong>
            <code>{operation.path}</code>
          </div>
        ))}
      </div>
    </div>
  )
}

export const ModelAccess: React.FC<{ row: ModelHubRow; catalog: BuiltInModelCatalog }> = ({
  row,
  catalog,
}) => (
  <div className={styles.detailStack}>
    {row.providers.length ? (
      row.providers.map(({ provider, model }) => {
        const protocols = model.protocols.flatMap((id) => {
          const protocol = catalog.protocols.find((candidate) => candidate.id === id)
          return protocol ? [protocol] : []
        })
        return (
          <section className={styles.providerCard} key={`${provider.id}/${model.id}`}>
            <header>
              <div className={styles.providerIdentity}>
                <ProviderMark provider={provider} />
                <span>
                  <strong>{provider.display_name}</strong>
                  <small>
                    {provider.support_tier} support · {readable(model.lifecycle)}
                  </small>
                </span>
              </div>
              <span>{relationshipLabel[model.relationship]}</span>
            </header>
            <div className={styles.providerMeta}>
              <code>{model.id}</code>
              {provider.default_base_url ? <span>{provider.default_base_url}</span> : null}
            </div>
            <div className={styles.protocolList}>
              {protocols.map((protocol) => (
                <ProtocolOperations key={protocol.id} provider={provider} protocol={protocol} />
              ))}
            </div>
            {model.pricing && Object.keys(model.pricing).length ? (
              <dl className={styles.pricingGrid}>
                {Object.entries(model.pricing).map(([key, value]) => (
                  <div key={key}>
                    <dt>{readable(key)}</dt>
                    <dd>{String(value)}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </section>
        )
      })
    ) : (
      <div className={styles.detailEmpty}>
        <strong>No direct provider binding</strong>
        <span>Virtual models are materialized through their router recipe and backend roles.</span>
      </div>
    )}
  </div>
)

export const VirtualPool: React.FC<{ row: ModelHubRow; catalog: BuiltInModelCatalog }> = ({
  row,
  catalog,
}) => {
  const modelMap = useMemo(
    () => new Map(catalog.models.map((model) => [model.id, model])),
    [catalog.models],
  )
  if (row.model.kind !== 'virtual') {
    return (
      <div className={styles.detailEmpty}>
        <strong>Single model</strong>
        <span>Backend pools apply only to virtual model recipes.</span>
      </div>
    )
  }
  return (
    <div className={styles.detailStack}>
      <section className={styles.recipeSummary}>
        <span>
          <small>Entrypoint</small>
          <code>{row.model.entrypoint}</code>
        </span>
        <span>
          <small>Recipe</small>
          <strong>{row.model.recipe ?? 'Built-in'}</strong>
        </span>
        <span>
          <small>Roles</small>
          <strong>{row.model.roles?.length ?? 0}</strong>
        </span>
      </section>
      <div className={styles.poolDiagram}>
        <div className={styles.poolEntrypoint}>
          <ModelMark model={row.model} />
          <span>
            <strong>{row.model.display_name}</strong>
            <small>request entrypoint</small>
          </span>
        </div>
        <div className={styles.poolSpine} aria-hidden="true" />
        {row.model.roles?.map((role) => (
          <section className={styles.poolRole} key={role.name}>
            <header>
              <div>
                <strong>{readable(role.name)}</strong>
                <small>{role.required ? 'Required role' : 'Optional role'}</small>
              </div>
              <span>min {role.minimum_candidates}</span>
            </header>
            <div className={styles.poolTraits}>
              {role.traits.map((trait) => (
                <span key={trait}>{readable(trait)}</span>
              ))}
            </div>
            <div className={styles.poolCandidates}>
              {role.recommended_pool.map((id) => {
                const model = modelMap.get(id)
                return (
                  <div key={id}>
                    {model ? (
                      <ModelMark model={model} />
                    ) : (
                      <span className={styles.customModelMark}>C</span>
                    )}
                    <span>
                      <strong>{model?.display_name ?? id}</strong>
                      <small>{model ? model.publisher : 'Custom model slot'}</small>
                    </span>
                  </div>
                )
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}

export const ModelDetail: React.FC<{
  row: ModelHubRow | null
  catalog: BuiltInModelCatalog
  modal?: boolean
  closeButtonRef?: React.RefObject<HTMLButtonElement>
  onClose?: () => void
}> = ({ row, catalog, modal = false, closeButtonRef, onClose }) => {
  const [tab, setTab] = useState<ModelHubDetailTab>('overview')
  if (!row)
    return (
      <aside className={styles.detailPanel} data-model-hub-detail>
        <div className={styles.detailEmpty}>Select a model to inspect it.</div>
      </aside>
    )
  const tabs: Array<ModelHubDetailTabItem & { hidden?: boolean }> = [
    { id: 'overview', label: 'Overview' },
    { id: 'evaluations', label: `Evaluations ${row.evaluationCount}` },
    { id: 'access', label: `Access ${row.providers.length}` },
    { id: 'pool', label: 'Backend pool', hidden: row.model.kind !== 'virtual' },
  ]
  const visibleTabs = tabs.filter((item) => !item.hidden)
  const detailID = `model-detail-${row.model.id.replace(/[^a-z0-9_-]+/gi, '-')}`
  return (
    <aside
      className={styles.detailPanel}
      data-model-hub-detail
      aria-live="polite"
      role={modal ? 'dialog' : undefined}
      aria-modal={modal || undefined}
      aria-labelledby={modal ? `${detailID}-title` : undefined}
      tabIndex={modal ? -1 : undefined}
    >
      {onClose ? (
        <button
          className={styles.mobileDetailClose}
          type="button"
          aria-label="Close model details"
          onClick={onClose}
          ref={closeButtonRef}
        >
          ×
        </button>
      ) : null}
      <header className={styles.detailHeading}>
        <ModelMark model={row.model} large />
        <div>
          <span>{row.model.publisher}</span>
          <h2 id={`${detailID}-title`}>{row.model.display_name}</h2>
          <code>{row.model.id}</code>
        </div>
      </header>
      <p className={styles.description}>{row.model.description}</p>
      <div className={styles.badges}>
        <span>{distributionLabel[row.model.distribution.type]}</span>
        <span>{row.model.lifecycle}</span>
        {row.model.parameter_size ? <span>{row.model.parameter_size}</span> : null}
      </div>
      <ModelHubDetailTabs detailID={detailID} tabs={visibleTabs} selected={tab} select={setTab} />
      <div
        id={`${detailID}-panel`}
        className={styles.detailBody}
        role="tabpanel"
        aria-labelledby={`${detailID}-tab-${tab}`}
        tabIndex={0}
      >
        {tab === 'overview' ? <Overview row={row} catalog={catalog} /> : null}
        {tab === 'evaluations' ? <Evaluations row={row} catalog={catalog} /> : null}
        {tab === 'access' ? <ModelAccess row={row} catalog={catalog} /> : null}
        {tab === 'pool' ? <VirtualPool row={row} catalog={catalog} /> : null}
      </div>
      <a
        className={styles.sourceLink}
        href={row.model.distribution.source}
        target="_blank"
        rel="noreferrer"
      >
        Open official model source ↗
      </a>
    </aside>
  )
}
