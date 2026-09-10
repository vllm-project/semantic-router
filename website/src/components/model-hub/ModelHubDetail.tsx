import React, { useEffect, useRef } from 'react'

import type {
  CatalogBenchmark,
  CatalogEvaluation,
  CatalogModel,
  CatalogModelBinding,
  CatalogProvider,
  CatalogReasoningFamily,
} from '../../data/modelHubCatalogTypes'
import { modelHubBenchmarkRawValueLabel } from '../../data/modelHubBenchmarkSupport'
import {
  modelHubEvaluationConditionLabel,
  modelHubEvaluationDateLabel,
} from '../../data/modelHubEvaluationLabel'
import { CatalogMark } from './ModelHubMark'
import {
  Badge,
  formatMetric,
  modelContextLabel,
  modelMaxOutputLabel,
  modelParameterLabel,
  readable,
  Tags,
} from './ModelHubPrimitives'
import styles from './modelHubDetail.module.css'

const relationshipLabel: Record<CatalogModelBinding['relationship'], string> = {
  first_party: 'First-party',
  managed_cloud: 'Managed cloud',
  gateway: 'Gateway',
  self_hosted: 'Self-hosted',
}

function DetailSection({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <section className={styles.detailSection}>
      <h3>{title}</h3>
      {children}
    </section>
  )
}

export function ModelHubDetail({
  model,
  providers,
  evaluations,
  benchmarks,
  family,
  modelByID,
  onClose,
}: {
  model: CatalogModel
  providers: Array<{ provider: CatalogProvider, binding: CatalogModelBinding }>
  evaluations: CatalogEvaluation[]
  benchmarks: CatalogBenchmark[]
  family?: CatalogReasoningFamily
  modelByID: Map<string, CatalogModel>
  onClose: () => void
}) {
  const dialogRef = useRef<HTMLElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const benchmarkByID = new Map(benchmarks.map(benchmark => [benchmark.id, benchmark]))

  useEffect(() => {
    const dialog = dialogRef.current
    const previousFocus = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    closeRef.current?.focus()
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key !== 'Tab' || !dialog) return
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      }
      else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', handleKeyDown)
      previousFocus?.focus()
    }
  }, [onClose])

  return (
    <div className={styles.backdrop} role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <aside ref={dialogRef} className={styles.detail} role="dialog" aria-modal="true" aria-labelledby="model-detail-title" aria-describedby="model-detail-description" tabIndex={-1}>
        <header className={styles.detailHero}>
          <CatalogMark presentation={model.presentation} large />
          <span>
            <span className={styles.eyebrow}>{model.publisher}</span>
            <h2 id="model-detail-title">{model.display_name}</h2>
            <code>{model.id}</code>
          </span>
          <button ref={closeRef} type="button" className={styles.closeButton} onClick={onClose} aria-label="Close model details">×</button>
        </header>
        <p id="model-detail-description" className={styles.description}>{model.description}</p>
        <div className={styles.badges}>
          <Badge value={model.kind} />
          <Badge value={model.distribution.type} />
          <Badge value={model.lifecycle} />
          {model.parameter_size ? <Badge value="size" label={model.parameter_size} /> : null}
        </div>
        <dl className={styles.facts}>
          <div>
            <dt>Context</dt>
            <dd>{modelContextLabel(model)}</dd>
          </div>
          <div>
            <dt>Max output</dt>
            <dd>{modelMaxOutputLabel(model)}</dd>
          </div>
          <div>
            <dt>Parameters</dt>
            <dd>{modelParameterLabel(model)}</dd>
          </div>
          <div>
            <dt>Released</dt>
            <dd>{model.released_at ?? '—'}</dd>
          </div>
          <div>
            <dt>Verified</dt>
            <dd>{model.verification.verified_at}</dd>
          </div>
        </dl>
        <DetailSection title="Capabilities">
          <Tags values={model.capabilities} />
          <p>
            Input:
            {model.modalities.input.map(readable).join(', ')}
            {' '}
            · Output:
            {model.modalities.output.map(readable).join(', ')}
          </p>
        </DetailSection>
        {family
          ? (
              <DetailSection title="Reasoning">
                <p>
                  <code>{family.id}</code>
                  {' '}
                  · default
                  {' '}
                  {family.default ? readable(family.default) : 'model selected'}
                  {family.default_mode ? ` · mode ${readable(family.default_mode)}` : ''}
                </p>
                <Tags values={[...(family.levels ?? []), ...(family.modes ?? [])]} />
              </DetailSection>
            )
          : null}
        {model.kind === 'virtual'
          ? (
              <DetailSection title="Backend pool">
                <div className={styles.recipeSummary}>
                  <span>
                    <small>Entrypoint</small>
                    <code>{model.entrypoint ?? model.id}</code>
                  </span>
                  <span>
                    <small>Recipe</small>
                    <strong>{model.recipe ?? 'Built-in'}</strong>
                  </span>
                  <span>
                    <small>Roles</small>
                    <strong>{model.roles?.length ?? 0}</strong>
                  </span>
                </div>
                <div className={styles.roleGrid}>
                  {(model.roles ?? []).map(role => (
                    <article key={role.name} className={styles.roleCard}>
                      <header>
                        <strong>{readable(role.name)}</strong>
                        <Badge value={role.required ? 'required' : 'optional'} />
                      </header>
                      <small>
                        Minimum
                        {' '}
                        {role.minimum_candidates}
                      </small>
                      <ul>
                        {role.recommended_pool.map((candidate) => {
                          const candidateModel = modelByID.get(candidate)
                          return (
                            <li key={candidate}>
                              {candidateModel ? <CatalogMark presentation={candidateModel.presentation} /> : <span className={styles.customMark}>C</span>}
                              <span>
                                <strong>{candidateModel?.display_name ?? candidate}</strong>
                                <small>{candidateModel?.publisher ?? 'Custom model slot'}</small>
                              </span>
                            </li>
                          )
                        })}
                      </ul>
                      <Tags values={role.traits} />
                    </article>
                  ))}
                </div>
              </DetailSection>
            )
          : (
              <DetailSection title="Providers">
                {providers.length
                  ? (
                      <div className={styles.providerList}>
                        {providers.map(({ provider, binding }) => (
                          <article key={`${provider.id}:${binding.id}`}>
                            <span>
                              <CatalogMark presentation={provider.presentation} />
                              <span>
                                <strong>{provider.display_name}</strong>
                                <code>{binding.id}</code>
                              </span>
                            </span>
                            <span>
                              <Badge value={binding.relationship} label={relationshipLabel[binding.relationship]} />
                              <Tags values={binding.protocols} />
                            </span>
                          </article>
                        ))}
                      </div>
                    )
                  : <p>No built-in provider choice. Connect a compatible provider and enter its model ID.</p>}
              </DetailSection>
            )}
        <DetailSection title={`Evidence · ${evaluations.length}`}>
          {evaluations.length
            ? (
                <div className={styles.evaluationList}>
                  {evaluations.map((evaluation) => {
                    const benchmark = benchmarkByID.get(evaluation.benchmark)
                    const date = modelHubEvaluationDateLabel(evaluation)
                    return (
                      <article key={evaluation.id}>
                        <header>
                          <span>
                            <strong>{benchmark?.display_name ?? evaluation.benchmark}</strong>
                            <small>
                              {modelHubEvaluationConditionLabel(model, evaluation.reasoning_effort)}
                              {' '}
                              ·
                              {' '}
                              {readable(evaluation.benchmark_profile)}
                            </small>
                          </span>
                          <span>
                            {Object.entries(evaluation.metrics ?? {}).map(([id, value]) => {
                              const definition = benchmark?.metrics.find(item => item.id === id)
                              const raw = definition && typeof value === 'number'
                                ? modelHubBenchmarkRawValueLabel(value, definition)
                                : undefined
                              return (
                                <span key={id}>
                                  <small>{readable(id)}</small>
                                  <b title={raw ? `Raw: ${raw}` : undefined}>
                                    {definition && typeof value === 'number' ? formatMetric(value, definition) : value}
                                  </b>
                                </span>
                              )
                            })}
                          </span>
                        </header>
                        {date || evaluation.evidence.source
                          ? (
                              <footer>
                                {date ? <small>{date}</small> : null}
                                {evaluation.evidence.source ? <a href={evaluation.evidence.source} target="_blank" rel="noreferrer">Source ↗</a> : null}
                              </footer>
                            )
                          : null}
                      </article>
                    )
                  })}
                </div>
              )
            : <p>No published measurements.</p>}
        </DetailSection>
        <a className={`site-btn site-btn--primary ${styles.sourceLink}`} href={model.distribution.source} target="_blank" rel="noreferrer">Official model ↗</a>
      </aside>
    </div>
  )
}
