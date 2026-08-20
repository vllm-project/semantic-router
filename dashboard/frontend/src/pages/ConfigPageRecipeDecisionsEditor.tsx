import { useState } from 'react'
import type { DecisionCondition, DecisionConfig } from './configPageSupport'
import styles from './ConfigPageEntrypointsRecipesSection.module.css'

interface ConfigPageRecipeDecisionsEditorProps {
  value: DecisionConfig[]
  catalog?: DecisionConfig[]
  onChange: (value: DecisionConfig[]) => void
}

const emptyDecision = (): DecisionConfig => ({
  name: '',
  description: '',
  priority: 100,
  rules: { operator: 'AND', conditions: [] },
  modelRefs: [],
})

const emptyCondition = (): DecisionCondition => ({
  type: 'metadata',
  name: '',
})

const conditionTypes = [
  'metadata',
  'classifier',
  'keyword',
  'embedding',
  'domain',
  'fact_check',
  'user_feedback',
  'reask',
  'preference',
  'language',
  'context',
  'structure',
  'complexity',
  'modality',
  'authz',
  'jailbreak',
  'pii',
  'kb',
  'conversation',
  'event',
  'projection',
]

export default function ConfigPageRecipeDecisionsEditor({
  value,
  catalog = [],
  onChange,
}: ConfigPageRecipeDecisionsEditorProps) {
  const rows = Array.isArray(value) ? value : []
  const [reuseName, setReuseName] = useState('')

  const updateDecision = (index: number, patch: Partial<DecisionConfig>) => {
    onChange(
      rows.map((decision, rowIndex) => (rowIndex === index ? { ...decision, ...patch } : decision)),
    )
  }

  const updateCondition = (
    decisionIndex: number,
    conditionIndex: number,
    patch: Partial<DecisionCondition>,
  ) => {
    const decision = rows[decisionIndex]
    const conditions = (decision.rules?.conditions ?? []).map((condition, index) =>
      index === conditionIndex ? { ...condition, ...patch } : condition,
    )
    updateDecision(decisionIndex, {
      rules: {
        operator: decision.rules?.operator ?? 'AND',
        conditions,
      },
    })
  }

  const removeCondition = (decisionIndex: number, conditionIndex: number) => {
    const decision = rows[decisionIndex]
    updateDecision(decisionIndex, {
      rules: {
        operator: decision.rules?.operator ?? 'AND',
        conditions: (decision.rules?.conditions ?? []).filter(
          (_, index) => index !== conditionIndex,
        ),
      },
    })
  }

  return (
    <div className={styles.decisionEditor}>
      <p className={styles.editorHint}>
        Reuse or compose a decision. Models are assigned when you create a model.
      </p>
      {catalog.length ? (
        <div className={styles.modelPoolHeader}>
          <label>
            <span>Reuse decision</span>
            <select value={reuseName} onChange={(event) => setReuseName(event.target.value)}>
              <option value="">Choose from your library</option>
              {catalog.map((decision) => (
                <option key={decision.name} value={decision.name}>
                  {decision.name}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className={styles.secondaryButton}
            disabled={!reuseName}
            onClick={() => {
              const source = catalog.find((decision) => decision.name === reuseName)
              if (source) onChange([...rows, JSON.parse(JSON.stringify(source)) as DecisionConfig])
              setReuseName('')
            }}
          >
            Add to recipe
          </button>
        </div>
      ) : null}
      {rows.map((decision, decisionIndex) => (
        <article key={decisionIndex} className={styles.decisionCard}>
          <div className={styles.decisionCardHeader}>
            <strong>Decision {decisionIndex + 1}</strong>
            <button
              type="button"
              className={styles.dangerButton}
              onClick={() => onChange(rows.filter((_, index) => index !== decisionIndex))}
            >
              Remove decision
            </button>
          </div>

          <div className={styles.editorGrid}>
            <label>
              <span>Decision name</span>
              <input
                value={decision.name ?? ''}
                onChange={(event) => updateDecision(decisionIndex, { name: event.target.value })}
                placeholder="recipe_route"
              />
            </label>
            <label>
              <span>Priority</span>
              <input
                type="number"
                value={decision.priority ?? 0}
                onChange={(event) =>
                  updateDecision(decisionIndex, {
                    priority: Number(event.target.value),
                  })
                }
                min="0"
              />
            </label>
          </div>

          <label className={styles.fullWidthControl}>
            <span>Description</span>
            <input
              value={decision.description ?? ''}
              onChange={(event) =>
                updateDecision(decisionIndex, {
                  description: event.target.value,
                })
              }
              placeholder="Explain the policy outcome for this route"
            />
          </label>

          <DecisionExecutionEditor
            decision={decision}
            onChange={(patch) => updateDecision(decisionIndex, patch)}
          />

          <div className={styles.modelPoolHeader}>
            <span>Policy conditions</span>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => {
                const rules = decision.rules ?? { operator: 'AND', conditions: [] }
                updateDecision(decisionIndex, {
                  rules: {
                    ...rules,
                    conditions: [...(rules.conditions ?? []), emptyCondition()],
                  },
                })
              }}
            >
              Add condition
            </button>
          </div>
          <label>
            <span>Condition operator</span>
            <select
              value={decision.rules?.operator ?? 'AND'}
              onChange={(event) =>
                updateDecision(decisionIndex, {
                  rules: {
                    operator: event.target.value as 'AND' | 'OR' | 'NOT',
                    conditions: decision.rules?.conditions ?? [],
                  },
                })
              }
            >
              <option value="AND">AND</option>
              <option value="OR">OR</option>
              <option value="NOT">NOT</option>
            </select>
          </label>
          <div className={styles.modelPool}>
            {(decision.rules?.conditions ?? []).map((condition, conditionIndex) => {
              const isComposite =
                Boolean(condition.operator) || (condition.conditions?.length ?? 0) > 0
              return (
                <div
                  key={`${condition.name || condition.operator || 'condition'}-${conditionIndex}`}
                  className={styles.modelReferenceCard}
                >
                  {isComposite ? (
                    <p className={styles.editorHint}>
                      Nested condition preserved. Use the DSL/YAML editor to modify its tree.
                    </p>
                  ) : (
                    <>
                      <label>
                        <span>Type</span>
                        <select
                          value={condition.type ?? ''}
                          onChange={(event) =>
                            updateCondition(decisionIndex, conditionIndex, {
                              type: event.target.value,
                              label: undefined,
                              predicate: undefined,
                              on_error: undefined,
                            })
                          }
                        >
                          {conditionTypes.map((type) => (
                            <option key={type} value={type}>
                              {type}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Signal name</span>
                        <input
                          value={condition.name ?? ''}
                          onChange={(event) =>
                            updateCondition(decisionIndex, conditionIndex, {
                              name: event.target.value,
                            })
                          }
                          placeholder="signal_name"
                        />
                      </label>
                      {condition.type === 'classifier' ? (
                        <>
                          <label>
                            <span>Label</span>
                            <input
                              value={condition.label ?? ''}
                              onChange={(event) =>
                                updateCondition(decisionIndex, conditionIndex, {
                                  label: event.target.value,
                                })
                              }
                              placeholder="RISKY"
                            />
                          </label>
                          <label>
                            <span>Minimum score (gte)</span>
                            <input
                              type="number"
                              min="0"
                              max="1"
                              step="0.01"
                              value={condition.predicate?.gte ?? ''}
                              onChange={(event) =>
                                updateCondition(decisionIndex, conditionIndex, {
                                  predicate: {
                                    gte:
                                      event.target.value === ''
                                        ? undefined
                                        : Number(event.target.value),
                                  },
                                })
                              }
                            />
                          </label>
                          <label>
                            <span>On error</span>
                            <select
                              value={condition.on_error ?? 'no_match'}
                              onChange={(event) =>
                                updateCondition(decisionIndex, conditionIndex, {
                                  on_error: event.target.value as 'no_match' | 'match',
                                })
                              }
                            >
                              <option value="no_match">No match</option>
                              <option value="match">Match</option>
                            </select>
                          </label>
                        </>
                      ) : null}
                    </>
                  )}
                  <button
                    type="button"
                    className={styles.iconDangerButton}
                    aria-label={`Remove condition ${conditionIndex + 1}`}
                    onClick={() => removeCondition(decisionIndex, conditionIndex)}
                  >
                    ×
                  </button>
                </div>
              )
            })}
          </div>
        </article>
      ))}
      <button
        type="button"
        className={styles.addDecisionButton}
        onClick={() => onChange([...rows, emptyDecision()])}
      >
        Add decision
      </button>
    </div>
  )
}

const ALGORITHM_TYPES = [
  'static',
  'multi_factor',
  'latency_aware',
  'confidence',
  'ratings',
  'fusion',
  'remom',
  'workflows',
]

function DecisionExecutionEditor({
  decision,
  onChange,
}: {
  decision: DecisionConfig
  onChange: (patch: Partial<DecisionConfig>) => void
}) {
  const algorithmType =
    typeof decision.algorithm?.type === 'string' ? decision.algorithm.type : 'static'
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [algorithmJSON, setAlgorithmJSON] = useState(() =>
    JSON.stringify(decision.algorithm ?? { type: 'static' }, null, 2),
  )
  const [pluginsJSON, setPluginsJSON] = useState(() =>
    JSON.stringify(decision.plugins ?? [], null, 2),
  )
  const [iterationsJSON, setIterationsJSON] = useState(() =>
    JSON.stringify(decision.candidateIterations ?? [], null, 2),
  )
  const [error, setError] = useState('')
  const commit = () => {
    try {
      const algorithm = JSON.parse(algorithmJSON) as Record<string, unknown>
      const plugins = JSON.parse(pluginsJSON) as DecisionConfig['plugins']
      const candidateIterations = JSON.parse(iterationsJSON) as unknown
      onChange({ algorithm, plugins, candidateIterations })
      setError('')
    } catch {
      setError('Execution options must be valid JSON.')
    }
  }
  return (
    <section className={styles.modelPool}>
      <div className={styles.editorGrid}>
        <label>
          <span>Algorithm</span>
          <select
            value={algorithmType}
            onChange={(event) => {
              const type = event.target.value
              const next = type === 'static' ? { type } : { type, [type]: {} }
              setAlgorithmJSON(JSON.stringify(next, null, 2))
              onChange({ algorithm: next })
            }}
          >
            {ALGORITHM_TYPES.map((type) => (
              <option key={type} value={type}>
                {type.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className={styles.secondaryButton}
          onClick={() => setDetailsOpen((value) => !value)}
        >
          {detailsOpen ? 'Hide execution options' : 'Configure execution'}
        </button>
      </div>
      {detailsOpen ? (
        <div className={styles.editorGrid}>
          {error ? (
            <p className={styles.editorHint} role="alert">
              {error}
            </p>
          ) : null}
          <label className={styles.fullWidthControl}>
            <span>Algorithm configuration</span>
            <textarea
              rows={7}
              value={algorithmJSON}
              onChange={(event) => setAlgorithmJSON(event.target.value)}
              onBlur={commit}
              spellCheck={false}
            />
          </label>
          <label className={styles.fullWidthControl}>
            <span>Plugins</span>
            <textarea
              rows={7}
              value={pluginsJSON}
              onChange={(event) => setPluginsJSON(event.target.value)}
              onBlur={commit}
              spellCheck={false}
            />
          </label>
          <label className={styles.fullWidthControl}>
            <span>Candidate iterations</span>
            <textarea
              rows={7}
              value={iterationsJSON}
              onChange={(event) => setIterationsJSON(event.target.value)}
              onBlur={commit}
              spellCheck={false}
            />
          </label>
        </div>
      ) : null}
    </section>
  )
}
