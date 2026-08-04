import type {
  DecisionCondition,
  DecisionConfig,
  DecisionModelRef,
  NormalizedModel,
} from './configPageSupport'
import styles from './ConfigPageEntrypointsRecipesSection.module.css'

interface ConfigPageRecipeDecisionsEditorProps {
  value: DecisionConfig[]
  models: NormalizedModel[]
  onChange: (value: DecisionConfig[]) => void
}

const emptyReference = (): DecisionModelRef => ({
  model: '',
  use_reasoning: false,
})

const emptyDecision = (): DecisionConfig => ({
  name: '',
  description: '',
  priority: 100,
  rules: { operator: 'AND', conditions: [] },
  modelRefs: [emptyReference()],
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
  models,
  onChange,
}: ConfigPageRecipeDecisionsEditorProps) {
  const rows = Array.isArray(value) ? value : []
  const modelOptions = models.flatMap((model) => [
    model.name,
    ...(model.loras ?? []).map((adapter) => adapter.name),
  ])

  const updateDecision = (index: number, patch: Partial<DecisionConfig>) => {
    onChange(
      rows.map((decision, rowIndex) => (rowIndex === index ? { ...decision, ...patch } : decision)),
    )
  }

  const updateReference = (
    decisionIndex: number,
    referenceIndex: number,
    patch: Partial<DecisionModelRef>,
  ) => {
    const decision = rows[decisionIndex]
    const modelRefs = (decision.modelRefs ?? []).map((reference, rowIndex) =>
      rowIndex === referenceIndex ? { ...reference, ...patch } : reference,
    )
    updateDecision(decisionIndex, { modelRefs })
  }

  const addReference = (decisionIndex: number) => {
    const decision = rows[decisionIndex]
    updateDecision(decisionIndex, {
      modelRefs: [...(decision.modelRefs ?? []), emptyReference()],
    })
  }

  const removeReference = (decisionIndex: number, referenceIndex: number) => {
    const decision = rows[decisionIndex]
    updateDecision(decisionIndex, {
      modelRefs: (decision.modelRefs ?? []).filter((_, index) => index !== referenceIndex),
    })
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
        Model allocation is editable here. Existing rules, algorithms, and plugins are preserved.
      </p>
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

          <div className={styles.modelPoolHeader}>
            <span>Target model pool</span>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => addReference(decisionIndex)}
            >
              Add model
            </button>
          </div>

          <div className={styles.modelPool}>
            {(decision.modelRefs ?? []).map((reference, referenceIndex) => (
              <div
                key={`${reference.model || 'model'}-${referenceIndex}`}
                className={styles.modelReferenceCard}
              >
                <label>
                  <span>Model</span>
                  <select
                    value={reference.model ?? ''}
                    onChange={(event) =>
                      updateReference(decisionIndex, referenceIndex, {
                        model: event.target.value,
                      })
                    }
                  >
                    <option value="">Select model</option>
                    {reference.model && !modelOptions.includes(reference.model) ? (
                      <option value={reference.model}>{reference.model}</option>
                    ) : null}
                    {modelOptions.map((modelName) => (
                      <option key={modelName} value={modelName}>
                        {modelName}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Reasoning effort</span>
                  <select
                    value={reference.reasoning_effort ?? ''}
                    onChange={(event) =>
                      updateReference(decisionIndex, referenceIndex, {
                        reasoning_effort: event.target.value || undefined,
                      })
                    }
                  >
                    <option value="">Default</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </label>
                <label>
                  <span>LoRA adapter</span>
                  <input
                    value={reference.lora_name ?? ''}
                    onChange={(event) =>
                      updateReference(decisionIndex, referenceIndex, {
                        lora_name: event.target.value || undefined,
                      })
                    }
                    placeholder="Optional"
                  />
                </label>
                <label>
                  <span>Weight</span>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={reference.weight ?? ''}
                    onChange={(event) =>
                      updateReference(decisionIndex, referenceIndex, {
                        weight: event.target.value === '' ? undefined : Number(event.target.value),
                      })
                    }
                    placeholder="Optional"
                  />
                </label>
                <label className={styles.checkboxControl}>
                  <input
                    type="checkbox"
                    checked={reference.use_reasoning === true}
                    onChange={(event) =>
                      updateReference(decisionIndex, referenceIndex, {
                        use_reasoning: event.target.checked,
                      })
                    }
                  />
                  <span>Use reasoning</span>
                </label>
                <button
                  type="button"
                  className={styles.iconDangerButton}
                  aria-label={`Remove model reference ${referenceIndex + 1}`}
                  onClick={() => removeReference(decisionIndex, referenceIndex)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </article>
      ))}
      <button
        type="button"
        className={styles.addDecisionButton}
        onClick={() => onChange([...rows, emptyDecision()])}
      >
        Add recipe decision
      </button>
    </div>
  )
}
