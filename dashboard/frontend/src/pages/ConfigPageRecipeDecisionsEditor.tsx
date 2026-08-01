import type { DecisionConfig, DecisionModelRef, NormalizedModel } from './configPageSupport'
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

export default function ConfigPageRecipeDecisionsEditor({
  value,
  models,
  onChange,
}: ConfigPageRecipeDecisionsEditorProps) {
  const rows = Array.isArray(value) ? value : []
  const modelOptions = models.map((model) => model.name)

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

  return (
    <div className={styles.decisionEditor}>
      <p className={styles.editorHint}>
        Model allocation is editable here. Existing rules, algorithms, and plugins are preserved.
      </p>
      {rows.map((decision, decisionIndex) => (
        <article key={`${decision.name || 'new'}-${decisionIndex}`} className={styles.decisionCard}>
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
