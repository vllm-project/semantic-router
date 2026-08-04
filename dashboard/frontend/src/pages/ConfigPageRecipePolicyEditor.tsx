import type { ClassifierSignal, ConfigSignals, MetadataSignal } from './configPageSupport'
import styles from './ConfigPageEntrypointsRecipesSection.module.css'

interface ConfigPageRecipePolicyEditorProps {
  value: ConfigSignals
  onChange: (value: ConfigSignals) => void
}

type MetadataPredicateType = 'equals' | 'in' | 'exists'

const emptyMetadataSignal = (): MetadataSignal => ({
  name: '',
  key: '',
  predicate: { equals: '' },
})

const emptyClassifierSignal = (): ClassifierSignal => ({
  name: '',
  type: 'local',
  model_path: '',
  labels: ['SAFE', 'MATCH'],
  use_cpu: true,
})

function metadataPredicateType(signal: MetadataSignal): MetadataPredicateType {
  if (signal.predicate.in !== undefined) return 'in'
  if (signal.predicate.exists !== undefined) return 'exists'
  return 'equals'
}

function metadataPredicateForType(
  signal: MetadataSignal,
  type: MetadataPredicateType,
): MetadataSignal['predicate'] {
  if (type === 'in') return { in: signal.predicate.in ?? [] }
  if (type === 'exists') return { exists: signal.predicate.exists ?? true }
  return { equals: signal.predicate.equals ?? '' }
}

export default function ConfigPageRecipePolicyEditor({
  value,
  onChange,
}: ConfigPageRecipePolicyEditorProps) {
  const signals = value ?? {}
  const metadata = signals.metadata ?? []
  const classifiers = signals.classifiers ?? []

  const updateMetadata = (index: number, patch: Partial<MetadataSignal>) => {
    onChange({
      ...signals,
      metadata: metadata.map((signal, signalIndex) =>
        signalIndex === index ? { ...signal, ...patch } : signal,
      ),
    })
  }

  const updateClassifier = (index: number, patch: Partial<ClassifierSignal>) => {
    onChange({
      ...signals,
      classifiers: classifiers.map((signal, signalIndex) =>
        signalIndex === index ? { ...signal, ...patch } : signal,
      ),
    })
  }

  return (
    <div className={styles.decisionEditor}>
      <p className={styles.editorHint}>
        These policy signals belong only to this recipe. Existing signal families are preserved.
      </p>

      <div className={styles.modelPoolHeader}>
        <span>Metadata signals</span>
        <button
          type="button"
          className={styles.secondaryButton}
          onClick={() =>
            onChange({
              ...signals,
              metadata: [...metadata, emptyMetadataSignal()],
            })
          }
        >
          Add metadata signal
        </button>
      </div>
      <div className={styles.modelPool}>
        {metadata.map((signal, index) => {
          const predicateType = metadataPredicateType(signal)
          return (
            <article key={`${signal.name || 'metadata'}-${index}`} className={styles.decisionCard}>
              <div className={styles.editorGrid}>
                <label>
                  <span>Name</span>
                  <input
                    value={signal.name}
                    onChange={(event) => updateMetadata(index, { name: event.target.value })}
                    placeholder="tenant_tier"
                  />
                </label>
                <label>
                  <span>Metadata key</span>
                  <input
                    value={signal.key}
                    onChange={(event) => updateMetadata(index, { key: event.target.value })}
                    placeholder="tier"
                  />
                </label>
                <label>
                  <span>Predicate</span>
                  <select
                    value={predicateType}
                    onChange={(event) =>
                      updateMetadata(index, {
                        predicate: metadataPredicateForType(
                          signal,
                          event.target.value as MetadataPredicateType,
                        ),
                      })
                    }
                  >
                    <option value="equals">Equals</option>
                    <option value="in">In list</option>
                    <option value="exists">Exists</option>
                  </select>
                </label>
                {predicateType === 'equals' ? (
                  <label>
                    <span>Value</span>
                    <input
                      value={signal.predicate.equals ?? ''}
                      onChange={(event) =>
                        updateMetadata(index, {
                          predicate: { equals: event.target.value },
                        })
                      }
                    />
                  </label>
                ) : null}
                {predicateType === 'in' ? (
                  <label>
                    <span>Values</span>
                    <input
                      value={(signal.predicate.in ?? []).join(', ')}
                      onChange={(event) =>
                        updateMetadata(index, {
                          predicate: {
                            in: event.target.value
                              .split(',')
                              .map((item) => item.trim())
                              .filter(Boolean),
                          },
                        })
                      }
                      placeholder="gold, platinum"
                    />
                  </label>
                ) : null}
                {predicateType === 'exists' ? (
                  <label className={styles.checkboxControl}>
                    <input
                      type="checkbox"
                      checked={signal.predicate.exists ?? true}
                      onChange={(event) =>
                        updateMetadata(index, {
                          predicate: { exists: event.target.checked },
                        })
                      }
                    />
                    <span>Must exist</span>
                  </label>
                ) : null}
              </div>
              <button
                type="button"
                className={styles.dangerButton}
                onClick={() =>
                  onChange({
                    ...signals,
                    metadata: metadata.filter((_, signalIndex) => signalIndex !== index),
                  })
                }
              >
                Remove metadata signal
              </button>
            </article>
          )
        })}
      </div>

      <div className={styles.modelPoolHeader}>
        <span>Classifier signals</span>
        <button
          type="button"
          className={styles.secondaryButton}
          onClick={() =>
            onChange({
              ...signals,
              classifiers: [...classifiers, emptyClassifierSignal()],
            })
          }
        >
          Add classifier signal
        </button>
      </div>
      <div className={styles.modelPool}>
        {classifiers.map((signal, index) => (
          <article key={`${signal.name || 'classifier'}-${index}`} className={styles.decisionCard}>
            <div className={styles.editorGrid}>
              <label>
                <span>Name</span>
                <input
                  value={signal.name}
                  onChange={(event) => updateClassifier(index, { name: event.target.value })}
                  placeholder="risk_score"
                />
              </label>
              <label>
                <span>Classifier type</span>
                <select
                  value={signal.type}
                  onChange={(event) => {
                    const type = event.target.value as ClassifierSignal['type']
                    updateClassifier(
                      index,
                      type === 'local'
                        ? {
                            type,
                            model: undefined,
                            instructions: undefined,
                            model_path: signal.model_path ?? '',
                            use_cpu: signal.use_cpu ?? true,
                          }
                        : {
                            type,
                            model_path: undefined,
                            use_cpu: undefined,
                            model: signal.model ?? '',
                            instructions: signal.instructions ?? '',
                          },
                    )
                  }}
                >
                  <option value="local">Local</option>
                  <option value="llm">LLM</option>
                </select>
              </label>
              <label>
                <span>Labels</span>
                <input
                  value={signal.labels.join(', ')}
                  onChange={(event) =>
                    updateClassifier(index, {
                      labels: event.target.value
                        .split(',')
                        .map((item) => item.trim())
                        .filter(Boolean),
                    })
                  }
                  placeholder="SAFE, RISKY"
                />
              </label>
              {signal.type === 'local' ? (
                <>
                  <label>
                    <span>Model path</span>
                    <input
                      value={signal.model_path ?? ''}
                      onChange={(event) =>
                        updateClassifier(index, {
                          model_path: event.target.value,
                        })
                      }
                      placeholder="models/risk"
                    />
                  </label>
                  <label className={styles.checkboxControl}>
                    <input
                      type="checkbox"
                      checked={signal.use_cpu === true}
                      onChange={(event) =>
                        updateClassifier(index, {
                          use_cpu: event.target.checked,
                        })
                      }
                    />
                    <span>Use CPU</span>
                  </label>
                </>
              ) : (
                <>
                  <label>
                    <span>External model</span>
                    <input
                      value={signal.model ?? ''}
                      onChange={(event) =>
                        updateClassifier(index, {
                          model: event.target.value,
                        })
                      }
                      placeholder="classification-judge"
                    />
                  </label>
                  <label className={styles.fullWidthControl}>
                    <span>Instructions</span>
                    <textarea
                      value={signal.instructions ?? ''}
                      onChange={(event) =>
                        updateClassifier(index, {
                          instructions: event.target.value,
                        })
                      }
                      placeholder="Return one declared label."
                    />
                  </label>
                </>
              )}
            </div>
            <button
              type="button"
              className={styles.dangerButton}
              onClick={() =>
                onChange({
                  ...signals,
                  classifiers: classifiers.filter((_, signalIndex) => signalIndex !== index),
                })
              }
            >
              Remove classifier signal
            </button>
          </article>
        ))}
      </div>
    </div>
  )
}
