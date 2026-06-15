import { useMemo } from 'react'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  emptyTaxonomyClassifierDraft,
  type TaxonomyClassifierDraft,
} from './configPageTaxonomyClassifierSupport'

interface TaxonomyClassifierEditorProps {
  value: unknown
  onChange: (value: TaxonomyClassifierDraft) => void
  disableName?: boolean
}

function isDraft(value: unknown): value is TaxonomyClassifierDraft {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export default function ConfigPageTaxonomyClassifierEditor({
  value,
  onChange,
  disableName = false,
}: TaxonomyClassifierEditorProps) {
  const draft = isDraft(value) ? value : emptyTaxonomyClassifierDraft()
  const labelOptions = useMemo(
    () => draft.labels.map((label) => label.name.trim()).filter(Boolean),
    [draft.labels]
  )
  const groupOptions = useMemo(
    () => draft.groups.map((group) => group.name.trim()).filter(Boolean),
    [draft.groups]
  )

  const updateDraft = (next: TaxonomyClassifierDraft) => {
    onChange(next)
  }

  return (
    <div className={styles.editor}>
      <div className={styles.editorBlock}>
        <div className={styles.editorGrid}>
          <label className={styles.editorField}>
            <span className={styles.editorLabel}>Knowledge Base Name</span>
            <input
              className={styles.editorInput}
              value={draft.name}
              disabled={disableName}
              placeholder="privacy_kb"
              onChange={(event) => updateDraft({ ...draft, name: event.target.value })}
            />
          </label>
          <label className={styles.editorField}>
            <span className={styles.editorLabel}>Threshold</span>
            <input
              className={styles.editorInput}
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={draft.threshold}
              onChange={(event) => updateDraft({ ...draft, threshold: Number(event.target.value) })}
            />
          </label>
        </div>
        <label className={styles.editorField}>
          <span className={styles.editorLabel}>Description</span>
          <textarea
            className={styles.editorTextarea}
            rows={3}
            value={draft.description}
            placeholder="What this knowledge base is used for."
            onChange={(event) => updateDraft({ ...draft, description: event.target.value })}
          />
        </label>
      </div>

      <div className={styles.editorBlock}>
        <div className={styles.editorSectionHeader}>
          <div>
            <h3 className={styles.editorSectionTitle}>Labels</h3>
            <p className={styles.editorSectionHint}>
              Labels are the atomic semantic units. Each label needs exemplar text so the KB can score requests.
            </p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                labels: [
                  ...draft.labels,
                  { name: '', description: '', exemplars: [''] },
                ],
              })
            }
          >
            Add Label
          </button>
        </div>
        <div className={styles.categoryList}>
          {draft.labels.map((label, index) => (
            <div key={`label-${index}`} className={styles.categoryCard}>
              <div className={styles.editorGrid}>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Label Name</span>
                  <input
                    className={styles.editorInput}
                    placeholder="prompt_injection"
                    value={label.name}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        labels: draft.labels.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, name: event.target.value } : entry
                        ),
                      })
                    }
                  />
                </label>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Description</span>
                  <input
                    className={styles.editorInput}
                    placeholder="Optional label description"
                    value={label.description ?? ''}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        labels: draft.labels.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, description: event.target.value } : entry
                        ),
                      })
                    }
                  />
                </label>
              </div>
              <label className={styles.editorField}>
                <span className={styles.editorLabel}>Exemplars</span>
                <textarea
                  className={styles.editorTextarea}
                  rows={4}
                  placeholder="One exemplar per line"
                  value={label.exemplars.join('\n')}
                  onChange={(event) =>
                    updateDraft({
                      ...draft,
                      labels: draft.labels.map((entry, entryIndex) =>
                        entryIndex === index
                          ? { ...entry, exemplars: event.target.value.split('\n') }
                          : entry
                      ),
                    })
                  }
                />
              </label>
              <div className={styles.categoryFooter}>
                <span className={styles.footerHint}>Signals can bind to this label by name.</span>
                <button
                  type="button"
                  className={styles.removeButton}
                  onClick={() =>
                    updateDraft({
                      ...draft,
                      labels: draft.labels.filter((_, entryIndex) => entryIndex !== index),
                      groups: draft.groups.map((group) => ({
                        ...group,
                        labels: group.labels
                          .split(',')
                          .map((item) => item.trim())
                          .filter((item) => item && item !== label.name)
                          .join(', '),
                      })),
                      label_thresholds: draft.label_thresholds.filter((entry) => entry.label !== label.name),
                    })
                  }
                >
                  Remove Label
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.editorBlock}>
        <div className={styles.editorSectionHeader}>
          <div>
            <h3 className={styles.editorSectionTitle}>Groups</h3>
            <p className={styles.editorSectionHint}>
              Groups collect labels into higher-level routing concepts. Use comma-separated label names.
            </p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                groups: [...draft.groups, { name: '', labels: '' }],
              })
            }
          >
            Add Group
          </button>
        </div>
        <div className={styles.stack}>
          {draft.groups.map((group, index) => (
            <div key={`group-${index}`} className={styles.editorRow}>
              <input
                className={styles.editorInput}
                placeholder="private"
                value={group.name}
                onChange={(event) =>
                  updateDraft({
                    ...draft,
                    groups: draft.groups.map((entry, entryIndex) =>
                      entryIndex === index ? { ...entry, name: event.target.value } : entry
                    ),
                  })
                }
              />
              <input
                className={styles.editorInput}
                placeholder={labelOptions.length > 0 ? labelOptions.join(', ') : 'Comma-separated label names'}
                value={group.labels}
                onChange={(event) =>
                  updateDraft({
                    ...draft,
                    groups: draft.groups.map((entry, entryIndex) =>
                      entryIndex === index ? { ...entry, labels: event.target.value } : entry
                    ),
                  })
                }
              />
              <button
                type="button"
                className={styles.removeButton}
                onClick={() =>
                  updateDraft({
                    ...draft,
                    groups: draft.groups.filter((_, entryIndex) => entryIndex !== index),
                    metrics: draft.metrics.filter(
                      (metric) => metric.positive_group !== group.name && metric.negative_group !== group.name
                    ),
                  })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.editorBlock}>
        <div className={styles.editorSectionHeader}>
          <div>
            <h3 className={styles.editorSectionTitle}>Label Threshold Overrides</h3>
            <p className={styles.editorSectionHint}>
              Override the base threshold for specific labels when they need tighter or looser matching.
            </p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                label_thresholds: [
                  ...draft.label_thresholds,
                  { label: labelOptions[0] ?? '', threshold: draft.threshold },
                ],
              })
            }
          >
            Add Override
          </button>
        </div>
        <div className={styles.stack}>
          {draft.label_thresholds.map((entry, index) => (
            <div key={`label-threshold-${index}`} className={styles.editorRow}>
              <select
                className={styles.editorSelect}
                value={entry.label}
                onChange={(event) =>
                  updateDraft({
                    ...draft,
                    label_thresholds: draft.label_thresholds.map((item, itemIndex) =>
                      itemIndex === index ? { ...item, label: event.target.value } : item
                    ),
                  })
                }
              >
                <option value="">Select label</option>
                {labelOptions.map((labelName) => (
                  <option key={labelName} value={labelName}>
                    {labelName}
                  </option>
                ))}
              </select>
              <input
                className={styles.editorInput}
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={entry.threshold}
                onChange={(event) =>
                  updateDraft({
                    ...draft,
                    label_thresholds: draft.label_thresholds.map((item, itemIndex) =>
                      itemIndex === index ? { ...item, threshold: Number(event.target.value) } : item
                    ),
                  })
                }
              />
              <button
                type="button"
                className={styles.removeButton}
                onClick={() =>
                  updateDraft({
                    ...draft,
                    label_thresholds: draft.label_thresholds.filter((_, itemIndex) => itemIndex !== index),
                  })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.editorBlock}>
        <div className={styles.editorSectionHeader}>
          <div>
            <h3 className={styles.editorSectionTitle}>Metrics</h3>
            <p className={styles.editorSectionHint}>
              Metrics expose numeric outputs for projections. The first generic metric is group margin.
            </p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                metrics: [
                  ...draft.metrics,
                  {
                    name: '',
                    type: 'group_margin',
                    positive_group: groupOptions[0],
                    negative_group: groupOptions[1] ?? groupOptions[0],
                  },
                ],
              })
            }
          >
            Add Metric
          </button>
        </div>
        <div className={styles.stack}>
          {draft.metrics.map((metric, index) => (
            <div key={`metric-${index}`} className={styles.categoryCard}>
              <div className={styles.editorGrid}>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Metric Name</span>
                  <input
                    className={styles.editorInput}
                    placeholder="private_vs_public"
                    value={metric.name}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        metrics: draft.metrics.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, name: event.target.value } : entry
                        ),
                      })
                    }
                  />
                </label>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Metric Type</span>
                  <select
                    className={styles.editorSelect}
                    value={metric.type}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        metrics: draft.metrics.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, type: event.target.value } : entry
                        ),
                      })
                    }
                  >
                    <option value="group_margin">group_margin</option>
                  </select>
                </label>
              </div>
              <div className={styles.editorGrid}>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Positive Group</span>
                  <select
                    className={styles.editorSelect}
                    value={metric.positive_group ?? ''}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        metrics: draft.metrics.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, positive_group: event.target.value } : entry
                        ),
                      })
                    }
                  >
                    <option value="">Select group</option>
                    {groupOptions.map((groupName) => (
                      <option key={`positive-${groupName}`} value={groupName}>
                        {groupName}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Negative Group</span>
                  <select
                    className={styles.editorSelect}
                    value={metric.negative_group ?? ''}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        metrics: draft.metrics.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, negative_group: event.target.value } : entry
                        ),
                      })
                    }
                  >
                    <option value="">Select group</option>
                    {groupOptions.map((groupName) => (
                      <option key={`negative-${groupName}`} value={groupName}>
                        {groupName}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className={styles.categoryFooter}>
                <span className={styles.footerHint}>Projection inputs reference this metric by name.</span>
                <button
                  type="button"
                  className={styles.removeButton}
                  onClick={() =>
                    updateDraft({
                      ...draft,
                      metrics: draft.metrics.filter((_, entryIndex) => entryIndex !== index),
                    })
                  }
                >
                  Remove Metric
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
