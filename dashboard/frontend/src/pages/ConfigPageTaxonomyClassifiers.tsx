import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FieldConfig } from '../components/EditModal'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import routerStyles from './ConfigPageRouterConfigSection.module.css'
import styles from './ConfigPageTaxonomyClassifiers.module.css'

interface TaxonomySignalBinding {
  kind: 'tier' | 'category'
  value: string
}

interface TaxonomySignalReference {
  name: string
  bind: TaxonomySignalBinding
}

interface TaxonomyClassifierTier {
  name: string
  description?: string
}

interface TaxonomyClassifierCategory {
  name: string
  tier: string
  description?: string
  exemplars: string[]
}

interface TaxonomyClassifierRecord {
  name: string
  type: string
  builtin: boolean
  managed: boolean
  editable: boolean
  threshold: number
  security_threshold?: number
  description?: string
  source: {
    path: string
    taxonomy_file?: string
  }
  tiers: TaxonomyClassifierTier[]
  categories: TaxonomyClassifierCategory[]
  tier_groups?: Record<string, string[]>
  signal_references: TaxonomySignalReference[]
  bind_options: {
    tiers: string[]
    categories: string[]
  }
  load_error?: string
}

interface TaxonomyClassifierListResponse {
  items: TaxonomyClassifierRecord[]
}

interface TaxonomyClassifierDraft {
  name: string
  threshold: number
  security_threshold: number
  description: string
  tiers: TaxonomyClassifierTier[]
  categories: TaxonomyClassifierCategory[]
  tier_groups: Array<{
    name: string
    categories: string
  }>
}

interface TaxonomyClassifierEditorProps {
  value: unknown
  onChange: (value: TaxonomyClassifierDraft) => void
  disableName?: boolean
}

interface ConfigPageTaxonomyClassifiersProps {
  isReadonly: boolean
  openEditModal: OpenEditModal
}

function emptyTaxonomyClassifierDraft(): TaxonomyClassifierDraft {
  return {
    name: '',
    threshold: 0.3,
    security_threshold: 0.25,
    description: '',
    tiers: [],
    categories: [
      {
        name: '',
        tier: '',
        description: '',
        exemplars: [''],
      },
    ],
    tier_groups: [],
  }
}

function isDraft(value: unknown): value is TaxonomyClassifierDraft {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function classifierDraftFromRecord(record: TaxonomyClassifierRecord): TaxonomyClassifierDraft {
  return {
    name: record.name,
    threshold: record.threshold,
    security_threshold: record.security_threshold ?? record.threshold,
    description: record.description ?? '',
    tiers: record.tiers.map((tier) => ({
      name: tier.name,
      description: tier.description ?? '',
    })),
    categories: record.categories.map((category) => ({
      name: category.name,
      tier: category.tier,
      description: category.description ?? '',
      exemplars: category.exemplars.length > 0 ? [...category.exemplars] : [''],
    })),
    tier_groups: Object.entries(record.tier_groups ?? {}).map(([name, categories]) => ({
      name,
      categories: categories.join(', '),
    })),
  }
}

function payloadFromDraft(draft: TaxonomyClassifierDraft) {
  const tierGroups = draft.tier_groups.reduce<Record<string, string[]>>((acc, group) => {
    const name = group.name.trim()
    if (!name) {
      return acc
    }
    const categories = group.categories
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    if (categories.length > 0) {
      acc[name] = categories
    }
    return acc
  }, {})

  return {
    name: draft.name.trim(),
    threshold: draft.threshold,
    security_threshold: draft.security_threshold,
    description: draft.description.trim(),
    tiers: draft.tiers.map((tier) => ({
      name: tier.name.trim(),
      description: tier.description?.trim() || '',
    })),
    categories: draft.categories.map((category) => ({
      name: category.name.trim(),
      tier: category.tier.trim(),
      description: category.description?.trim() || '',
      exemplars: category.exemplars
        .map((exemplar) => exemplar.trim())
        .filter(Boolean),
    })),
    tier_groups: tierGroups,
  }
}

function TaxonomyClassifierEditor({ value, onChange, disableName = false }: TaxonomyClassifierEditorProps) {
  const draft = isDraft(value) ? value : emptyTaxonomyClassifierDraft()
  const tierOptions = useMemo(
    () => draft.tiers.map((tier) => tier.name.trim()).filter(Boolean),
    [draft.tiers]
  )

  const updateDraft = (next: TaxonomyClassifierDraft) => {
    onChange(next)
  }

  const updateTier = (index: number, patch: Partial<TaxonomyClassifierTier>) => {
    const tiers = draft.tiers.map((tier, tierIndex) =>
      tierIndex === index ? { ...tier, ...patch } : tier
    )
    updateDraft({ ...draft, tiers })
  }

  const updateCategory = (index: number, patch: Partial<TaxonomyClassifierCategory>) => {
    const categories = draft.categories.map((category, categoryIndex) =>
      categoryIndex === index ? { ...category, ...patch } : category
    )
    updateDraft({ ...draft, categories })
  }

  const updateGroup = (index: number, patch: Partial<TaxonomyClassifierDraft['tier_groups'][number]>) => {
    const tier_groups = draft.tier_groups.map((group, groupIndex) =>
      groupIndex === index ? { ...group, ...patch } : group
    )
    updateDraft({ ...draft, tier_groups })
  }

  return (
    <div className={styles.editor}>
      <div className={styles.editorBlock}>
        <div className={styles.editorGrid}>
          <label className={styles.editorField}>
            <span className={styles.editorLabel}>Classifier Name</span>
            <input
              className={styles.editorInput}
              value={draft.name}
              disabled={disableName}
              placeholder="privacy_classifier"
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
          <label className={styles.editorField}>
            <span className={styles.editorLabel}>Security Threshold</span>
            <input
              className={styles.editorInput}
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={draft.security_threshold}
              onChange={(event) => updateDraft({ ...draft, security_threshold: Number(event.target.value) })}
            />
          </label>
        </div>
        <label className={styles.editorField}>
          <span className={styles.editorLabel}>Description</span>
          <textarea
            className={styles.editorTextarea}
            rows={3}
            value={draft.description}
            placeholder="What this classifier routes and why."
            onChange={(event) => updateDraft({ ...draft, description: event.target.value })}
          />
        </label>
      </div>

      <div className={styles.editorBlock}>
        <div className={styles.editorSectionHeader}>
          <div>
            <h3 className={styles.editorSectionTitle}>Tiers</h3>
            <p className={styles.editorSectionHint}>Stable routing buckets taxonomy signals can bind to.</p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                tiers: [...draft.tiers, { name: '', description: '' }],
              })
            }
          >
            Add Tier
          </button>
        </div>
        <div className={styles.stack}>
          {draft.tiers.map((tier, index) => (
            <div key={`tier-${index}`} className={styles.editorRow}>
              <input
                className={styles.editorInput}
                placeholder="privacy_policy"
                value={tier.name}
                onChange={(event) => updateTier(index, { name: event.target.value })}
              />
              <input
                className={styles.editorInput}
                placeholder="Optional tier description"
                value={tier.description ?? ''}
                onChange={(event) => updateTier(index, { description: event.target.value })}
              />
              <button
                type="button"
                className={styles.removeButton}
                onClick={() =>
                  updateDraft({
                    ...draft,
                    tiers: draft.tiers.filter((_, tierIndex) => tierIndex !== index),
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
            <h3 className={styles.editorSectionTitle}>Categories</h3>
            <p className={styles.editorSectionHint}>Fine-grained labels with exemplar sets and tier bindings.</p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                categories: [
                  ...draft.categories,
                  { name: '', tier: tierOptions[0] ?? '', description: '', exemplars: [''] },
                ],
              })
            }
          >
            Add Category
          </button>
        </div>
        <div className={styles.categoryList}>
          {draft.categories.map((category, index) => (
            <div key={`category-${index}`} className={styles.categoryCard}>
              <div className={styles.editorGrid}>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Category Name</span>
                  <input
                    className={styles.editorInput}
                    placeholder="proprietary_code"
                    value={category.name}
                    onChange={(event) => updateCategory(index, { name: event.target.value })}
                  />
                </label>
                <label className={styles.editorField}>
                  <span className={styles.editorLabel}>Tier</span>
                  <select
                    className={styles.editorSelect}
                    value={category.tier}
                    onChange={(event) => updateCategory(index, { tier: event.target.value })}
                  >
                    <option value="">Select tier</option>
                    {tierOptions.map((tierName) => (
                      <option key={tierName} value={tierName}>
                        {tierName}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <label className={styles.editorField}>
                <span className={styles.editorLabel}>Description</span>
                <input
                  className={styles.editorInput}
                  placeholder="Optional category description"
                  value={category.description ?? ''}
                  onChange={(event) => updateCategory(index, { description: event.target.value })}
                />
              </label>
              <label className={styles.editorField}>
                <span className={styles.editorLabel}>Exemplars</span>
                <textarea
                  className={styles.editorTextarea}
                  rows={4}
                  placeholder="One exemplar per line"
                  value={category.exemplars.join('\n')}
                  onChange={(event) =>
                    updateCategory(index, {
                      exemplars: event.target.value.split('\n'),
                    })
                  }
                />
              </label>
              <div className={styles.categoryFooter}>
                <span className={styles.footerHint}>Signals can bind to this category by name.</span>
                <button
                  type="button"
                  className={styles.removeButton}
                  onClick={() =>
                    updateDraft({
                      ...draft,
                      categories: draft.categories.filter((_, categoryIndex) => categoryIndex !== index),
                    })
                  }
                >
                  Remove Category
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.editorBlock}>
        <div className={styles.editorSectionHeader}>
          <div>
            <h3 className={styles.editorSectionTitle}>Tier Groups</h3>
            <p className={styles.editorSectionHint}>Optional category groups used by built-in contrastive scoring.</p>
          </div>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() =>
              updateDraft({
                ...draft,
                tier_groups: [...draft.tier_groups, { name: '', categories: '' }],
              })
            }
          >
            Add Group
          </button>
        </div>
        <div className={styles.stack}>
          {draft.tier_groups.map((group, index) => (
            <div key={`group-${index}`} className={styles.editorRow}>
              <input
                className={styles.editorInput}
                placeholder="privacy_categories"
                value={group.name}
                onChange={(event) => updateGroup(index, { name: event.target.value })}
              />
              <input
                className={styles.editorInput}
                placeholder="Comma-separated category names"
                value={group.categories}
                onChange={(event) => updateGroup(index, { categories: event.target.value })}
              />
              <button
                type="button"
                className={styles.removeButton}
                onClick={() =>
                  updateDraft({
                    ...draft,
                    tier_groups: draft.tier_groups.filter((_, groupIndex) => groupIndex !== index),
                  })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function formatSignalReference(reference: TaxonomySignalReference): string {
  return `${reference.name} -> ${reference.bind.kind}:${reference.bind.value}`
}

export default function ConfigPageTaxonomyClassifiers({
  isReadonly,
  openEditModal,
}: ConfigPageTaxonomyClassifiersProps) {
  const [classifiers, setClassifiers] = useState<TaxonomyClassifierRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadClassifiers = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/router/config/classifiers')
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      const payload = await response.json() as TaxonomyClassifierListResponse
      setClassifiers(payload.items || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load taxonomy classifiers')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadClassifiers()
  }, [loadClassifiers])

  const counts = useMemo(() => ({
    total: classifiers.length,
    builtin: classifiers.filter((classifier) => classifier.builtin).length,
    custom: classifiers.filter((classifier) => classifier.editable).length,
  }), [classifiers])

  const classifierEditorField = useCallback(
    (disableName: boolean): FieldConfig[] => [
      {
        name: 'draft',
        label: 'Classifier',
        type: 'custom',
        customRender: (value, onChange) => (
          <TaxonomyClassifierEditor
            value={value}
            onChange={(nextValue) => onChange(nextValue)}
            disableName={disableName}
          />
        ),
      },
    ],
    []
  )

  const persistClassifier = useCallback(async (endpoint: string, method: 'POST' | 'PUT', draft: TaxonomyClassifierDraft) => {
    const response = await fetch(endpoint, {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payloadFromDraft(draft)),
    })
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
    }
    await loadClassifiers()
  }, [loadClassifiers])

  const openCreateModal = () => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      'Add Taxonomy Classifier',
      { draft: emptyTaxonomyClassifierDraft() },
      classifierEditorField(false),
      async (data) => {
        await persistClassifier('/api/router/config/classifiers', 'POST', data.draft)
      },
      'add',
    )
  }

  const openEditClassifierModal = (classifier: TaxonomyClassifierRecord) => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      `Edit ${classifier.name}`,
      { draft: classifierDraftFromRecord(classifier) },
      classifierEditorField(true),
      async (data) => {
        await persistClassifier(`/api/router/config/classifiers/${classifier.name}`, 'PUT', data.draft)
      },
      'edit',
    )
  }

  const handleDeleteClassifier = async (classifier: TaxonomyClassifierRecord) => {
    if (!window.confirm(`Delete classifier "${classifier.name}"? This also removes its managed directory.`)) {
      return
    }
    setError(null)
    try {
      const response = await fetch(`/api/router/config/classifiers/${classifier.name}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      await loadClassifiers()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete taxonomy classifier')
    }
  }

  return (
    <section id="taxonomy-classifiers" className={styles.section}>
      <div className={routerStyles.blockHeader}>
        <div>
          <h2 className={routerStyles.blockTitle}>Taxonomy Classifiers</h2>
          <p className={routerStyles.blockDescription}>
            Manage router-wide taxonomy packages loaded from `global.model_catalog.classifiers[]`. Built-ins stay read-only; custom classifiers live under `classifiers/custom/` and expose tier/category bindings for taxonomy signals.
          </p>
        </div>
        <div className={styles.actionRow}>
          <button type="button" className={styles.secondaryButton} onClick={() => void loadClassifiers()}>
            Refresh
          </button>
          {!isReadonly && (
            <button type="button" className={styles.primaryButton} onClick={openCreateModal}>
              Add Classifier
            </button>
          )}
        </div>
      </div>

      <div className={routerStyles.overviewGrid}>
        <div className={routerStyles.overviewCard}>
          <span className={routerStyles.overviewLabel}>Total Classifiers</span>
          <strong className={routerStyles.overviewValue}>{counts.total}</strong>
          <span className={routerStyles.overviewHint}>All classifier instances currently loaded by the router.</span>
        </div>
        <div className={routerStyles.overviewCard}>
          <span className={routerStyles.overviewLabel}>Built-In</span>
          <strong className={routerStyles.overviewValue}>{counts.builtin}</strong>
          <span className={routerStyles.overviewHint}>Router-shipped default classifiers that stay read-only in the dashboard.</span>
        </div>
        <div className={routerStyles.overviewCard}>
          <span className={routerStyles.overviewLabel}>Custom</span>
          <strong className={routerStyles.overviewValue}>{counts.custom}</strong>
          <span className={routerStyles.overviewHint}>Managed classifier directories created under `classifiers/custom/`.</span>
        </div>
      </div>

      {loading && <div className={styles.notice}>Loading classifier catalog…</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div className={routerStyles.sectionGrid}>
        {classifiers.map((classifier) => (
          <article key={classifier.name} className={routerStyles.systemCard}>
            <div className={routerStyles.cardHeader}>
              <div className={routerStyles.cardCopy}>
                <span className={routerStyles.cardEyebrow}>Classifier</span>
                <h3 className={routerStyles.cardTitle}>{classifier.name}</h3>
                <p className={routerStyles.cardDescription}>
                  {classifier.description || 'No classifier description provided.'}
                </p>
              </div>
              <div className={routerStyles.cardBadges}>
                <span className={`${routerStyles.badge} ${classifier.builtin ? routerStyles.badgeActive : routerStyles.badgeInfo}`}>
                  {classifier.builtin ? 'Built-in' : classifier.editable ? 'Custom managed' : 'External'}
                </span>
                <span className={`${routerStyles.badge} ${classifier.load_error ? routerStyles.badgeInactive : routerStyles.badgeActive}`}>
                  {classifier.load_error ? 'Load issue' : classifier.type}
                </span>
              </div>
            </div>

            <div className={routerStyles.summaryList}>
              <div className={routerStyles.summaryRow}>
                <span className={routerStyles.summaryLabel}>Thresholds</span>
                <span className={routerStyles.summaryValue}>
                  base {classifier.threshold} / security {classifier.security_threshold ?? classifier.threshold}
                </span>
              </div>
              <div className={routerStyles.summaryRow}>
                <span className={routerStyles.summaryLabel}>Source</span>
                <span className={routerStyles.summaryValue}>{classifier.source.path}</span>
              </div>
              <div className={routerStyles.summaryRow}>
                <span className={routerStyles.summaryLabel}>Signals</span>
                <span className={routerStyles.summaryValue}>
                  {classifier.signal_references.length > 0 ? classifier.signal_references.length : 'Unused'}
                </span>
              </div>
              {classifier.load_error && (
                <div className={routerStyles.summaryRow}>
                  <span className={routerStyles.summaryLabel}>Load Error</span>
                  <span className={routerStyles.summaryValue}>{classifier.load_error}</span>
                </div>
              )}
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Bindable Tiers</div>
              <div className={styles.tagList}>
                {classifier.bind_options.tiers.map((tier) => (
                  <span key={`${classifier.name}-tier-${tier}`} className={styles.tag}>
                    {tier}
                  </span>
                ))}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Categories</div>
              <div className={styles.tagList}>
                {classifier.categories.map((category) => (
                  <span key={`${classifier.name}-category-${category.name}`} className={styles.tag}>
                    {category.name} · {category.tier} · {category.exemplars.length} exemplar{category.exemplars.length === 1 ? '' : 's'}
                  </span>
                ))}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Taxonomy Signal References</div>
              <div className={styles.tagList}>
                {classifier.signal_references.length > 0 ? (
                  classifier.signal_references.map((reference) => (
                    <span key={`${classifier.name}-signal-${reference.name}`} className={styles.tag}>
                      {formatSignalReference(reference)}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No taxonomy signals reference this classifier yet.</span>
                )}
              </div>
            </div>

            <div className={styles.footer}>
              <div className={styles.footerHint}>
                Signal bind options: {classifier.bind_options.tiers.length} tiers, {classifier.bind_options.categories.length} categories.
              </div>
              {classifier.editable && !isReadonly && (
                <div className={styles.actionRow}>
                  <button type="button" className={styles.secondaryButton} onClick={() => openEditClassifierModal(classifier)}>
                    Edit
                  </button>
                  <button
                    type="button"
                    className={styles.removeButton}
                    onClick={() => {
                      void handleDeleteClassifier(classifier)
                    }}
                  >
                    Delete
                  </button>
                </div>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}
