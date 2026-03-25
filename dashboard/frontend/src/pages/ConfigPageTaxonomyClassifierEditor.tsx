import { useMemo } from 'react'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  emptyTaxonomyClassifierDraft,
  type TaxonomyClassifierCategory,
  type TaxonomyClassifierDraft,
  type TaxonomyClassifierTier,
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
