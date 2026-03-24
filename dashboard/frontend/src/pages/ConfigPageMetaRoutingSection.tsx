import { useEffect, useMemo, useState } from 'react'

import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import TableHeader from '../components/TableHeader'

import { cloneConfigData, ensureRoutingConfig } from './configPageCanonicalization'
import styles from './ConfigPage.module.css'
import metaStyles from './ConfigPageMetaRoutingSection.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import type { MetaRoutingConfig } from './configPageSupport'
import {
  META_MODE_OPTIONS,
  AllowedActionFormState,
  AllowedActionRow,
  asOptionalNumber,
  buildActionFields,
  buildConfiguredFamilyOptions,
  buildDisagreementFields,
  buildRequiredFamilyFields,
  cloneMetaRoutingConfig,
  ConfigPageMetaRoutingSectionProps,
  createAllowedActionColumns,
  createDisagreementColumns,
  createRequiredFamilyColumns,
  FamilyDisagreementFormState,
  FamilyDisagreementRow,
  metaRoutingConfigFingerprint,
  normalizeMetaRoutingConfig,
  RequiredFamilyFormState,
  RequiredFamilyRow,
} from './configPageMetaRoutingSupport'

export default function ConfigPageMetaRoutingSection({
  config,
  isReadonly,
  saveConfig,
  openEditModal,
}: ConfigPageMetaRoutingSectionProps) {
  const sourceMetaRouting = config?.routing?.meta
  const [enabled, setEnabled] = useState(Boolean(sourceMetaRouting))
  const [draft, setDraft] = useState<MetaRoutingConfig>(cloneMetaRoutingConfig(sourceMetaRouting))
  const [saving, setSaving] = useState(false)

  const sourceFingerprint = useMemo(
    () => metaRoutingConfigFingerprint(sourceMetaRouting ? normalizeMetaRoutingConfig(sourceMetaRouting) : null),
    [sourceMetaRouting],
  )

  useEffect(() => {
    setEnabled(Boolean(sourceMetaRouting))
    setDraft(cloneMetaRoutingConfig(sourceMetaRouting))
  }, [sourceFingerprint, sourceMetaRouting])

  const draftFingerprint = useMemo(
    () => metaRoutingConfigFingerprint(enabled ? normalizeMetaRoutingConfig(draft) : null),
    [draft, enabled],
  )
  const isDirty = draftFingerprint !== sourceFingerprint

  const requiredFamilyRows = useMemo<RequiredFamilyRow[]>(
    () => (draft.trigger_policy?.required_families || []).map((family, index) => ({ ...family, id: `${family.type}-${index}` })),
    [draft.trigger_policy?.required_families],
  )
  const disagreementRows = useMemo<FamilyDisagreementRow[]>(
    () => (draft.trigger_policy?.family_disagreements || []).map((family, index) => ({ ...family, id: `${family.cheap}-${family.expensive}-${index}` })),
    [draft.trigger_policy?.family_disagreements],
  )
  const allowedActionRows = useMemo<AllowedActionRow[]>(
    () => (draft.allowed_actions || []).map((action, index) => ({ ...action, id: `${action.type}-${index}` })),
    [draft.allowed_actions],
  )

  const updateDraft = (updater: (current: MetaRoutingConfig) => MetaRoutingConfig) => {
    setDraft((current) => updater(cloneMetaRoutingConfig(current)))
  }

  const persistDraft = async (nextEnabled: boolean, nextDraft: MetaRoutingConfig) => {
    if (!config) {
      return
    }
    setSaving(true)
    try {
      const nextConfig = cloneConfigData(config)
      const routing = ensureRoutingConfig(nextConfig)
      if (!nextEnabled) {
        delete routing.meta
      } else {
        routing.meta = normalizeMetaRoutingConfig(nextDraft)
      }
      await saveConfig(nextConfig)
    } finally {
      setSaving(false)
    }
  }

  const resetDraft = () => {
    setEnabled(Boolean(sourceMetaRouting))
    setDraft(cloneMetaRoutingConfig(sourceMetaRouting))
  }

  const configuredFamilyOptions = useMemo(() => buildConfiguredFamilyOptions(config), [config])
  const requiredFamilyFields: FieldConfig<RequiredFamilyFormState>[] = useMemo(
    () => buildRequiredFamilyFields(configuredFamilyOptions),
    [configuredFamilyOptions],
  )
  const disagreementFields: FieldConfig<FamilyDisagreementFormState>[] = useMemo(
    () => buildDisagreementFields(configuredFamilyOptions),
    [configuredFamilyOptions],
  )
  const actionFields: FieldConfig<AllowedActionFormState>[] = useMemo(
    () => buildActionFields(configuredFamilyOptions),
    [configuredFamilyOptions],
  )
  const requiredFamilyColumns: Column<RequiredFamilyRow>[] = useMemo(() => createRequiredFamilyColumns(), [])
  const disagreementColumns: Column<FamilyDisagreementRow>[] = useMemo(() => createDisagreementColumns(), [])
  const allowedActionColumns: Column<AllowedActionRow>[] = useMemo(() => createAllowedActionColumns(), [])

  const openAddRequiredFamily = () => {
    openEditModal<RequiredFamilyFormState>(
      'Add Required Signal Family',
      { type: configuredFamilyOptions[0] || 'embedding' },
      requiredFamilyFields,
      async (form) => {
        updateDraft((current) => {
          const next = current
          const triggerPolicy = next.trigger_policy || {}
          next.trigger_policy = {
            ...triggerPolicy,
            required_families: [
              ...(triggerPolicy.required_families || []),
              {
                type: form.type.trim(),
                min_confidence: asOptionalNumber(form.min_confidence),
                min_matches: asOptionalNumber(form.min_matches),
              },
            ],
          }
          return next
        })
      },
      'add',
    )
  }

  const openEditRequiredFamily = (row: RequiredFamilyRow) => {
    const index = requiredFamilyRows.findIndex((candidate) => candidate.id === row.id)
    if (index < 0) {
      return
    }
    openEditModal<RequiredFamilyFormState>(
      `Edit Required Family: ${row.type}`,
      {
        type: row.type,
        min_confidence: row.min_confidence,
        min_matches: row.min_matches,
      },
      requiredFamilyFields,
      async (form) => {
        updateDraft((current) => {
          const next = current
          const triggerPolicy = next.trigger_policy || {}
          const families = [...(triggerPolicy.required_families || [])]
          families[index] = {
            type: form.type.trim(),
            min_confidence: asOptionalNumber(form.min_confidence),
            min_matches: asOptionalNumber(form.min_matches),
          }
          next.trigger_policy = { ...triggerPolicy, required_families: families }
          return next
        })
      },
    )
  }

  const deleteRequiredFamily = (row: RequiredFamilyRow) => {
    const index = requiredFamilyRows.findIndex((candidate) => candidate.id === row.id)
    if (index < 0) {
      return
    }
    updateDraft((current) => {
      const next = current
      const triggerPolicy = next.trigger_policy || {}
      next.trigger_policy = {
        ...triggerPolicy,
        required_families: (triggerPolicy.required_families || []).filter((_family, currentIndex) => currentIndex !== index),
      }
      return next
    })
  }

  const openAddDisagreement = () => {
    openEditModal<FamilyDisagreementFormState>(
      'Add Signal Family Disagreement',
      { cheap: configuredFamilyOptions[0] || 'keyword', expensive: configuredFamilyOptions[1] || 'embedding' },
      disagreementFields,
      async (form) => {
        updateDraft((current) => {
          const next = current
          const triggerPolicy = next.trigger_policy || {}
          next.trigger_policy = {
            ...triggerPolicy,
            family_disagreements: [
              ...(triggerPolicy.family_disagreements || []),
              {
                cheap: form.cheap.trim(),
                expensive: form.expensive.trim(),
              },
            ],
          }
          return next
        })
      },
      'add',
    )
  }

  const openEditDisagreement = (row: FamilyDisagreementRow) => {
    const index = disagreementRows.findIndex((candidate) => candidate.id === row.id)
    if (index < 0) {
      return
    }
    openEditModal<FamilyDisagreementFormState>(
      `Edit Disagreement: ${row.cheap} vs ${row.expensive}`,
      { cheap: row.cheap, expensive: row.expensive },
      disagreementFields,
      async (form) => {
        updateDraft((current) => {
          const next = current
          const triggerPolicy = next.trigger_policy || {}
          const disagreements = [...(triggerPolicy.family_disagreements || [])]
          disagreements[index] = { cheap: form.cheap.trim(), expensive: form.expensive.trim() }
          next.trigger_policy = { ...triggerPolicy, family_disagreements: disagreements }
          return next
        })
      },
    )
  }

  const deleteDisagreement = (row: FamilyDisagreementRow) => {
    const index = disagreementRows.findIndex((candidate) => candidate.id === row.id)
    if (index < 0) {
      return
    }
    updateDraft((current) => {
      const next = current
      const triggerPolicy = next.trigger_policy || {}
      next.trigger_policy = {
        ...triggerPolicy,
        family_disagreements: (triggerPolicy.family_disagreements || []).filter((_family, currentIndex) => currentIndex !== index),
      }
      return next
    })
  }

  const openAddAllowedAction = () => {
    openEditModal<AllowedActionFormState>(
      'Add Allowed Refinement Action',
      { type: 'disable_compression', signal_families: [] },
      actionFields,
      async (form) => {
        updateDraft((current) => {
          const next = current
          next.allowed_actions = [
            ...(next.allowed_actions || []),
            {
              type: form.type,
              signal_families: form.type === 'rerun_signal_families' ? form.signal_families : undefined,
            },
          ]
          return next
        })
      },
      'add',
    )
  }

  const openEditAllowedAction = (row: AllowedActionRow) => {
    const index = allowedActionRows.findIndex((candidate) => candidate.id === row.id)
    if (index < 0) {
      return
    }
    openEditModal<AllowedActionFormState>(
      `Edit Action: ${row.type}`,
      {
        type: row.type,
        signal_families: row.signal_families || [],
      },
      actionFields,
      async (form) => {
        updateDraft((current) => {
          const next = current
          const actions = [...(next.allowed_actions || [])]
          actions[index] = {
            type: form.type,
            signal_families: form.type === 'rerun_signal_families' ? form.signal_families : undefined,
          }
          next.allowed_actions = actions
          return next
        })
      },
    )
  }

  const deleteAllowedAction = (row: AllowedActionRow) => {
    const index = allowedActionRows.findIndex((candidate) => candidate.id === row.id)
    if (index < 0) {
      return
    }
    updateDraft((current) => {
      const next = current
      next.allowed_actions = (next.allowed_actions || []).filter((_action, currentIndex) => currentIndex !== index)
      return next
    })
  }

  return (
    <ConfigPageManagerLayout
      eyebrow="Routing"
      title="Meta Routing"
      description="Author the request-phase assess-and-refine loop that can observe, shadow, or actively refine a routing decision after the base pass."
      configArea="Routing"
      scope="Assessments, triggers, pass budget, and bounded refinement actions"
      panelEyebrow="Runtime"
      panelTitle="Meta Routing Policy"
      panelDescription="Keep the signal, projection, and decision layers pure while configuring the orchestration seam that decides when a second pass is worthwhile."
      pills={[
        { label: enabled ? 'Enabled' : 'Disabled', active: enabled },
        { label: `Mode: ${draft.mode}` },
        { label: `${draft.trigger_policy?.required_families?.length || 0} required families` },
        { label: `${draft.allowed_actions?.length || 0} allowed actions` },
      ]}
    >
      <div className={metaStyles.sectionGrid}>
        <div className={metaStyles.summaryGrid}>
          <article className={metaStyles.summaryCard}>
            <span className={metaStyles.summaryLabel}>Rollout Mode</span>
            <strong className={`${metaStyles.summaryValue} ${metaStyles.highlightValue}`}>{enabled ? draft.mode : 'disabled'}</strong>
          </article>
          <article className={metaStyles.summaryCard}>
            <span className={metaStyles.summaryLabel}>Pass Budget</span>
            <strong className={metaStyles.summaryValue}>{draft.max_passes || 1}</strong>
          </article>
          <article className={metaStyles.summaryCard}>
            <span className={metaStyles.summaryLabel}>Trigger Checks</span>
            <strong className={metaStyles.summaryValue}>
              {(draft.trigger_policy?.required_families?.length || 0) + (draft.trigger_policy?.family_disagreements?.length || 0) + 3}
            </strong>
          </article>
          <article className={metaStyles.summaryCard}>
            <span className={metaStyles.summaryLabel}>Pending Changes</span>
            <strong className={`${metaStyles.summaryValue} ${isDirty ? metaStyles.highlightValue : ''}`}>{isDirty ? 'Unsaved' : 'Saved'}</strong>
          </article>
        </div>

        <section className={metaStyles.sectionCard}>
          <div className={metaStyles.toggleRow}>
            <div className={metaStyles.sectionCopy}>
              <h3 className={metaStyles.sectionTitle}>Rollout and Pass Budget</h3>
              <p className={metaStyles.sectionDescription}>
                `routing.meta` stays optional. When enabled, the router records a base pass, assesses whether refinement is needed, then optionally shadows or adopts a refined pass.
              </p>
            </div>
            <div className={metaStyles.statusRow}>
              <label className={metaStyles.toggle}>
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(event) => setEnabled(event.target.checked)}
                  disabled={isReadonly || saving}
                />
                <span>{enabled ? 'Meta routing enabled' : 'Meta routing disabled'}</span>
              </label>
              <span className={`${styles.statusBadge} ${enabled ? styles.statusActive : styles.statusInactive}`}>
                {enabled ? 'Active in config draft' : 'Disabled'}
              </span>
            </div>
          </div>

          <div className={metaStyles.fieldGrid}>
            <label className={metaStyles.field}>
              <span className={metaStyles.label}>Mode</span>
              <select
                className={metaStyles.select}
                value={draft.mode}
                disabled={!enabled || isReadonly || saving}
                onChange={(event) => updateDraft((current) => ({ ...current, mode: event.target.value as MetaRoutingConfig['mode'] }))}
              >
                {META_MODE_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <p className={metaStyles.helpText}>`observe` records only, `shadow` executes refinement but keeps the base decision, and `active` can adopt the refined outcome.</p>
            </label>

            <label className={metaStyles.field}>
              <span className={metaStyles.label}>Max Passes</span>
              <input
                className={metaStyles.input}
                type="number"
                min={1}
                max={4}
                step={1}
                value={draft.max_passes ?? 2}
                disabled={!enabled || isReadonly || saving}
                onChange={(event) => {
                  const nextValue = Number.parseInt(event.target.value, 10)
                  updateDraft((current) => ({ ...current, max_passes: Number.isFinite(nextValue) ? nextValue : 1 }))
                }}
              />
              <p className={metaStyles.helpText}>v1 defaults to one additional refinement pass. Keep this small to bound latency.</p>
            </label>
          </div>
        </section>

        {enabled ? (
          <section className={metaStyles.sectionCard}>
            <div className={metaStyles.sectionCopy}>
              <h3 className={metaStyles.sectionTitle}>Deterministic Trigger Policy</h3>
              <p className={metaStyles.sectionDescription}>
                These thresholds decide when the base pass looks fragile enough to justify a bounded refinement plan.
              </p>
            </div>

            <div className={metaStyles.fieldGrid}>
              <label className={metaStyles.field}>
                <span className={metaStyles.label}>Decision Margin Below</span>
                <input
                  className={metaStyles.input}
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={draft.trigger_policy?.decision_margin_below ?? ''}
                  disabled={isReadonly || saving}
                  onChange={(event) => {
                    const value = event.target.value
                    updateDraft((current) => ({
                      ...current,
                      trigger_policy: {
                        ...(current.trigger_policy || {}),
                        decision_margin_below: value === '' ? undefined : Number(value),
                      },
                    }))
                  }}
                  placeholder="0.18"
                />
                <p className={metaStyles.helpText}>Trigger when the winner and runner-up decisions are too close.</p>
              </label>

              <label className={metaStyles.field}>
                <span className={metaStyles.label}>Projection Boundary Within</span>
                <input
                  className={metaStyles.input}
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={draft.trigger_policy?.projection_boundary_within ?? ''}
                  disabled={isReadonly || saving}
                  onChange={(event) => {
                    const value = event.target.value
                    updateDraft((current) => ({
                      ...current,
                      trigger_policy: {
                        ...(current.trigger_policy || {}),
                        projection_boundary_within: value === '' ? undefined : Number(value),
                      },
                    }))
                  }}
                  placeholder="0.07"
                />
                <p className={metaStyles.helpText}>Trigger when a projection score lands too close to a named mapping boundary.</p>
              </label>

              <label className={metaStyles.field}>
                <span className={metaStyles.label}>Partition Conflict</span>
                <label className={metaStyles.toggle}>
                  <input
                    type="checkbox"
                    checked={draft.trigger_policy?.partition_conflict === true}
                    disabled={isReadonly || saving}
                    onChange={(event) =>
                      updateDraft((current) => ({
                        ...current,
                        trigger_policy: {
                          ...(current.trigger_policy || {}),
                          partition_conflict: event.target.checked ? true : undefined,
                        },
                      }))
                    }
                  />
                  <span>Refine when exclusive partitions disagree</span>
                </label>
                <p className={metaStyles.helpText}>Use this when partition-level evidence should not silently collapse into a single band.</p>
              </label>
            </div>
          </section>
        ) : (
          <section className={metaStyles.sectionCard}>
            <div className={metaStyles.emptyState}>
              Enable meta routing to author rollout mode, trigger policy, required families, disagreement checks, and allowed refinement actions.
            </div>
          </section>
        )}

        <section className={styles.sectionTableBlock}>
          <TableHeader
            title="Required Signal Families"
            icon="🧩"
            count={requiredFamilyRows.length}
            searchPlaceholder=""
            onAdd={enabled && !isReadonly ? openAddRequiredFamily : undefined}
            addButtonText="Add Required Family"
            disabled={!enabled || isReadonly || saving}
            variant="embedded"
          />
          <DataTable
            columns={requiredFamilyColumns}
            data={requiredFamilyRows}
            keyExtractor={(row) => row.id}
            onEdit={enabled && !isReadonly ? openEditRequiredFamily : undefined}
            onDelete={enabled && !isReadonly ? deleteRequiredFamily : undefined}
            emptyMessage="No required families configured."
            readonly={isReadonly || !enabled}
          />
        </section>

        <section className={styles.sectionTableBlock}>
          <TableHeader
            title="Family Disagreements"
            icon="⚖️"
            count={disagreementRows.length}
            searchPlaceholder=""
            onAdd={enabled && !isReadonly ? openAddDisagreement : undefined}
            addButtonText="Add Disagreement"
            disabled={!enabled || isReadonly || saving}
            variant="embedded"
          />
          <DataTable
            columns={disagreementColumns}
            data={disagreementRows}
            keyExtractor={(row) => row.id}
            onEdit={enabled && !isReadonly ? openEditDisagreement : undefined}
            onDelete={enabled && !isReadonly ? deleteDisagreement : undefined}
            emptyMessage="No cheap-vs-expensive family disagreement checks configured."
            readonly={isReadonly || !enabled}
          />
        </section>

        <section className={styles.sectionTableBlock}>
          <TableHeader
            title="Allowed Refinement Actions"
            icon="🔁"
            count={allowedActionRows.length}
            searchPlaceholder=""
            onAdd={enabled && !isReadonly ? openAddAllowedAction : undefined}
            addButtonText="Add Action"
            disabled={!enabled || isReadonly || saving}
            variant="embedded"
          />
          <DataTable
            columns={allowedActionColumns}
            data={allowedActionRows}
            keyExtractor={(row) => row.id}
            onEdit={enabled && !isReadonly ? openEditAllowedAction : undefined}
            onDelete={enabled && !isReadonly ? deleteAllowedAction : undefined}
            emptyMessage="No refinement actions configured."
            readonly={isReadonly || !enabled}
          />
        </section>

        <section className={metaStyles.sectionCard}>
          <div className={metaStyles.ctaRow}>
            <div className={metaStyles.sectionCopy}>
              <h3 className={metaStyles.sectionTitle}>Save Policy Changes</h3>
              <p className={metaStyles.sectionDescription}>
                Meta outputs remain orchestration-only in v1. Decisions still consume raw signals and projection outputs rather than these meta artifacts.
              </p>
            </div>
            <div className={metaStyles.ctaActions}>
              <button
                type="button"
                className={metaStyles.secondaryButton}
                disabled={!isDirty || saving}
                onClick={resetDraft}
              >
                Discard Changes
              </button>
              <button
                type="button"
                className={enabled ? metaStyles.primaryButton : metaStyles.dangerButton}
                disabled={isReadonly || !isDirty || saving}
                onClick={() => void persistDraft(enabled, draft)}
              >
                {saving ? 'Saving...' : enabled ? 'Save Meta Routing' : 'Disable Meta Routing'}
              </button>
            </div>
          </div>
        </section>
      </div>
    </ConfigPageManagerLayout>
  )
}
