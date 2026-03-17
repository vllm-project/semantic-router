import { useCallback, useEffect, useMemo, useState } from 'react'
import styles from './ConfigPageRouterConfigSection.module.css'
import pageStyles from './ConfigPage.module.css'
import ConfigPageLegacyCategoriesSection from './ConfigPageLegacyCategoriesSection'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import {
  buildEffectiveRouterConfig,
  buildRouterSectionCards,
  ROUTER_LAYER_META,
  type RouterSectionBadge,
} from './configPageRouterDefaultsSupport'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import type { CanonicalGlobalConfig, ConfigData, Tool } from './configPageSupport'

type GlobalEditorMode = 'visual' | 'raw'

interface ConfigPageRouterConfigSectionProps {
  config: ConfigData | null
  toolsData: Tool[]
  toolsLoading: boolean
  toolsError: string | null
  isReadonly: boolean
  openEditModal: OpenEditModal
  saveConfig: (config: ConfigData) => Promise<void>
  refreshConfig: () => Promise<void>
  showLegacyCategories?: boolean
}

function badgeClassName(badge: RouterSectionBadge): string {
  switch (badge.tone) {
    case 'active':
      return styles.badgeActive
    case 'inactive':
      return styles.badgeInactive
    default:
      return styles.badgeInfo
  }
}

export default function ConfigPageRouterConfigSection({
  config,
  toolsData,
  toolsLoading,
  toolsError,
  isReadonly,
  openEditModal,
  saveConfig,
  refreshConfig,
  showLegacyCategories = false,
}: ConfigPageRouterConfigSectionProps) {
  const [routerDefaults, setRouterDefaults] = useState<CanonicalGlobalConfig | null>(null)
  const [editorMode, setEditorMode] = useState<GlobalEditorMode>('visual')
  const [rawYaml, setRawYaml] = useState('{}\n')
  const [rawLoading, setRawLoading] = useState(false)
  const [rawSaving, setRawSaving] = useState(false)
  const [rawError, setRawError] = useState<string | null>(null)
  const [rawDirty, setRawDirty] = useState(false)

  const loadRawGlobalConfig = useCallback(async () => {
    setRawLoading(true)
    setRawError(null)
    try {
      const response = await fetch('/api/router/config/global/raw')
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }

      const text = await response.text()
      setRawYaml(text || '{}\n')
      setRawDirty(false)
    } catch (err) {
      setRawError(err instanceof Error ? err.message : 'Failed to load raw global config')
    } finally {
      setRawLoading(false)
    }
  }, [])

  const loadRouterDefaults = useCallback(async () => {
    const response = await fetch('/api/router/config/global')
    if (!response.ok) {
      setRouterDefaults(null)
      return
    }
    setRouterDefaults(await response.json())
  }, [])

  useEffect(() => {
    let cancelled = false

    const fetchGlobalState = async () => {
      try {
        if (!cancelled) {
          await loadRouterDefaults()
        }
      } catch {
        if (!cancelled) {
          setRouterDefaults(null)
        }
      }

      if (!cancelled) {
        await loadRawGlobalConfig()
      }
    }

    void fetchGlobalState()

    return () => {
      cancelled = true
    }
  }, [loadRawGlobalConfig, loadRouterDefaults])

  const effectiveRouterConfig = useMemo(() => {
    return buildEffectiveRouterConfig(routerDefaults, config)
  }, [config, routerDefaults])

  const sectionCards = buildRouterSectionCards({
    config,
    routerConfig: effectiveRouterConfig,
    routerDefaults,
    toolsData,
    toolsLoading,
    toolsError,
  })

  const configuredCount = sectionCards.filter((card) => card.data !== undefined).length
  const routerDefaultsCount = sectionCards.filter((card) => card.sourceLabel === 'router effective defaults').length
  const missingCount = sectionCards.length - configuredCount
  const sectionGroups = useMemo(() => {
    const groups = new Map<string, typeof sectionCards>()
    for (const card of sectionCards) {
      const existing = groups.get(card.layer) || []
      existing.push(card)
      groups.set(card.layer, existing)
    }
    return Array.from(groups.entries()).map(([layer, cards]) => ({
      layer,
      meta: ROUTER_LAYER_META[layer as keyof typeof ROUTER_LAYER_META],
      cards,
    }))
  }, [sectionCards])

  const saveRouterSettings = async (updates: Partial<ConfigData>) => {
    if (showLegacyCategories) {
      if (!config) {
        throw new Error('Configuration not loaded yet.')
      }

      await saveConfig({
        ...config,
        ...updates,
      })
      return
    }

    const response = await fetch('/api/router/config/global/update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || `HTTP ${response.status}: ${response.statusText}`)
    }

    await refreshConfig()
    await loadRouterDefaults()
    await loadRawGlobalConfig()
  }

  const handleEditSection = (card: typeof sectionCards[number]) => {
    const isConfigured = card.data !== undefined

    openEditModal(
      `${isConfigured ? 'Edit' : 'Add'} ${card.title}`,
      card.editData,
      card.editFields,
      async (data) => {
        await saveRouterSettings(card.save(data))
      },
      isConfigured ? 'edit' : 'add',
    )
  }

  const handleSaveRawYaml = async () => {
    setRawSaving(true)
    setRawError(null)
    try {
      const response = await fetch('/api/router/config/global/raw/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'text/yaml; charset=utf-8',
        },
        body: rawYaml,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }

      await refreshConfig()
      await loadRouterDefaults()
      await loadRawGlobalConfig()
    } catch (err) {
      setRawError(err instanceof Error ? err.message : 'Failed to save raw global config')
    } finally {
      setRawSaving(false)
    }
  }

  return (
    <ConfigPageManagerLayout
      eyebrow="Runtime"
      title="Global Config"
      description="Router-owned runtime defaults are merged with your `config.yaml` `global` override. This surface edits the canonical `global` block while preserving the router's built-in defaults."
      configArea="Global"
      scope="Router runtime defaults and overrides"
      panelEyebrow="Runtime"
      panelTitle="Canonical Global Config"
      panelDescription="Review the resolved router defaults, edit layered `global` overrides, and keep runtime modules aligned with the canonical v0.3 structure."
      pills={sectionGroups.map((group, index) => ({
        label: group.meta?.title || group.layer,
        active: index === 0,
      }))}
    >
      <div className={pageStyles.sectionPanel}>
        <div className={pageStyles.sectionTableBlock}>
          <div className={styles.blockHeader}>
            <div>
              <h2 className={styles.blockTitle}>Global Config Overview</h2>
              <p className={styles.blockDescription}>
                {routerDefaults
                  ? 'The router resolved effective defaults successfully. Edits in this section write back to `config.yaml` under `global:`.'
                  : 'Effective router defaults are unavailable right now. Cards below still show the canonical `global` sections and any loaded config.yaml overrides.'}
              </p>
            </div>
            <div className={styles.modeToggle} role="tablist" aria-label="Global config editor mode">
              <button
                type="button"
                className={`${styles.modeButton} ${editorMode === 'visual' ? styles.modeButtonActive : ''}`}
                onClick={() => setEditorMode('visual')}
              >
                Visual
              </button>
              <button
                type="button"
                className={`${styles.modeButton} ${editorMode === 'raw' ? styles.modeButtonActive : ''}`}
                onClick={() => setEditorMode('raw')}
              >
                Raw YAML
              </button>
            </div>
          </div>

          <div className={styles.overviewGrid}>
            <div className={styles.overviewCard}>
              <span className={styles.overviewLabel}>Global Sections</span>
              <strong className={styles.overviewValue}>{sectionCards.length}</strong>
              <span className={styles.overviewHint}>Canonical `global` sections tracked by the dashboard.</span>
            </div>
            <div className={styles.overviewCard}>
              <span className={styles.overviewLabel}>Resolved By Router</span>
              <strong className={styles.overviewValue}>{routerDefaultsCount}</strong>
              <span className={styles.overviewHint}>Sections currently backed by router-owned defaults or effective merged values.</span>
            </div>
            <div className={styles.overviewCard}>
              <span className={styles.overviewLabel}>Missing Or Inactive</span>
              <strong className={styles.overviewValue}>{missingCount}</strong>
              <span className={styles.overviewHint}>Sections not currently present in the effective `global` surface.</span>
            </div>
          </div>
        </div>

        {editorMode === 'visual' ? (
          <div className={pageStyles.sectionTableBlock}>
            <div className={styles.blockHeader}>
              <div>
                <h2 className={styles.blockTitle}>Runtime Global Sections</h2>
                <p className={styles.blockDescription}>
                  Cards mirror the layered canonical `global` block. Each editor writes back to the matching `global.router`, `global.services`, `global.stores`, `global.integrations`, or `global.model_catalog` path.
                </p>
              </div>
            </div>

            {sectionGroups.map((group) => (
              <section key={group.layer} className={styles.sectionGroup}>
                <div className={styles.groupHeader}>
                  <div>
                    <h3 className={styles.groupTitle}>{group.meta.title}</h3>
                    <p className={styles.groupDescription}>{group.meta.description}</p>
                  </div>
                </div>
                <div className={styles.sectionGrid}>
                  {group.cards.map((card) => (
                    <article key={card.key} className={styles.systemCard}>
                      <div className={styles.cardHeader}>
                        <div className={styles.cardCopy}>
                          <span className={styles.cardEyebrow}>{card.eyebrow}</span>
                          <h3 className={styles.cardTitle}>{card.title}</h3>
                          <p className={styles.cardDescription}>{card.description}</p>
                        </div>
                        <div className={styles.cardBadges}>
                          <span className={`${styles.badge} ${badgeClassName({ label: card.sourceLabel, tone: card.sourceTone })}`}>
                            {card.sourceLabel}
                          </span>
                          <span className={`${styles.badge} ${badgeClassName(card.status)}`}>
                            {card.status.label}
                          </span>
                        </div>
                      </div>

                      <div className={styles.summaryList}>
                        {card.summary.map((item) => (
                          <div key={`${card.key}-${item.label}`} className={styles.summaryRow}>
                            <span className={styles.summaryLabel}>{item.label}</span>
                            <span className={styles.summaryValue} title={item.value}>
                              {item.value}
                            </span>
                          </div>
                        ))}
                      </div>

                      {card.badges.length > 0 && (
                        <div className={styles.tagRow}>
                          {card.badges.map((badge) => (
                            <span key={`${card.key}-${badge.label}`} className={`${styles.badge} ${badgeClassName(badge)}`}>
                              {badge.label}
                            </span>
                          ))}
                        </div>
                      )}

                      <div className={styles.cardFooter}>
                        <code className={styles.sectionKey}>{`global.${card.path.join('.')}`}</code>
                        {!isReadonly ? (
                          <div className={styles.cardActions}>
                            <button
                              className={pageStyles.sectionEditButton}
                              onClick={() => {
                                handleEditSection(card)
                              }}
                            >
                              {card.data !== undefined ? 'Edit Section' : 'Add Section'}
                            </button>
                          </div>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            ))}
          </div>
        ) : (
          <div className={pageStyles.sectionTableBlock}>
            <div className={styles.blockHeader}>
              <div>
                <h2 className={styles.blockTitle}>Raw Global YAML</h2>
                <p className={styles.blockDescription}>
                  This editor shows the effective merged `global` block: router defaults plus your current overrides. Saving raw YAML writes the full `global:` block back to `config.yaml`.
                </p>
              </div>
              <div className={styles.rawToolbar}>
                <button
                  type="button"
                  className={styles.rawToolbarButton}
                  onClick={() => {
                    void loadRawGlobalConfig()
                  }}
                  disabled={rawLoading || rawSaving}
                >
                  Reload
                </button>
                {!isReadonly ? (
                  <button
                    type="button"
                    className={`${styles.rawToolbarButton} ${styles.rawToolbarButtonPrimary}`}
                    onClick={() => {
                      void handleSaveRawYaml()
                    }}
                    disabled={rawLoading || rawSaving || !rawDirty}
                  >
                    {rawSaving ? 'Saving…' : 'Save YAML'}
                  </button>
                ) : null}
              </div>
            </div>

            <div className={styles.rawEditorPanel}>
              {rawError ? <div className={styles.rawError}>{rawError}</div> : null}
              <textarea
                className={styles.rawEditor}
                value={rawYaml}
                onChange={(event) => {
                  setRawYaml(event.target.value)
                  setRawDirty(true)
                }}
                spellCheck={false}
                readOnly={isReadonly || rawLoading || rawSaving}
              />
              <div className={styles.rawHintRow}>
                <span className={styles.rawHint}>
                  Empty content removes the `global:` override block. Router defaults remain available at runtime.
                </span>
                <span className={styles.rawDirtyState}>
                  {rawLoading ? 'Loading…' : rawDirty ? 'Unsaved changes' : 'Saved'}
                </span>
              </div>
            </div>
          </div>
        )}

        {showLegacyCategories ? (
          <ConfigPageLegacyCategoriesSection
            config={config}
            isReadonly={isReadonly}
            openEditModal={openEditModal}
            saveConfig={saveConfig}
          />
        ) : null}
      </div>
    </ConfigPageManagerLayout>
  )
}
