import { useCallback, useEffect, useMemo, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import styles from './ConfigPageRouterConfigSection.module.css'
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

  const handleEditSection = (card: (typeof sectionCards)[number]) => {
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
      description="Shape how the router runs. Start with a layer, then tune only what you need."
    >
      <div className={styles.globalWorkspace}>
        <header className={styles.workspaceHeader}>
          <div className={styles.workspaceHeading}>
            <span className={styles.workspaceEyebrow}>Runtime layers</span>
            <h2>One place for router-wide behavior</h2>
            <p>
              {routerDefaults
                ? 'Effective values are live. Edit a section to override only what should change.'
                : 'Router defaults are offline. Saved overrides remain available to inspect and edit.'}
            </p>
          </div>
          <div className={styles.workspaceControls}>
            <div
              className={styles.modeToggle}
              role="tablist"
              aria-label="Global config editor mode"
            >
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
            <div className={styles.workspaceStatus}>
              <span>{configuredCount} configured</span>
              <span aria-hidden="true">·</span>
              <span>{sectionCards.length} sections</span>
            </div>
          </div>
        </header>

        {editorMode === 'visual' ? (
          <div className={styles.visualWorkspace}>
            <nav className={styles.layerNav} aria-label="Global config layers">
              <span className={styles.layerNavLabel}>Layers</span>
              {sectionGroups.map((group) => (
                <button
                  key={group.layer}
                  type="button"
                  className={styles.layerNavItem}
                  onClick={() =>
                    document
                      .getElementById(`global-layer-${group.layer}`)
                      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  }
                >
                  <span>{group.meta.title}</span>
                  <span>{group.cards.length}</span>
                </button>
              ))}
            </nav>

            <div className={styles.layerStack}>
              {sectionGroups.map((group, groupIndex) => (
                <section
                  key={group.layer}
                  id={`global-layer-${group.layer}`}
                  className={styles.sectionGroup}
                >
                  <header className={styles.groupHeader}>
                    <span className={styles.groupIndex}>
                      {String(groupIndex + 1).padStart(2, '0')}
                    </span>
                    <div>
                      <h3 className={styles.groupTitle}>{group.meta.title}</h3>
                      <p className={styles.groupDescription}>{group.meta.description}</p>
                    </div>
                  </header>
                  <div className={styles.sectionList}>
                    {group.cards.map((card) => (
                      <article key={card.key} className={styles.systemRow}>
                        <div className={styles.rowMain}>
                          <div className={styles.cardCopy}>
                            <div className={styles.cardTitleRow}>
                              <h4 className={styles.cardTitle}>{card.title}</h4>
                              <span className={`${styles.badge} ${badgeClassName(card.status)}`}>
                                {card.status.label}
                              </span>
                            </div>
                            <p className={styles.cardDescription}>{card.description}</p>
                          </div>
                          <div className={styles.rowActions}>
                            <span
                              className={`${styles.sourceBadge} ${badgeClassName({ label: card.sourceLabel, tone: card.sourceTone })}`}
                            >
                              {card.sourceLabel}
                            </span>
                            {!isReadonly ? (
                              <button
                                type="button"
                                className={styles.editButton}
                                onClick={() => handleEditSection(card)}
                              >
                                <ProductIcon name="edit" width={14} height={14} />
                                {card.data !== undefined ? 'Edit' : 'Configure'}
                              </button>
                            ) : null}
                          </div>
                        </div>

                        <div className={styles.summaryGrid}>
                          {card.summary.map((item) => (
                            <div key={`${card.key}-${item.label}`} className={styles.summaryItem}>
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
                              <span
                                key={`${card.key}-${badge.label}`}
                                className={`${styles.badge} ${badgeClassName(badge)}`}
                              >
                                {badge.label}
                              </span>
                            ))}
                          </div>
                        )}
                        <code className={styles.sectionKey}>{`global.${card.path.join('.')}`}</code>
                      </article>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </div>
        ) : (
          <section className={styles.rawWorkspace}>
            <div className={styles.blockHeader}>
              <div>
                <h2 className={styles.blockTitle}>Raw Global YAML</h2>
                <p className={styles.blockDescription}>
                  Edit the complete effective <code>global</code> block. Saved values override
                  router defaults.
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
                  Empty content removes the `global:` override block. Router defaults remain
                  available at runtime.
                </span>
                <span className={styles.rawDirtyState}>
                  {rawLoading ? 'Loading…' : rawDirty ? 'Unsaved changes' : 'Saved'}
                </span>
              </div>
            </div>
          </section>
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
