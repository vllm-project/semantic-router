import { useEffect, useMemo, useState } from 'react'
import styles from './ConfigPageRouterConfigSection.module.css'
import pageStyles from './ConfigPage.module.css'
import ConfigPageLegacyCategoriesSection from './ConfigPageLegacyCategoriesSection'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import {
  buildEffectiveRouterConfig,
  buildRouterSectionCards,
  type RouterConfigSectionData,
  type RouterSectionBadge,
} from './configPageRouterDefaultsSupport'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import type { ConfigData, Tool } from './configPageSupport'

interface ConfigPageRouterConfigSectionProps {
  config: ConfigData | null
  routerConfig: RouterConfigSectionData
  toolsData: Tool[]
  toolsLoading: boolean
  toolsError: string | null
  isReadonly: boolean
  openEditModal: OpenEditModal
  saveConfig: (config: ConfigData) => Promise<void>
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
  routerConfig,
  toolsData,
  toolsLoading,
  toolsError,
  isReadonly,
  openEditModal,
  saveConfig,
  showLegacyCategories = false,
}: ConfigPageRouterConfigSectionProps) {
  const [routerDefaults, setRouterDefaults] = useState<ConfigData | null>(null)

  useEffect(() => {
    let cancelled = false

    const fetchRouterDefaults = async () => {
      try {
        const response = await fetch('/api/router/config/defaults')
        if (!response.ok) {
          if (!cancelled) {
            setRouterDefaults(null)
          }
          return
        }

        const data = await response.json()
        if (!cancelled) {
          setRouterDefaults(data)
        }
      } catch {
        if (!cancelled) {
          setRouterDefaults(null)
        }
      }
    }

    fetchRouterDefaults()

    return () => {
      cancelled = true
    }
  }, [])

  const effectiveRouterConfig = useMemo(() => {
    const configWithParentFallback = {
      ...(config || {}),
      ...(routerConfig || {}),
    } as ConfigData

    return buildEffectiveRouterConfig(routerDefaults, configWithParentFallback)
  }, [config, routerConfig, routerDefaults])

  const sectionCards = buildRouterSectionCards({
    config,
    routerConfig: effectiveRouterConfig,
    routerDefaults,
    toolsData,
    toolsLoading,
    toolsError,
  })

  const configuredCount = sectionCards.filter((card) => card.data !== undefined).length
  const routerDefaultsCount = sectionCards.filter((card) => card.sourceLabel === '.vllm-sr/router-defaults.yaml').length
  const missingCount = sectionCards.length - configuredCount

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

    const response = await fetch('/api/router/config/defaults/update', {
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

    const refreshed = await fetch('/api/router/config/defaults')
    if (refreshed.ok) {
      setRouterDefaults(await refreshed.json())
    }
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

  return (
    <ConfigPageManagerLayout
      eyebrow="Runtime"
      title="Router Config"
      description="System-level defaults come from .vllm-sr/router-defaults.yaml for Python CLI setups. This surface keeps dashboard runtime controls aligned with the CLI template while preserving config.yaml fallbacks when older deployments still inline these settings."
    >
      <div className={pageStyles.sectionPanel}>
        <div className={pageStyles.sectionTableBlock}>
          <div className={styles.blockHeader}>
            <div>
              <h2 className={styles.blockTitle}>System Defaults Overview</h2>
              <p className={styles.blockDescription}>
                {routerDefaults
                  ? 'Local router-defaults.yaml is loaded. Edits in this section write back to the system defaults file.'
                  : 'No local router-defaults.yaml was loaded. Cards below still follow the Python CLI template order and show config.yaml fallback values when present.'}
              </p>
            </div>
          </div>

          <div className={styles.overviewGrid}>
            <div className={styles.overviewCard}>
              <span className={styles.overviewLabel}>Template Sections</span>
              <strong className={styles.overviewValue}>{sectionCards.length}</strong>
              <span className={styles.overviewHint}>Top-level router-defaults surfaces tracked by the dashboard.</span>
            </div>
            <div className={styles.overviewCard}>
              <span className={styles.overviewLabel}>Loaded From Defaults</span>
              <strong className={styles.overviewValue}>{routerDefaultsCount}</strong>
              <span className={styles.overviewHint}>Sections sourced directly from `.vllm-sr/router-defaults.yaml`.</span>
            </div>
            <div className={styles.overviewCard}>
              <span className={styles.overviewLabel}>Missing Or Template-Only</span>
              <strong className={styles.overviewValue}>{missingCount}</strong>
              <span className={styles.overviewHint}>Template sections that still need local values if you want them persisted.</span>
            </div>
          </div>
        </div>

        <div className={pageStyles.sectionTableBlock}>
          <div className={styles.blockHeader}>
            <div>
              <h2 className={styles.blockTitle}>Runtime System Sections</h2>
              <p className={styles.blockDescription}>
                Cards mirror the Python CLI router-defaults template instead of the older flat config subset. Each editor writes an entire top-level section so nested runtime defaults stay intact.
              </p>
            </div>
          </div>

          <div className={styles.sectionGrid}>
            {sectionCards.map((card) => (
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
                      <span className={styles.summaryValue}>{item.value}</span>
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
                  <code className={styles.sectionKey}>{card.key}</code>
                  {!isReadonly ? (
                    <button
                      className={pageStyles.sectionEditButton}
                      onClick={() => {
                        handleEditSection(card)
                      }}
                    >
                      {card.data !== undefined ? 'Edit Section' : 'Add Section'}
                    </button>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        </div>

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
