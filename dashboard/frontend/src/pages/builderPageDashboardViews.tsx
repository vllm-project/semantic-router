import React from 'react'

import { formatRoutingMetadataValue } from '@/components/routingMetadataDisplay'
import { useDSLStore } from '@/stores/dslStore'

import styles from './BuilderPage.module.css'
import { PluginIcon, RouteIcon, SignalIcon } from './builderPageFormPrimitives'
import type { EntityKind, Selection } from './builderPageTypes'

interface SidebarSectionProps {
  title: string
  count: number
  open: boolean
  onToggle: () => void
  onAdd?: () => void
  children: React.ReactNode
}

const SidebarSection: React.FC<SidebarSectionProps> = ({
  title,
  count,
  open,
  onToggle,
  onAdd,
  children,
}) => (
  <div className={styles.sidebarSection}>
    <div className={styles.sidebarSectionHeader} onClick={onToggle}>
      <span className={styles.sidebarSectionTitle}>
        {title}
        <span className={styles.sidebarCount}>{count}</span>
      </span>
      <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
        {onAdd ? (
          <button
            className={styles.sidebarAddBtn}
            onClick={(event) => {
              event.stopPropagation()
              onAdd()
            }}
            title={`Add ${title.slice(0, -1)}`}
            style={{ width: 'auto', padding: '0.125rem 0.25rem' }}
          >
            +
          </button>
        ) : null}
        <svg
          className={`${styles.sidebarSectionChevron} ${open ? styles.sidebarSectionChevronOpen : ''}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          aria-hidden="true"
        >
          <path d="M6 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    </div>
    {open ? <ul className={styles.sidebarList}>{children}</ul> : null}
  </div>
)

interface EntityListViewProps {
  readOnly: boolean
  kind: EntityKind
  ast: ReturnType<typeof useDSLStore.getState>['ast']
  onSelect: (selection: Selection) => void
  onBack: () => void
  onAddEntity: (kind: EntityKind) => void
}

const ENTITY_METADATA: Partial<
  Record<EntityKind, { title: string; icon: React.FC<{ className?: string }>; color: string }>
> = {
  signal: { title: 'Signals', icon: SignalIcon, color: 'rgb(161, 161, 170)' },
  route: { title: 'Routes', icon: RouteIcon, color: 'rgb(244, 244, 245)' },
  plugin: { title: 'Plugins', icon: PluginIcon, color: 'rgb(113, 113, 122)' },
}

const EntityListView: React.FC<EntityListViewProps> = ({
  readOnly,
  kind,
  ast,
  onSelect,
  onBack,
  onAddEntity,
}) => {
  const meta = ENTITY_METADATA[kind]
  if (!meta) return null
  const Icon = meta.icon
  const items: { name: string; type: string; desc?: string }[] = (() => {
    switch (kind) {
      case 'signal':
        return (ast?.signals ?? []).map((signal) => ({
          name: signal.name,
          type: signal.signalType,
          desc:
            Object.keys(signal.fields).length > 0
              ? `${Object.keys(signal.fields).length} field(s)`
              : undefined,
        }))
      case 'route':
        return (ast?.routes ?? []).map((route) => ({
          name: route.name,
          type: route.when ? `P${route.priority}` : 'default',
          desc: route.description || undefined,
        }))
      case 'plugin':
        return (ast?.plugins ?? []).map((plugin) => ({
          name: plugin.name,
          type: plugin.pluginType,
          desc:
            Object.keys(plugin.fields).length > 0
              ? `${Object.keys(plugin.fields).length} field(s)`
              : undefined,
        }))
      default:
        return []
    }
  })()

  return (
    <div className={styles.entityListPanel}>
      <div className={styles.entityListHeader}>
        <button className={styles.backBtn} onClick={onBack} title="Back to Dashboard">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <Icon className={styles.statIcon} />
        <span className={styles.entityListTitle}>{meta.title}</span>
        <span className={styles.entityListCount}>{items.length}</span>
        {!readOnly ? (
          <div style={{ marginLeft: 'auto' }}>
            <button
              className={styles.quickActionBtn}
              onClick={() => onAddEntity(kind)}
              style={{ padding: '0.5rem 1rem', fontSize: '0.8125rem' }}
            >
              <span
                className={styles.quickActionIcon}
                style={{ width: 24, height: 24, fontSize: '0.875rem' }}
                aria-hidden="true"
              >
                +
              </span>
              New {meta.title.replace(/s$/, '')}
            </button>
          </div>
        ) : null}
      </div>
      <div className={styles.entityListGrid}>
        {items.map((item) => (
          <div
            key={item.name}
            className={styles.entityListCard}
            onClick={() => onSelect({ kind, name: item.name })}
            style={{ '--entity-accent': meta.color } as React.CSSProperties}
          >
            <div className={styles.entityListCardHeader}>
              <Icon className={styles.entityListCardIcon} />
              <span className={styles.entityListCardName}>
                {kind === 'route'
                  ? formatRoutingMetadataValue('x-vsr-selected-decision', item.name)
                  : kind === 'signal'
                    ? formatRoutingMetadataValue(
                        `x-vsr-matched-${item.type.replace('_', '-')}`,
                        item.name,
                      )
                    : item.name}
              </span>
            </div>
            <span className={styles.entityListCardType}>{item.type}</span>
            {item.desc ? <span className={styles.entityListCardDesc}>{item.desc}</span> : null}
            <div className={styles.entityListCardArrow}>
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <polyline points="9 6 15 12 9 18" />
              </svg>
            </div>
          </div>
        ))}
      </div>
      {items.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            <Icon className={styles.statIcon} />
          </div>
          <div>No {meta.title.toLowerCase()} defined yet</div>
          <div
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--color-text-muted)',
            }}
          >
            {readOnly
              ? 'Nothing is configured in this section'
              : 'Click the button above to create one'}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export { default as DashboardView } from './builderPageDashboardView'
export { EntityListView, SidebarSection }
