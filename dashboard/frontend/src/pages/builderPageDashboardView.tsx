import React from 'react'

import { formatRoutingMetadataValue } from '@/components/routingMetadataDisplay'
import { useDSLStore } from '@/stores/dslStore'
import type { BoolExprNode, EditorMode } from '@/types/dsl'

import styles from './BuilderPage.module.css'
import { ModelIcon, PluginIcon, RouteIcon, SignalIcon } from './builderPageFormPrimitives'
import type { EntityKind, Selection } from './builderPageTypes'

interface DashboardViewProps {
  readOnly: boolean
  ast: ReturnType<typeof useDSLStore.getState>['ast']
  modelCount: number
  signalCount: number
  routeCount: number
  pluginCount: number
  isValid: boolean
  errorCount: number
  onSelect: (selection: Selection) => void
  onAddEntity: (kind: EntityKind) => void
  onModeSwitch: (mode: EditorMode) => void
}

interface EditorModeOption {
  mode: EditorMode
  label: string
  description: string
  icon: React.ReactNode
}

const EDITOR_MODES: EditorModeOption[] = [
  {
    mode: 'visual',
    label: 'Visual',
    description: 'Build with guided forms',
    icon: (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <rect x="2.5" y="3" width="5" height="5" rx="1" />
        <rect x="12.5" y="12" width="5" height="5" rx="1" />
        <path d="M7.5 5.5h3a2 2 0 0 1 2 2v4.5" />
      </svg>
    ),
  },
  {
    mode: 'dsl',
    label: 'DSL',
    description: 'Edit the source directly',
    icon: (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="m7.5 5-4 5 4 5M12.5 5l4 5-4 5M11 3 9 17" />
      </svg>
    ),
  },
  {
    mode: 'nl',
    label: 'Natural Language',
    description: 'Describe the change you want',
    icon: (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M4 3.5h12a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H9l-4.5 3v-3H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2Z" />
        <path d="M6 7h8M6 10.5h5" />
      </svg>
    ),
  },
]

const boolExprToText = (node: BoolExprNode | null, maxLength = 60): string => {
  if (!node) return '(always)'

  const serialize = (expression: BoolExprNode): string => {
    switch (expression.type) {
      case 'signal_ref':
        return `${expression.signalType}("${expression.signalName}")`
      case 'not':
        return `NOT ${serialize(expression.expr)}`
      case 'and':
        return `${serialize(expression.left)} AND ${serialize(expression.right)}`
      case 'or':
        return `(${serialize(expression.left)} OR ${serialize(expression.right)})`
    }
  }

  const text = serialize(node)
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text
}

const DashboardView: React.FC<DashboardViewProps> = ({
  readOnly,
  ast,
  modelCount,
  signalCount,
  routeCount,
  pluginCount,
  isValid,
  errorCount,
  onSelect,
  onAddEntity,
  onModeSwitch,
}) => {
  const routes = ast?.routes ?? []
  const defaultRoute = routes.find((route) => !route.when)
  const conditionalRoutes = routes
    .filter((route) => Boolean(route.when))
    .sort((left, right) => right.priority - left.priority)
  const stats = [
    {
      label: 'Models',
      count: modelCount,
      kind: 'model' as const,
      icon: <ModelIcon className={styles.statIcon} />,
    },
    {
      label: 'Signals',
      count: signalCount,
      kind: 'signal' as const,
      icon: <SignalIcon className={styles.statIcon} />,
    },
    {
      label: 'Routes',
      count: routeCount,
      kind: 'route' as const,
      icon: <RouteIcon className={styles.statIcon} />,
    },
    {
      label: 'Plugins',
      count: pluginCount,
      kind: 'plugin' as const,
      icon: <PluginIcon className={styles.statIcon} />,
    },
  ]

  return (
    <div className={styles.dashboard}>
      <div className={styles.dashboardHeader}>
        <div className={styles.dashboardHeading}>
          <span className={styles.dashboardEyebrow}>Visual Builder</span>
          <h1 className={styles.dashboardTitle}>Routing workspace</h1>
          <p>Shape the path from request to model.</p>
        </div>
        <div
          className={`${styles.dashboardBadge} ${isValid ? styles.dashboardBadgeOk : styles.dashboardBadgeErr}`}
        >
          <span aria-hidden="true" />
          {isValid ? 'Valid' : `${errorCount} error${errorCount !== 1 ? 's' : ''}`}
        </div>
      </div>

      <div className={styles.statsGrid}>
        {stats.map((card) => (
          <button
            type="button"
            key={card.label}
            className={styles.statCard}
            disabled={card.count === 0}
            onClick={() => onSelect({ kind: card.kind, name: '__list__' })}
          >
            {card.icon}
            <span className={styles.statValue}>{card.count}</span>
            <span className={styles.statLabel}>{card.label}</span>
            <span
              className={`${styles.statBadge} ${card.count > 0 ? styles.statBadgeOk : styles.statBadgeEmpty}`}
            >
              {card.count > 0 ? 'Ready' : 'Not set'}
            </span>
          </button>
        ))}
      </div>

      <div className={styles.dashboardWorkspace}>
        <section className={`${styles.dashSection} ${styles.routeSection}`}>
          <div className={styles.dashSectionHeading}>
            <div>
              <span className={styles.dashSectionTitle}>Route Map</span>
              <p>How each request reaches its model pool.</p>
            </div>
            <span className={styles.routeCount}>{routes.length}</span>
          </div>
          <div className={styles.routeMap}>
            <div className={styles.routeMapEntry}>
              <span className={styles.routeMapEntryLabel}>Request</span>
            </div>
            <div className={styles.routeMapFlow}>
              {conditionalRoutes.map((route) => (
                <div
                  key={route.name}
                  className={styles.routeMapBranch}
                  onClick={() => onSelect({ kind: 'route', name: route.name })}
                >
                  <div className={styles.routeMapCondition}>
                    <span className={styles.routeMapCondIcon}>├─</span>
                    <code className={styles.routeMapCondText}>{boolExprToText(route.when)}</code>
                  </div>
                  <div className={styles.routeMapTarget}>
                    <span className={styles.routeMapTargetArrow}>└→</span>
                    <span className={styles.routeMapRouteName}>
                      &quot;
                      {formatRoutingMetadataValue('x-vsr-selected-decision', route.name)}
                      &quot;
                    </span>
                    <span className={styles.routeMapTargetArrow}>→</span>
                    <span className={styles.routeMapModel}>
                      {route.models.length > 0
                        ? route.models.map((model) => model.model).join(', ')
                        : '(no model)'}
                    </span>
                  </div>
                </div>
              ))}
              {defaultRoute ? (
                <div
                  className={styles.routeMapBranch}
                  onClick={() => onSelect({ kind: 'route', name: defaultRoute.name })}
                >
                  <div className={styles.routeMapCondition}>
                    <span className={styles.routeMapCondIcon}>└─</span>
                    <code className={styles.routeMapCondText}>Default path</code>
                  </div>
                  <div className={styles.routeMapTarget}>
                    <span className={styles.routeMapTargetArrow}>└→</span>
                    <span className={styles.routeMapRouteName}>
                      &quot;
                      {formatRoutingMetadataValue('x-vsr-selected-decision', defaultRoute.name)}
                      &quot;
                    </span>
                    <span className={styles.routeMapTargetArrow}>→</span>
                    <span className={styles.routeMapModel}>
                      {defaultRoute.models.length > 0
                        ? defaultRoute.models.map((model) => model.model).join(', ')
                        : '(no model)'}
                    </span>
                  </div>
                </div>
              ) : null}
              {routes.length === 0 ? (
                <div className={styles.routeMapEmpty}>
                  <strong>No routes yet</strong>
                  <span>Add a route to connect requests to models.</span>
                </div>
              ) : null}
            </div>
          </div>
        </section>

        <aside className={styles.dashboardRail}>
          {!readOnly ? (
            <section className={styles.dashSection}>
              <div className={styles.dashSectionHeading}>
                <div>
                  <span className={styles.dashSectionTitle}>Add to workspace</span>
                  <p>Extend this configuration.</p>
                </div>
              </div>
              <div className={styles.quickActions}>
                {(['model', 'signal', 'route', 'plugin'] as EntityKind[]).map((kind) => (
                  <button
                    key={kind}
                    type="button"
                    className={styles.quickActionBtn}
                    onClick={() => onAddEntity(kind)}
                  >
                    <span className={styles.quickActionIcon} aria-hidden="true">
                      +
                    </span>
                    <span>Add {kind}</span>
                  </button>
                ))}
              </div>
            </section>
          ) : null}

          <section className={styles.dashSection}>
            <div className={styles.dashSectionHeading}>
              <div>
                <span className={styles.dashSectionTitle}>Editor</span>
                <p>Choose how you want to work.</p>
              </div>
            </div>
            <div className={styles.dashModes}>
              {EDITOR_MODES.filter((option) => !readOnly || option.mode !== 'nl').map((mode) => (
                <button
                  key={mode.mode}
                  type="button"
                  className={styles.dashModeBtn}
                  onClick={() => onModeSwitch(mode.mode)}
                >
                  <span className={styles.dashModeBtnIcon}>{mode.icon}</span>
                  <span className={styles.dashModeBtnCopy}>
                    <span className={styles.dashModeBtnLabel}>{mode.label}</span>
                    <span className={styles.dashModeBtnDesc}>{mode.description}</span>
                  </span>
                  <svg className={styles.dashModeBtnArrow} viewBox="0 0 16 16" aria-hidden="true">
                    <path d="m6 3 5 5-5 5" />
                  </svg>
                </button>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  )
}

export default DashboardView
