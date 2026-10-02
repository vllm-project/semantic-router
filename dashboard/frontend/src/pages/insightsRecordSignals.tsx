import { formatRoutingMetadataValue } from '../components/routingMetadataDisplay'
import type { ViewField } from '../components/ViewPanel'
import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import type { Signal } from './insightsPageTypes'
import styles from './InsightsPage.module.css'

const signalFamilies = new Map<string, { label: string; order: number }>(
  ROUTER_CONFIG_EXTENSION.signals.map(({ type, display_name }, order) => [
    type,
    { label: display_name, order },
  ]),
)

function recordedSignalGroups(signals: Signal) {
  return Object.entries(signals ?? {})
    .flatMap(([type, recorded]) => {
      if (!Array.isArray(recorded)) return []
      const values = recorded.filter((value): value is string => typeof value === 'string')
      if (!values.length) return []
      return [{ type, values }]
    })
    .sort((left, right) => {
      const leftOrder = signalFamilies.get(left.type)?.order ?? Number.MAX_SAFE_INTEGER
      const rightOrder = signalFamilies.get(right.type)?.order ?? Number.MAX_SAFE_INTEGER
      return leftOrder - rightOrder || left.type.localeCompare(right.type)
    })
}

function formatSignalMatch(type: string, value: string): string {
  return formatRoutingMetadataValue(`x-vsr-matched-${type.replace(/_/g, '-')}`, value)
}

export function collectSignals(signals: Signal): string[] {
  return recordedSignalGroups(signals).flatMap(({ type, values }) =>
    values.map((value) => formatSignalMatch(type, value)),
  )
}

export function buildSignalFields(signals: Signal): ViewField[] {
  const groups = recordedSignalGroups(signals)
  return [
    {
      label: 'Recorded matches',
      fullWidth: true,
      value: (
        <div className={styles.pluginStack}>
          <p className={styles.costSubtle}>
            Recorded matches only. Measurements are available in Routing Metadata; unmatched or
            unrecorded rules are not listed.
          </p>
          {!groups.length ? <p>No signal matches were recorded for this request.</p> : null}
        </div>
      ),
    },
    ...groups.map(({ type, values }) => ({
      label: `${signalFamilies.get(type)?.label ?? formatSignalMatch('family', type)} signals`,
      fullWidth: true,
      value: (
        <div className={styles.modalSignalList}>
          {values.map((value, index) => (
            <span key={`${type}-${index}`} className={styles.modalSignalPill} title={value}>
              {formatSignalMatch(type, value)}
            </span>
          ))}
        </div>
      ),
    })),
  ]
}
