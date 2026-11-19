import { formatRoutingMetadataValue, matchedSignalKey } from '../components/routingMetadataDisplay'
import type { ViewField } from '../components/ViewPanel'
import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import { Badge } from '../ui'
import type { Signal } from './insightsPageTypes'

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

/** A signal family as people read it, such as Domain for domain. */
export function signalFamilyLabel(type: string): string {
  return signalFamilies.get(type)?.label ?? formatSignalMatch('family', type)
}

function formatSignalMatch(type: string, value: string): string {
  return formatRoutingMetadataValue(matchedSignalKey(type), value)
}

// Each recorded match with its family and the words it shows as.
export function recordedSignalMatches(signals: Signal) {
  return recordedSignalGroups(signals).flatMap(({ type, values }) =>
    values.map((value) => ({ type, value, text: formatSignalMatch(type, value) })),
  )
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
        <div className="flex flex-col gap-1 text-[13px]">
          <p className="text-xs text-subtle-foreground">
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
        <div className="flex flex-wrap gap-1.5">
          {values.map((value, index) => (
            <Badge key={`${type}-${index}`} tone="gold" title={value}>
              {formatSignalMatch(type, value)}
            </Badge>
          ))}
        </div>
      ),
    })),
  ]
}
