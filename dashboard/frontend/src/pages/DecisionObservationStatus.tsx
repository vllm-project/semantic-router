import type { Observation } from '../utils/dashboardObservation'
import { StatusDot, cn } from '../ui'

type ObservationStatus = Pick<
  Observation<unknown>,
  'loading' | 'refreshing' | 'stale' | 'error' | 'updatedAt'
>

/** One observation's freshness; quiet while it is current, so only problems take space. */
export default function DecisionObservationStatus({
  label,
  observation,
  always = false,
  className,
}: {
  label: string
  observation: ObservationStatus
  always?: boolean
  className?: string
}) {
  const { loading, refreshing, stale, error, updatedAt } = observation
  const problem = Boolean(error) || stale || (!loading && !updatedAt)
  if (!always && (loading || !problem)) return null
  return (
    <p
      role="status"
      className={cn('flex items-start gap-2 text-xs leading-5 text-muted-foreground', className)}
    >
      <StatusDot tone={error ? 'warning' : stale ? 'info' : 'neutral'} className="mt-1.5" />
      <span className="min-w-0">
        {loading
          ? `Loading ${label.toLowerCase()}…`
          : updatedAt
            ? `${label} ${stale ? 'last observed' : 'updated'} ${new Date(updatedAt).toLocaleTimeString()}${stale ? ' · awaiting confirmation' : ''}${refreshing ? ' · refreshing…' : ''}`
            : `${label} unavailable.`}
        {error && ` ${error}`}
      </span>
    </p>
  )
}
