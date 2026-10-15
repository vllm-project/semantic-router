import type { Observation } from '../utils/dashboardObservation'
import styles from './DecisionModelPage.module.css'

type ObservationStatus = Pick<
  Observation<unknown>,
  'loading' | 'refreshing' | 'stale' | 'error' | 'updatedAt'
>

export default function DecisionObservationStatus({
  label,
  observation,
}: {
  label: string
  observation: ObservationStatus
}) {
  const { loading, refreshing, stale, error, updatedAt } = observation
  return (
    <p className={styles.muted} role="status">
      {loading
        ? `Loading ${label.toLowerCase()}…`
        : updatedAt
          ? `${label} ${stale ? 'last observed' : 'updated'} ${new Date(updatedAt).toLocaleTimeString()}${stale ? ' · awaiting confirmation' : ''}${refreshing ? ' · refreshing…' : ''}`
          : `${label} unavailable.`}
      {error && ` ${error}`}
    </p>
  )
}
