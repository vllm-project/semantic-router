import { benchmarkTitle, evaluationRoleTitle } from './datasetPresentation'
import { number } from './model'
import type { DatasetPreparation } from './types'
import styles from './SrBench.module.css'

export default function DatasetExposure({
  preparation,
  benchmarks,
  compact = false,
}: {
  preparation?: DatasetPreparation
  benchmarks?: string[]
  compact?: boolean
}) {
  if (!preparation || !Object.keys(preparation).length) return null
  const entries = [...new Set([...Object.keys(preparation), ...(benchmarks ?? [])])]
    .sort()
    .map((family) => [family, preparation[family]] as const)
  if (compact) {
    const roles = entries.map(([, entry]) => evaluationRoleTitle(entry?.evaluation_role ?? null))
    return (
      <p className={styles.muted}>
        {['Holdout', 'Retest', 'Role unspecified']
          .flatMap((role) => {
            const count = roles.filter((value) => value === role).length
            return count
              ? [`${number(count)} ${role === 'Role unspecified' ? 'Unspecified' : role}`]
              : []
          })
          .join(' · ')}
      </p>
    )
  }
  const families = (
    <ul>
      {entries.map(([family, entry]) => (
        <li key={family}>
          <strong>{benchmarkTitle(family)}</strong>:{' '}
          {evaluationRoleTitle(entry?.evaluation_role ?? null)}
          {!entry && ' · No preparation provenance'}
          {entry && (
            <>
              {' · '}
              {number(entry.selected_count)} selected · {number(entry.excluded_count)} excluded
              {' · '}
              {entry.coverage === 'named-memberships-only'
                ? 'Recorded history checked'
                : 'No history qualification'}
            </>
          )}
        </li>
      ))}
    </ul>
  )
  return (
    <section className={styles.notice} aria-label="Dataset evaluation roles">
      <h3>Evaluation roles</h3>
      {families}
      {entries.some(([, entry]) => entry?.evaluation_role === 'retest') && (
        <p>Includes retest families. Combined results are not an unseen holdout aggregate.</p>
      )}
      {entries.some(([, entry]) => entry?.coverage === 'named-memberships-only') && (
        <p>
          History exclusions cover only the frozen dataset and run memberships. Other prior exposure
          is not ruled out.
        </p>
      )}
    </section>
  )
}
