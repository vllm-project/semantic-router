import { EmptyRunInspector, LoadedRunInspector } from './EvaluationRunInspectorContent'
import type { EvaluationRunInspectorProps } from './EvaluationRunInspector.types'
import styles from './EvaluationRunInspector.module.css'

export default function EvaluationRunInspector(props: EvaluationRunInspectorProps) {
  const { run, loading, controlledPairLoading, controlledPairRefreshing } = props
  return (
    <aside
      className={styles.runInspector}
      aria-label="Selected evaluation run"
      aria-busy={loading || controlledPairLoading || controlledPairRefreshing}
    >
      {run ? <LoadedRunInspector {...props} run={run} /> : <EmptyRunInspector {...props} />}
    </aside>
  )
}
