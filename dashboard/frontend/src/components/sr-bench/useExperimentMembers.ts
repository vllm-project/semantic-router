import { useEffect, useMemo, useState } from 'react'
import type { ExperimentMember } from './experimentApi'
import { readExperimentMembers } from './experimentComparison'

const empty: ExperimentMember[] = []

export default function useExperimentMembers(experiment?: string, revision = 0) {
  const [reloadCount, setReloadCount] = useState(0)
  const [state, setState] = useState({
    scope: '',
    members: empty,
    complete: false,
    error: '',
  })
  const scope = `${experiment ?? ''}:${revision}:${reloadCount}`
  const current = state.scope === scope ? state : null
  useEffect(() => {
    if (!experiment) return
    const controller = new AbortController()
    setState({ scope, members: empty, complete: false, error: '' })
    void readExperimentMembers(experiment, controller.signal).then(
      (members) => {
        if (!controller.signal.aborted) setState({ scope, members, complete: true, error: '' })
      },
      (cause: unknown) => {
        if (!controller.signal.aborted)
          setState({
            scope,
            members: empty,
            complete: false,
            error: cause instanceof Error ? cause.message : 'Could not read experiment membership.',
          })
      },
    )
    return () => controller.abort()
  }, [experiment, scope])
  const members = current?.members ?? empty
  const ids = useMemo(() => new Set(members.map((member) => member.run_id)), [members])
  return {
    members,
    ids,
    complete: !!experiment && !!current?.complete,
    loading: !!experiment && !current?.complete && !current?.error,
    error: experiment ? (current?.error ?? '') : '',
    reload: () => setReloadCount((value) => value + 1),
  }
}
