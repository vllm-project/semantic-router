import { useEffect, useState } from 'react'
import useRunOptions from './useRunOptions'
import { verifyExperimentBaselines } from './experimentComparison'
import useExperimentMembers from './useExperimentMembers'

export default function useExperimentComparison(baseline: string, experiment?: string) {
  const [revision, setRevision] = useState(0)
  const membership = useExperimentMembers(experiment, revision)
  const scope = `${experiment ?? ''}:${revision}`
  const scopeReady = !experiment || membership.complete
  const globalBaselines = useRunOptions('comparison', undefined, scopeReady)
  const baselineInScope = !experiment || membership.ids.has(baseline)
  const globalChoices = useRunOptions(
    'comparison',
    baseline || undefined,
    scopeReady && !!baseline && baselineInScope,
  )
  const [verified, setVerified] = useState({
    key: '',
    ids: new Set<string>(),
    error: '',
    scanLimited: false,
  })
  const verificationKey = `${scope}:${globalBaselines.items.map((item) => item.run_id).join('|')}`
  useEffect(() => {
    if (!experiment || !membership.complete || !globalBaselines.loaded) return
    const controller = new AbortController()
    setVerified({ key: '', ids: new Set(), error: '', scanLimited: false })
    void verifyExperimentBaselines(globalBaselines.items, membership.ids, controller.signal).then(
      (value) => {
        if (!controller.signal.aborted) setVerified({ key: verificationKey, ...value, error: '' })
      },
      (cause: unknown) => {
        if (!controller.signal.aborted)
          setVerified({
            key: verificationKey,
            ids: new Set(),
            scanLimited: false,
            error: cause instanceof Error ? cause.message : 'Could not verify experiment options.',
          })
      },
    )
    return () => controller.abort()
  }, [
    experiment,
    globalBaselines.items,
    globalBaselines.loaded,
    membership.complete,
    membership.ids,
    verificationKey,
  ])
  const checked = verified.key === verificationKey ? verified : null
  const choiceItems = globalChoices.items.filter(
    (item) => !experiment || membership.ids.has(item.run_id),
  )
  const baselineItems = globalBaselines.items.filter(
    (item) => !experiment || checked?.ids.has(item.run_id),
  )
  if (
    globalChoices.baseline &&
    baselineInScope &&
    choiceItems.length > 0 &&
    !baselineItems.some((item) => item.run_id === baseline)
  )
    baselineItems.unshift(globalChoices.baseline)
  return {
    membershipLoading: membership.loading,
    membershipError: membership.error,
    reloadMembership: membership.reload,
    baselines: {
      ...globalBaselines,
      items: baselineItems,
      loading:
        globalBaselines.loading ||
        (!!experiment && scopeReady && globalBaselines.loaded && !checked),
      error: globalBaselines.error || checked?.error || '',
      scanLimited: globalBaselines.scanLimited || !!checked?.scanLimited,
      reload: () => {
        setRevision((value) => value + 1)
        globalBaselines.refresh()
      },
    },
    choices: { ...globalChoices, items: choiceItems },
    baselineInScope,
  }
}
