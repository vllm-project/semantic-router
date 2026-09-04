import { useEffect, useState } from 'react'

import type { EvaluationCampaignReadiness } from '../types/evaluationCampaign'
import type { EvaluationCatalogChangeProfile } from '../types/evaluationPlane'
import type { EvaluationCampaignReadinessAnchors } from '../utils/evaluationCampaignReadinessContract'
import { getEvaluationCampaignReadiness } from '../utils/evaluationPlaneApi'

interface CampaignReadinessState {
  key: string
  readiness: EvaluationCampaignReadiness | null
  loading: boolean
  error: string | null
}

export default function useEvaluationCampaignReadiness(
  profile: EvaluationCatalogChangeProfile | undefined,
  enabled: boolean,
  anchors: EvaluationCampaignReadinessAnchors = {},
  ledgerRevision = '',
) {
  const controlledPairBaselineRunID = anchors.controlledPairBaselineRunID || ''
  const fidelityReferenceRunID = anchors.fidelityReferenceRunID || ''
  const key = profile
    ? `${profile.id}:${controlledPairBaselineRunID}:${fidelityReferenceRunID}:${ledgerRevision}`
    : ''
  const [state, setState] = useState<CampaignReadinessState>({
    key: '',
    readiness: null,
    loading: false,
    error: null,
  })

  useEffect(() => {
    if (!enabled || !profile) {
      setState({ key, readiness: null, loading: false, error: null })
      return
    }
    const controller = new AbortController()
    setState({ key, readiness: null, loading: true, error: null })
    void getEvaluationCampaignReadiness(
      profile,
      {
        ...(controlledPairBaselineRunID ? { controlledPairBaselineRunID } : {}),
        ...(fidelityReferenceRunID ? { fidelityReferenceRunID } : {}),
      },
      controller.signal,
    )
      .then((readiness) => {
        if (!controller.signal.aborted) {
          setState({ key, readiness, loading: false, error: null })
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setState({
            key,
            readiness: null,
            loading: false,
            error:
              error instanceof Error
                ? error.message
                : 'Release-check readiness could not be loaded.',
          })
        }
      })
    return () => controller.abort()
  }, [controlledPairBaselineRunID, enabled, fidelityReferenceRunID, key, profile])

  return state.key === key
    ? state
    : { key, readiness: null, loading: enabled, error: null }
}
