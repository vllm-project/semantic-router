import { useCallback, useEffect, useRef, useState } from 'react'

import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign,
} from '../types/evaluationCampaign'
import { createEvaluationCampaign, getEvaluationCampaign } from '../utils/evaluationPlaneApi'

function message(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

type CampaignState =
  | { key: string; status: 'idle'; campaign: null; error: null }
  | { key: string; status: 'loading'; campaign: null; error: string | null }
  | { key: string; status: 'ready'; campaign: EvaluationCampaign; error: null }
  | { key: string; status: 'error'; campaign: null; error: string }

export function useEvaluationCampaign(id: string | null) {
  const key = id || ''
  const [state, setState] = useState<CampaignState>({
    key,
    status: 'idle',
    campaign: null,
    error: null,
  })
  const requestVersion = useRef(0)
  const controller = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    if (!id) return
    const version = ++requestVersion.current
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setState((current) => ({
      key,
      status: 'loading',
      campaign: null,
      error: current.key === key ? current.error : null,
    }))
    try {
      const campaign = await getEvaluationCampaign(id, nextController.signal)
      if (nextController.signal.aborted || version !== requestVersion.current) return
      setState({ key, status: 'ready', campaign, error: null })
    } catch (error) {
      if (nextController.signal.aborted || version !== requestVersion.current) return
      setState({
        key,
        status: 'error',
        campaign: null,
        error: message(error, 'Failed to load the promotion campaign.'),
      })
    }
  }, [id, key])

  useEffect(() => {
    requestVersion.current += 1
    controller.current?.abort()
    if (!id) {
      setState({ key, status: 'idle', campaign: null, error: null })
      return
    }
    void refresh()
    return () => {
      requestVersion.current += 1
      controller.current?.abort()
    }
  }, [id, key, refresh])

  const current = state.key === key ? state : null
  return {
    campaign: current?.campaign || null,
    loading: Boolean(id) && (!current || current.status === 'idle' || current.status === 'loading'),
    error: current?.error || null,
    refresh,
  }
}

export function useCreateEvaluationCampaign() {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [campaign, setCampaign] = useState<EvaluationCampaign | null>(null)
  const lock = useRef(false)

  const create = useCallback(async (request: CreateEvaluationCampaignPayload) => {
    if (lock.current) return null
    lock.current = true
    setPending(true)
    setError(null)
    try {
      const created = await createEvaluationCampaign(request)
      setCampaign(created)
      return created
    } catch (createError) {
      setError(message(createError, 'Failed to create the promotion campaign.'))
      return null
    } finally {
      lock.current = false
      setPending(false)
    }
  }, [])

  return {
    pending,
    error,
    campaign,
    create,
    clearError: () => setError(null),
    reset: () => {
      setCampaign(null)
      setError(null)
    },
  }
}
