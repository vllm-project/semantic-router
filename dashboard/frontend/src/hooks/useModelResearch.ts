import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type {
  ModelResearchCampaign,
  ModelResearchCreateRequest,
  ModelResearchEvent,
  ModelResearchGoalTemplate,
  ModelResearchRecipesResponse,
} from '../types/modelResearch'
import * as api from '../utils/modelResearchApi'

export function useModelResearch() {
  const [recipesResponse, setRecipesResponse] = useState<ModelResearchRecipesResponse | null>(null)
  const [campaigns, setCampaigns] = useState<ModelResearchCampaign[]>([])
  const [selectedCampaignId, setSelectedCampaignId] = useState<string | null>(null)
  const [selectedCampaign, setSelectedCampaign] = useState<ModelResearchCampaign | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const eventCleanupRef = useRef<(() => void) | null>(null)

  const refreshOverview = useCallback(async () => {
    const [recipes, campaignList] = await Promise.all([api.getRecipes(), api.listCampaigns()])
    setRecipesResponse(recipes)
    setCampaigns(campaignList)
    setSelectedCampaignId((current) => current ?? campaignList[0]?.id ?? null)
  }, [])

  const refreshSelectedCampaign = useCallback(async (campaignId: string) => {
    const campaign = await api.getCampaign(campaignId)
    setSelectedCampaign(campaign)
    setCampaigns((prev) =>
      prev.map((item) => (item.id === campaign.id ? campaign : item))
    )
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    refreshOverview()
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load model research catalog')
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [refreshOverview])

  useEffect(() => {
    if (!selectedCampaignId) {
      setSelectedCampaign(null)
      return
    }

    let cancelled = false
    refreshSelectedCampaign(selectedCampaignId).catch((err) => {
      if (!cancelled) {
        setError(err instanceof Error ? err.message : 'Failed to load campaign detail')
      }
    })

    return () => {
      cancelled = true
    }
  }, [refreshSelectedCampaign, selectedCampaignId])

  useEffect(() => {
    if (!selectedCampaign || !selectedCampaignId) {
      if (eventCleanupRef.current) {
        eventCleanupRef.current()
        eventCleanupRef.current = null
      }
      return
    }

    if (!['pending', 'running'].includes(selectedCampaign.status)) {
      if (eventCleanupRef.current) {
        eventCleanupRef.current()
        eventCleanupRef.current = null
      }
      return
    }

    if (eventCleanupRef.current) {
      eventCleanupRef.current()
      eventCleanupRef.current = null
    }

    eventCleanupRef.current = api.subscribeToCampaignEvents(
      selectedCampaignId,
      (event: ModelResearchEvent) => {
        setSelectedCampaign((prev) => {
          if (!prev || prev.id !== selectedCampaignId) {
            return prev
          }
          const nextKey = `${event.timestamp}|${event.kind}|${event.message}|${event.trial_index ?? 0}`
          const existing = new Set(
            (prev.events ?? []).map(
              (item) => `${item.timestamp}|${item.kind}|${item.message}|${item.trial_index ?? 0}`
            )
          )
          if (existing.has(nextKey)) {
            return prev
          }
          const events = [...(prev.events ?? []), event]
          return {
            ...prev,
            events: events.slice(-200),
          }
        })
      },
      () => {
        void refreshSelectedCampaign(selectedCampaignId)
        void refreshOverview()
      },
      (streamError) => {
        setError(streamError.message)
      }
    )

    const poll = window.setInterval(() => {
      void refreshSelectedCampaign(selectedCampaignId)
      void refreshOverview()
    }, 5000)

    return () => {
      window.clearInterval(poll)
      if (eventCleanupRef.current) {
        eventCleanupRef.current()
        eventCleanupRef.current = null
      }
    }
  }, [refreshOverview, refreshSelectedCampaign, selectedCampaign, selectedCampaignId])

  const createCampaign = useCallback(async (payload: ModelResearchCreateRequest) => {
    setSubmitting(true)
    setError(null)
    try {
      const campaign = await api.createCampaign(payload)
      setSelectedCampaignId(campaign.id)
      await refreshOverview()
      await refreshSelectedCampaign(campaign.id)
      return campaign
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create campaign'
      setError(message)
      throw err
    } finally {
      setSubmitting(false)
    }
  }, [refreshOverview, refreshSelectedCampaign])

  const stopSelectedCampaign = useCallback(async () => {
    if (!selectedCampaignId) return
    setError(null)
    await api.stopCampaign(selectedCampaignId)
    await refreshSelectedCampaign(selectedCampaignId)
    await refreshOverview()
  }, [refreshOverview, refreshSelectedCampaign, selectedCampaignId])

  const recipes = useMemo(() => recipesResponse?.recipes ?? [], [recipesResponse])

  const targetsByGoal = useMemo(() => {
    return {
      improve_accuracy: recipes.filter((recipe) => recipe.goal_templates.includes('improve_accuracy')),
      explore_signal: recipes.filter((recipe) => recipe.goal_templates.includes('explore_signal')),
    } satisfies Record<ModelResearchGoalTemplate, typeof recipes>
  }, [recipes])

  return {
    recipesResponse,
    recipes,
    campaigns,
    selectedCampaign,
    selectedCampaignId,
    setSelectedCampaignId,
    targetsByGoal,
    loading,
    submitting,
    error,
    setError,
    createCampaign,
    stopSelectedCampaign,
    refreshOverview,
  }
}
