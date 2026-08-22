import { useCallback, useEffect, useMemo, useState } from 'react'

import type { ConfigData, RecipeConfig } from './configPageSupport'
import {
  applyRoutingScopeProjection,
  listRoutingScopes,
  projectConfigForRoutingScope,
  type RoutingScopedConfigLike,
} from '../utils/routingScopes'
import { announceRecipeDraftChange, recipeDraftApi, type RecipeDraft } from '../utils/recipeDrafts'

const requestedRecipeScope = () =>
  typeof window === 'undefined'
    ? ''
    : new URLSearchParams(window.location.search).get('recipe')?.trim() || ''

export function useRoutingScopeManager(
  config: ConfigData | null,
  saveConfig: (config: ConfigData) => Promise<void>,
) {
  const [drafts, setDrafts] = useState<RecipeDraft[]>([])
  const [selectedScopeId, setSelectedScopeId] = useState(requestedRecipeScope)

  const loadDrafts = useCallback(async () => {
    try {
      setDrafts((await recipeDraftApi.list()).items)
    } catch {
      setDrafts([])
    }
  }, [])

  useEffect(() => {
    void loadDrafts()
    const refresh = () => void loadDrafts()
    window.addEventListener('recipe-drafts-changed', refresh)
    return () => window.removeEventListener('recipe-drafts-changed', refresh)
  }, [loadDrafts])

  const combinedConfig = useMemo<ConfigData | null>(() => {
    if (!config) return null
    const liveNames = new Set((config.recipes ?? []).map((recipe) => recipe.name))
    return {
      ...config,
      recipes: [...(config.recipes ?? []), ...drafts.filter((draft) => !liveNames.has(draft.name))],
    }
  }, [config, drafts])

  const routingScopes = useMemo(
    () => listRoutingScopes(combinedConfig as ConfigData & RoutingScopedConfigLike),
    [combinedConfig],
  )

  useEffect(() => {
    if (routingScopes.some((scope) => scope.id === selectedScopeId)) return
    setSelectedScopeId(routingScopes[0]?.id ?? '')
  }, [routingScopes, selectedScopeId])

  const scopedConfig = useMemo(
    () =>
      combinedConfig
        ? projectConfigForRoutingScope(
            combinedConfig as ConfigData & RoutingScopedConfigLike,
            selectedScopeId,
          )
        : null,
    [combinedConfig, selectedScopeId],
  )

  const saveScopedConfig = useCallback(
    async (projectedConfig: ConfigData): Promise<void> => {
      if (!config || !combinedConfig) {
        throw new Error('Configuration not loaded yet.')
      }
      const draft = drafts.find((candidate) => candidate.name === selectedScopeId)
      const live = (config.recipes ?? []).some((recipe) => recipe.name === selectedScopeId)
      if (draft && !live) {
        const updated = applyRoutingScopeProjection(
          combinedConfig as ConfigData & RoutingScopedConfigLike,
          projectedConfig as ConfigData & RoutingScopedConfigLike,
          selectedScopeId,
        )
        const recipe = updated.recipes?.find((candidate) => candidate.name === selectedScopeId) as
          | RecipeConfig
          | undefined
        if (!recipe) throw new Error('Recipe draft no longer exists.')
        await recipeDraftApi.save(recipe)
        announceRecipeDraftChange()
        return
      }
      await saveConfig(
        applyRoutingScopeProjection(
          config as ConfigData & RoutingScopedConfigLike,
          projectedConfig as ConfigData & RoutingScopedConfigLike,
          selectedScopeId,
        ),
      )
    },
    [combinedConfig, config, drafts, saveConfig, selectedScopeId],
  )

  return {
    isDraftScope: drafts.some((draft) => draft.name === selectedScopeId),
    routingScopes,
    saveScopedConfig,
    scopedConfig,
    selectedScope: routingScopes.find((scope) => scope.id === selectedScopeId),
    selectedScopeId,
    setSelectedScopeId,
  }
}
