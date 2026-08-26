import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import {
  routingManagementApi,
  waitForRoutingMutation,
  type RoutingRecipe,
  type RoutingRecipeWrite,
} from '../utils/routingManagementApi'
import type { RoutingProfileLike, RoutingScope } from '../utils/routingScopes'
import type { ConfigData } from './configPageSupport'

const recipeScopeFromSearch = (searchParams: URLSearchParams) =>
  searchParams.get('recipe')?.trim() || ''

export function withRecipeScope(searchParams: URLSearchParams, recipeId: string): URLSearchParams {
  const next = new URLSearchParams(searchParams)
  const normalizedRecipeId = recipeId.trim()
  if (normalizedRecipeId) next.set('recipe', normalizedRecipeId)
  else next.delete('recipe')
  return next
}

const cloneValue = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

function recipeProfile(recipe: RoutingRecipe): RoutingProfileLike {
  const document = recipe.document
  return {
    strategy: document.strategy,
    signals: document.signals as Record<string, unknown> | undefined,
    projections: document.projections as Record<string, unknown> | undefined,
    decisions: Array.isArray(document.decisions) ? document.decisions : [],
  }
}

function recipeScope(recipe: RoutingRecipe): RoutingScope {
  return {
    id: recipe.id,
    label: recipe.name,
    description: recipe.description,
    entrypointModelNames: [],
    document: recipeProfile(recipe),
  }
}

export function managedRecipeConfig(recipe: RoutingRecipe): ConfigData {
  return cloneValue(recipeProfile(recipe)) as ConfigData
}

export function managedRecipeDocument(config: ConfigData): Record<string, unknown> {
  return {
    ...(config.strategy ? { strategy: config.strategy } : {}),
    ...(config.signals ? { signals: cloneValue(config.signals) } : {}),
    ...(config.projections ? { projections: cloneValue(config.projections) } : {}),
    decisions: cloneValue(config.decisions ?? []),
  }
}

export function useRoutingScopeManager() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [recipes, setRecipes] = useState<RoutingRecipe[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const requestedScopeId = recipeScopeFromSearch(searchParams)

  const loadRecipes = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const next = await routingManagementApi.listRecipes()
      setRecipes(next)
    } catch (cause) {
      setRecipes([])
      setError(cause instanceof Error ? cause.message : 'Recipes could not be loaded.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadRecipes()
  }, [loadRecipes])

  const routingScopes = useMemo(() => recipes.map(recipeScope), [recipes])
  const selectedRecipe = useMemo(
    () =>
      recipes.find(
        (recipe) => recipe.id === requestedScopeId || recipe.name === requestedScopeId,
      ) ?? recipes[0],
    [recipes, requestedScopeId],
  )
  const scopedConfig = useMemo(
    () => (selectedRecipe ? managedRecipeConfig(selectedRecipe) : null),
    [selectedRecipe],
  )

  const setSelectedScopeId = useCallback(
    (scopeId: string) => {
      setSearchParams((current) => withRecipeScope(current, scopeId), { replace: true })
    },
    [setSearchParams],
  )

  useEffect(() => {
    if (!selectedRecipe || requestedScopeId === selectedRecipe.id) return
    setSearchParams((current) => withRecipeScope(current, selectedRecipe.id), { replace: true })
  }, [requestedScopeId, selectedRecipe, setSearchParams])

  const saveScopedConfig = useCallback(
    async (projectedConfig: ConfigData): Promise<void> => {
      if (!selectedRecipe) throw new Error('Choose a Recipe first.')
      if (selectedRecipe.immutable) {
        throw new Error('Built-in Recipes are read-only. Duplicate it to make changes.')
      }
      const input: RoutingRecipeWrite = {
        name: selectedRecipe.name,
        description: selectedRecipe.description,
        document: managedRecipeDocument(projectedConfig),
      }
      await waitForRoutingMutation(
        await routingManagementApi.updateRecipe(selectedRecipe.id, selectedRecipe.revision, input),
      )
      await loadRecipes()
    },
    [loadRecipes, selectedRecipe],
  )

  return {
    error,
    loading,
    reload: loadRecipes,
    routingScopes,
    saveScopedConfig,
    scopedConfig,
    selectedRecipe,
    selectedScope: selectedRecipe ? recipeScope(selectedRecipe) : undefined,
    selectedScopeId: selectedRecipe?.id ?? requestedScopeId,
    setSelectedScopeId,
  }
}
