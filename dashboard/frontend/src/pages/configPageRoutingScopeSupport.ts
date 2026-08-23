import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  routingManagementApi,
  waitForRoutingMutation,
  type RoutingRecipe,
  type RoutingRecipeWrite,
} from '../utils/routingManagementApi'
import type { RoutingProfileLike, RoutingScope } from '../utils/routingScopes'
import type { ConfigData } from './configPageSupport'

const requestedRecipeScope = () =>
  typeof window === 'undefined'
    ? ''
    : new URLSearchParams(window.location.search).get('recipe')?.trim() || ''

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
  const [recipes, setRecipes] = useState<RoutingRecipe[]>([])
  const [selectedScopeId, setSelectedScopeId] = useState(requestedRecipeScope)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadRecipes = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const next = await routingManagementApi.listRecipes()
      setRecipes(next)
      setSelectedScopeId((current) => {
        const match = next.find((recipe) => recipe.id === current || recipe.name === current)
        return match?.id ?? next[0]?.id ?? ''
      })
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
    () => recipes.find((recipe) => recipe.id === selectedScopeId) ?? recipes[0],
    [recipes, selectedScopeId],
  )
  const scopedConfig = useMemo(
    () => (selectedRecipe ? managedRecipeConfig(selectedRecipe) : null),
    [selectedRecipe],
  )

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
    selectedScopeId: selectedRecipe?.id ?? selectedScopeId,
    setSelectedScopeId,
  }
}
