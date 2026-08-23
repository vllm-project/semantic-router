import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useDSLStore } from '@/stores/dslStore'
import {
  routingManagementApi,
  waitForRoutingMutation,
  type RoutingRecipe,
} from '@/utils/routingManagementApi'

import { compileBuilderRecipe, loadManagedRecipeSource } from './builderRecipeClient'

interface BuilderRecipeClientState {
  recipes: RoutingRecipe[]
  selectedRecipe: RoutingRecipe | null
  selectedRecipeId: string
  loading: boolean
  saving: boolean
  error: string | null
  notice: string | null
}

const message = (cause: unknown, fallback: string) =>
  cause instanceof Error && cause.message.trim() ? cause.message : fallback

export function useBuilderRecipeClient(enabled: boolean) {
  const [state, setState] = useState<BuilderRecipeClientState>({
    recipes: [],
    selectedRecipe: null,
    selectedRecipeId: '',
    loading: true,
    saving: false,
    error: null,
    notice: null,
  })
  const loadedRevision = useRef('')
  const noticeTimer = useRef<ReturnType<typeof window.setTimeout> | null>(null)

  const showNotice = useCallback((notice: string) => {
    if (noticeTimer.current) window.clearTimeout(noticeTimer.current)
    setState((current) => ({ ...current, notice }))
    noticeTimer.current = window.setTimeout(
      () => setState((current) => ({ ...current, notice: null })),
      2400,
    )
  }, [])

  useEffect(
    () => () => {
      if (noticeTimer.current) window.clearTimeout(noticeTimer.current)
    },
    [],
  )

  const refresh = useCallback(async (preferredId?: string) => {
    setState((current) => ({ ...current, loading: true, error: null }))
    try {
      const recipes = await routingManagementApi.listRecipes()
      setState((current) => {
        const requested = preferredId || current.selectedRecipeId
        const selectedRecipe =
          recipes.find((recipe) => recipe.id === requested) ?? recipes[0] ?? null
        return {
          ...current,
          recipes,
          selectedRecipe,
          selectedRecipeId: selectedRecipe?.id ?? '',
          loading: false,
        }
      })
    } catch (cause) {
      setState((current) => ({
        ...current,
        loading: false,
        error: message(cause, 'Recipes are unavailable.'),
      }))
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    void refresh(new URLSearchParams(window.location.search).get('recipe') || undefined)
  }, [enabled, refresh])

  useEffect(() => {
    if (!enabled || !state.selectedRecipe) return
    const revisionKey = `${state.selectedRecipe.id}:${state.selectedRecipe.revision}`
    if (loadedRevision.current === revisionKey) return
    try {
      const compiled = loadManagedRecipeSource(state.selectedRecipe)
      const store = useDSLStore.getState()
      store.loadDsl(compiled.source)
      store.compile()
      store.parseAST()
      loadedRevision.current = revisionKey
      setState((current) => ({ ...current, error: null }))
    } catch (cause) {
      setState((current) => ({
        ...current,
        error: message(cause, 'The selected Recipe could not be opened.'),
      }))
    }
  }, [enabled, state.selectedRecipe])

  const selectRecipe = useCallback((recipeId: string) => {
    setState((current) => ({
      ...current,
      selectedRecipeId: recipeId,
      selectedRecipe: current.recipes.find((recipe) => recipe.id === recipeId) ?? null,
      error: null,
    }))
  }, [])

  const save = useCallback(async () => {
    const recipe = state.selectedRecipe
    if (!recipe) throw new Error('Select a Recipe first.')
    if (recipe.immutable) throw new Error('Built-in Recipes are read only. Duplicate it to edit.')
    const compiled = compileBuilderRecipe(useDSLStore.getState().dslSource, recipe)
    setState((current) => ({ ...current, saving: true, error: null }))
    try {
      const receipt = await routingManagementApi.updateRecipe(recipe.id, recipe.revision, {
        name: recipe.name,
        ...(recipe.description ? { description: recipe.description } : {}),
        document: compiled.document,
      })
      await waitForRoutingMutation(receipt)
      loadedRevision.current = ''
      await refresh(recipe.id)
      showNotice('Recipe saved')
    } catch (cause) {
      const error = message(cause, 'Recipe could not be saved.')
      setState((current) => ({ ...current, error }))
      throw new Error(error)
    } finally {
      setState((current) => ({ ...current, saving: false }))
    }
  }, [refresh, showNotice, state.selectedRecipe])

  const duplicate = useCallback(
    async (name: string, description?: string) => {
      const recipe = state.selectedRecipe
      if (!recipe) throw new Error('Select a Recipe first.')
      const compiled = compileBuilderRecipe(useDSLStore.getState().dslSource, recipe)
      setState((current) => ({ ...current, saving: true, error: null }))
      try {
        const receipt = await routingManagementApi.createRecipe({
          name,
          ...(description?.trim() ? { description: description.trim() } : {}),
          document: compiled.document,
        })
        await waitForRoutingMutation(receipt)
        loadedRevision.current = ''
        await refresh(receipt.resource?.id)
        showNotice('Recipe duplicated')
      } catch (cause) {
        const error = message(cause, 'Recipe could not be duplicated.')
        setState((current) => ({ ...current, error }))
        throw new Error(error)
      } finally {
        setState((current) => ({ ...current, saving: false }))
      }
    },
    [refresh, showNotice, state.selectedRecipe],
  )

  return useMemo(
    () => ({ ...state, refresh, selectRecipe, save, duplicate, showNotice }),
    [duplicate, refresh, save, selectRecipe, showNotice, state],
  )
}
