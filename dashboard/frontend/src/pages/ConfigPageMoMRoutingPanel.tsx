import { useCallback, useEffect, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import ProductLoadingState from '../components/ProductLoadingState'
import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import {
  routingManagementApi,
  waitForRoutingMutation,
  type RoutingEntrypoint,
  type RoutingEntrypointWrite,
  type RoutingModelCardView,
  type RoutingRecipe,
  type RoutingRecipeWrite,
} from '../utils/routingManagementApi'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import ConfigPageMixtureDialog from './ConfigPageMixtureDialog'
import ConfigPageMoMTopologyDialog from './ConfigPageMoMTopologyDialog'
import { ConfigPageMoMEntrypointsList, ConfigPageMoMRecipesList } from './ConfigPageMoMRoutingLists'
import ConfigPageRecipeDialog from './ConfigPageRecipeDialog'
import type { MixtureWorkspaceView } from './ConfigPageEntrypointsRecipesSection'

interface Props {
  activeView: MixtureWorkspaceView
  canRead: boolean
  canManage: boolean
}

type PendingDelete =
  | { kind: 'entrypoint'; value: RoutingEntrypoint }
  | { kind: 'recipe'; value: RoutingRecipe }

export default function ConfigPageMoMRoutingPanel({ activeView, canRead, canManage }: Props) {
  const { catalogError, catalogSnapshot, catalogStatus, retryCatalog, usesKeyScopedCatalog } =
    useInferenceRoutingAccess()
  const [models, setModels] = useState<RoutingModelCardView[]>([])
  const [recipes, setRecipes] = useState<RoutingRecipe[]>([])
  const [entrypoints, setEntrypoints] = useState<RoutingEntrypoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [mixtureEditor, setMixtureEditor] = useState<RoutingEntrypoint | null | undefined>(
    undefined,
  )
  const [recipeEditor, setRecipeEditor] = useState<RoutingRecipe | null | undefined>(undefined)
  const [recipeTemplate, setRecipeTemplate] = useState<RoutingRecipe | null>(null)
  const [topologyTarget, setTopologyTarget] = useState<RoutingEntrypoint | null>(null)
  const [topologyError, setTopologyError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null)
  const [mutationPending, setMutationPending] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!canRead) {
      setLoading(false)
      return
    }
    if (usesKeyScopedCatalog) {
      setError(catalogError)
      if (catalogStatus === 'ready' && catalogSnapshot) {
        setModels(catalogSnapshot.models)
        setRecipes(catalogSnapshot.recipes)
        setEntrypoints(catalogSnapshot.entrypoints)
        setLoading(false)
        return
      }
      setModels([])
      setRecipes([])
      setEntrypoints([])
      setLoading(catalogStatus === 'idle' || catalogStatus === 'loading')
      if (catalogStatus === 'unavailable') {
        setError('Create an API key to view your available models.')
      }
      return
    }
    setLoading(true)
    setError(null)
    try {
      const [nextModels, nextRecipes, summaries] = await Promise.all([
        routingManagementApi.listModelCards(),
        routingManagementApi.listRecipes(),
        routingManagementApi.listEntrypoints(),
      ])
      setModels(nextModels)
      setRecipes(nextRecipes)
      setEntrypoints(summaries)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Routing could not be loaded.')
    } finally {
      setLoading(false)
    }
  }, [canRead, catalogError, catalogSnapshot, catalogStatus, usesKeyScopedCatalog])

  useEffect(() => {
    void load()
  }, [load])

  const openEntrypoint = async (entrypoint: RoutingEntrypoint) => {
    setTopologyError(null)
    try {
      const detail = entrypoint.rules
        ? entrypoint
        : await routingManagementApi.getEntrypointTopology(entrypoint.id)
      setTopologyTarget(detail)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Topology is not available.')
    }
  }

  const editEntrypoint = async (entrypoint: RoutingEntrypoint) => {
    if (!canManage) return
    setTopologyError(null)
    try {
      const detail = entrypoint.rules
        ? entrypoint
        : await routingManagementApi.getEntrypointTopology(entrypoint.id)
      setTopologyTarget(null)
      setMixtureEditor(detail)
    } catch (cause) {
      setTopologyError(cause instanceof Error ? cause.message : 'Model could not be opened.')
    }
  }

  const saveEntrypoint = async (input: RoutingEntrypointWrite) => {
    if (!canManage) throw new Error('Routing management access is required.')
    const receipt = mixtureEditor
      ? await routingManagementApi.updateEntrypoint(mixtureEditor.id, mixtureEditor.revision, input)
      : await routingManagementApi.createEntrypoint(input)
    await waitForRoutingMutation(receipt)
    await load()
  }

  const saveRecipe = async (input: RoutingRecipeWrite) => {
    if (!canManage) throw new Error('Routing management access is required.')
    const receipt = recipeEditor
      ? await routingManagementApi.updateRecipe(recipeEditor.id, recipeEditor.revision, input)
      : await routingManagementApi.createRecipe(input)
    await waitForRoutingMutation(receipt)
    await load()
  }

  const changePublication = async (entrypoint: RoutingEntrypoint) => {
    if (!canManage) return
    setMutationPending(true)
    setTopologyError(null)
    try {
      const receipt =
        entrypoint.status === 'active'
          ? await routingManagementApi.unpublishEntrypoint(entrypoint.id, entrypoint.revision)
          : await routingManagementApi.publishEntrypoint(entrypoint.id, entrypoint.revision)
      await waitForRoutingMutation(receipt)
      setTopologyTarget(null)
      await load()
    } catch (cause) {
      setTopologyError(cause instanceof Error ? cause.message : 'Publication could not be changed.')
    } finally {
      setMutationPending(false)
    }
  }

  const confirmDelete = async () => {
    if (!pendingDelete || !canManage) return
    setMutationPending(true)
    setDeleteError(null)
    try {
      if (pendingDelete.kind === 'entrypoint') {
        await routingManagementApi.deleteEntrypoint(
          pendingDelete.value.id,
          pendingDelete.value.revision,
        )
      } else {
        await routingManagementApi.deleteRecipe(
          pendingDelete.value.id,
          pendingDelete.value.revision,
        )
      }
      setPendingDelete(null)
      setTopologyTarget(null)
      setRecipeEditor(undefined)
      setRecipeTemplate(null)
      await load()
    } catch (cause) {
      setDeleteError(cause instanceof Error ? cause.message : 'Resource could not be deleted.')
    } finally {
      setMutationPending(false)
    }
  }

  if (!canRead) {
    return <div className={pageStyles.emptyState}>Routing access is required.</div>
  }
  if (loading) return <ProductLoadingState compact label="Loading Mixture-of-Models" />
  if (error && usesKeyScopedCatalog) {
    return (
      <div className={pageStyles.emptyState} role="alert">
        <span>{error}</span>
        <button type="button" onClick={retryCatalog}>
          Try again
        </button>
      </div>
    )
  }

  return (
    <>
      {error ? (
        <div className={pageStyles.emptyState} role="alert">
          {error}
        </div>
      ) : null}
      <div className={pageStyles.tablesGrid}>
        {activeView === 'models' ? (
          <ConfigPageMoMEntrypointsList
            entrypoints={entrypoints}
            canManage={canManage}
            onAdd={() => setMixtureEditor(null)}
            onView={(entrypoint) => void openEntrypoint(entrypoint)}
          />
        ) : (
          <ConfigPageMoMRecipesList
            recipes={recipes}
            canManage={canManage}
            onAdd={() => {
              setRecipeTemplate(null)
              setRecipeEditor(null)
            }}
            onView={(recipe) => {
              setRecipeTemplate(null)
              setRecipeEditor(recipe)
            }}
          />
        )}
      </div>

      {mixtureEditor !== undefined ? (
        <ConfigPageMixtureDialog
          models={models}
          recipes={recipes}
          entrypoint={mixtureEditor ?? undefined}
          onClose={() => setMixtureEditor(undefined)}
          onSave={saveEntrypoint}
        />
      ) : null}
      {recipeEditor !== undefined ? (
        <ConfigPageRecipeDialog
          key={recipeEditor?.id ?? recipeTemplate?.id ?? 'new-recipe'}
          recipe={recipeEditor ?? undefined}
          duplicateFrom={recipeTemplate ?? undefined}
          readOnly={!canManage || Boolean(recipeEditor?.immutable)}
          onClose={() => {
            setRecipeEditor(undefined)
            setRecipeTemplate(null)
          }}
          onSave={saveRecipe}
          onDuplicate={
            recipeEditor?.immutable && canManage
              ? () => {
                  setRecipeTemplate(recipeEditor)
                  setRecipeEditor(null)
                }
              : undefined
          }
          onDelete={
            recipeEditor && canManage && !recipeEditor.immutable
              ? () => {
                  setDeleteError(null)
                  setPendingDelete({ kind: 'recipe', value: recipeEditor })
                }
              : undefined
          }
        />
      ) : null}
      {topologyTarget ? (
        <ConfigPageMoMTopologyDialog
          entrypoint={topologyTarget}
          recipes={recipes}
          models={models}
          canManage={canManage}
          pending={mutationPending}
          error={topologyError}
          onClose={() => {
            setTopologyTarget(null)
            setTopologyError(null)
          }}
          onEdit={() => void editEntrypoint(topologyTarget)}
          onPublish={() => void changePublication(topologyTarget)}
          onDelete={() => {
            setDeleteError(null)
            setPendingDelete({ kind: 'entrypoint', value: topologyTarget })
          }}
        />
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== null}
        title={`Delete ${pendingDelete?.kind === 'recipe' ? 'recipe' : 'model'}?`}
        description="This cannot be undone."
        eyebrow="Routing"
        confirmLabel="Delete"
        pending={mutationPending}
        details={deleteError ? <div role="alert">{deleteError}</div> : undefined}
        onCancel={() => {
          if (!mutationPending) {
            setPendingDelete(null)
            setDeleteError(null)
          }
        }}
        onConfirm={confirmDelete}
      />
    </>
  )
}
