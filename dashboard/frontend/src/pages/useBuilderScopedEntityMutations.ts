import { useCallback, type Dispatch, type SetStateAction } from 'react'

import * as dslMutations from '@/lib/dslMutations'
import type { RouteInput } from '@/lib/dslMutations'
import { useDSLStore } from '@/stores/dslStore'
import type { DSLFieldObject } from '@/types/dsl'

import { mutateBuilderRecipeSource } from './builderPageRoutingScopeSupport'
import type { EntityKind, Selection } from './builderPageTypes'

interface BuilderScopedEntityMutationOptions {
  recipeName: string | null
  setAddingEntity: Dispatch<SetStateAction<EntityKind | null>>
  setSelection: Dispatch<SetStateAction<Selection | null>>
}

export const useBuilderScopedEntityMutations = ({
  recipeName,
  setAddingEntity,
  setSelection,
}: BuilderScopedEntityMutationOptions) => {
  const applyRoutingMutation = useCallback(
    (mutation: (source: string) => string) => {
      const store = useDSLStore.getState()
      const nextSource = mutateBuilderRecipeSource(store.dslSource, recipeName, mutation)
      if (nextSource === store.dslSource) return
      store.setDslSource(nextSource)
      if (store.wasmReady) store.parseAST()
    },
    [recipeName],
  )

  const handleDeleteEntity = useCallback(
    (kind: EntityKind, name: string, subType?: string) => {
      switch (kind) {
        case 'signal':
          if (subType) {
            applyRoutingMutation((source) => dslMutations.deleteSignal(source, subType, name))
          }
          break
        case 'projection-partition':
          applyRoutingMutation((source) => dslMutations.deleteProjectionPartition(source, name))
          break
        case 'projection-score':
          applyRoutingMutation((source) => dslMutations.deleteProjection(source, 'score', name))
          break
        case 'projection-mapping':
          applyRoutingMutation((source) => dslMutations.deleteProjection(source, 'mapping', name))
          break
        case 'route':
          applyRoutingMutation((source) => dslMutations.deleteRoute(source, name))
          break
        case 'plugin':
          if (subType) {
            applyRoutingMutation((source) => dslMutations.deletePlugin(source, name, subType))
          }
          break
      }
      setSelection(null)
    },
    [applyRoutingMutation, setSelection],
  )

  const handleUpdateSignalFields = useCallback(
    (signalType: string, name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.updateSignal(source, signalType, name, fields))
    },
    [applyRoutingMutation],
  )

  const handleUpdatePluginFields = useCallback(
    (name: string, pluginType: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.updatePlugin(source, name, pluginType, fields))
    },
    [applyRoutingMutation],
  )

  const handleUpdateProjectionPartitionFields = useCallback(
    (name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.updateProjectionPartition(source, name, fields))
    },
    [applyRoutingMutation],
  )

  const handleUpdateProjectionScoreFields = useCallback(
    (name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.updateProjection(source, 'score', name, fields))
    },
    [applyRoutingMutation],
  )

  const handleUpdateProjectionMappingFields = useCallback(
    (name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) =>
        dslMutations.updateProjection(source, 'mapping', name, fields),
      )
    },
    [applyRoutingMutation],
  )

  const selectAddedEntity = useCallback(
    (kind: EntityKind, name: string) => {
      setSelection({ kind, name })
      setAddingEntity(null)
    },
    [setAddingEntity, setSelection],
  )

  const handleAddSignal = useCallback(
    (signalType: string, name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.addSignal(source, signalType, name, fields))
      selectAddedEntity('signal', name)
    },
    [applyRoutingMutation, selectAddedEntity],
  )

  const handleAddPlugin = useCallback(
    (name: string, pluginType: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.addPlugin(source, name, pluginType, fields))
      selectAddedEntity('plugin', name)
    },
    [applyRoutingMutation, selectAddedEntity],
  )

  const handleAddProjectionPartition = useCallback(
    (name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.addProjectionPartition(source, name, fields))
      selectAddedEntity('projection-partition', name)
    },
    [applyRoutingMutation, selectAddedEntity],
  )

  const handleAddProjectionScore = useCallback(
    (name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.addProjection(source, 'score', name, fields))
      selectAddedEntity('projection-score', name)
    },
    [applyRoutingMutation, selectAddedEntity],
  )

  const handleAddProjectionMapping = useCallback(
    (name: string, fields: DSLFieldObject) => {
      applyRoutingMutation((source) => dslMutations.addProjection(source, 'mapping', name, fields))
      selectAddedEntity('projection-mapping', name)
    },
    [applyRoutingMutation, selectAddedEntity],
  )

  const handleUpdateRoute = useCallback(
    (name: string, input: RouteInput) => {
      applyRoutingMutation((source) => dslMutations.updateRoute(source, name, input))
    },
    [applyRoutingMutation],
  )

  const handleAddRoute = useCallback(
    (name: string, input: RouteInput) => {
      applyRoutingMutation((source) => dslMutations.addRoute(source, name, input))
      selectAddedEntity('route', name)
    },
    [applyRoutingMutation, selectAddedEntity],
  )

  return {
    handleAddPlugin,
    handleAddProjectionMapping,
    handleAddProjectionPartition,
    handleAddProjectionScore,
    handleAddRoute,
    handleAddSignal,
    handleDeleteEntity,
    handleUpdatePluginFields,
    handleUpdateProjectionMappingFields,
    handleUpdateProjectionPartitionFields,
    handleUpdateProjectionScoreFields,
    handleUpdateRoute,
    handleUpdateSignalFields,
  }
}
