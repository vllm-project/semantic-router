import { useMemo, useState } from 'react'

import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import type { ConnectedModelInput } from './ConfigPageConnectModelsDialog'
import {
  buildEvaluationRecordsConfig,
  evaluationRecordsFields,
  evaluationRecordsFormData,
} from './configPageEvaluationRecordsSupport'
import {
  buildAddedModelConfig,
  buildConnectedModelsConfig,
  buildDeletedModelsConfig,
  buildEditedModelConfig,
} from './configPageModelMutations'
import {
  filterModelInventory,
  getModelDeleteBlocker,
  getModelReferenceCounts,
  getReasoningFamilyFilterOptions,
  validateModelStructuredFields,
  validateNewModelName,
  type ModelEndpointFilter,
  type ModelRoleFilter,
} from './configPageModelInventory'
import {
  editModelFormData,
  modelDialogFields,
  modelViewSections,
  newModelFormData,
} from './configPageModelsSectionSupport'
import type { ConfigPageModelsSectionProps } from './configPageModelsSectionTypes'
import {
  filterReasoningFamilyRows,
  reasoningFamilyRows,
  reasoningFamilyViewSections,
} from './configPageReasoningFamilySupport'
import type { NormalizedModel } from './configPageSupport'
import { useModelLiveVerification } from './useModelLiveVerification'

function useModelInventoryFilters(props: ConfigPageModelsSectionProps) {
  const [reasoningFamily, setReasoningFamily] = useState('all')
  const [endpointState, setEndpointState] = useState<ModelEndpointFilter>('all')
  const [role, setRole] = useState<ModelRoleFilter>('all')
  const reasoningFamilyOptions = useMemo(
    () => getReasoningFamilyFilterOptions(props.models),
    [props.models],
  )
  const referenceCounts = useMemo(() => getModelReferenceCounts(props.config), [props.config])
  const models = useMemo(
    () =>
      filterModelInventory(props.models, {
        search: props.modelsSearch,
        reasoningFamily,
        endpointState,
        role,
        defaultModel: props.defaultModel,
      }),
    [endpointState, props.defaultModel, props.models, props.modelsSearch, reasoningFamily, role],
  )
  const active = Boolean(
    props.modelsSearch.trim() ||
      reasoningFamily !== 'all' ||
      endpointState !== 'all' ||
      role !== 'all',
  )
  const clear = () => {
    props.onModelsSearchChange('')
    setReasoningFamily('all')
    setEndpointState('all')
    setRole('all')
  }
  return {
    reasoningFamily,
    setReasoningFamily,
    endpointState,
    setEndpointState,
    role,
    setRole,
    reasoningFamilyOptions,
    referenceCounts,
    models,
    active,
    clear,
  }
}

function useModelFormActions(props: ConfigPageModelsSectionProps) {
  const reasoningFamilyNames = Object.keys(props.reasoningFamilies)
  const edit = (model: NormalizedModel) => {
    props.openEditModal(
      `Edit Model: ${model.name}`,
      editModelFormData(model),
      modelDialogFields(reasoningFamilyNames, 'edit'),
      async (data) => {
        if (!props.config) return
        validateModelStructuredFields(data)
        await props.saveConfig(buildEditedModelConfig(props.config, model, data, props.isPythonCLI))
      },
      'edit',
    )
  }
  const add = () => {
    props.openEditModal(
      'Add New Model',
      newModelFormData(),
      modelDialogFields(reasoningFamilyNames, 'add'),
      async (data) => {
        if (!props.config) return
        validateModelStructuredFields(data)
        const modelName = validateNewModelName(data.model_name, props.models)
        await props.saveConfig(
          buildAddedModelConfig(props.config, modelName, data, props.isPythonCLI),
        )
      },
      'add',
    )
  }
  const view = (model: NormalizedModel) => {
    props.openViewModal(
      `Model: ${model.name}`,
      modelViewSections(model, props.defaultModel, props.isReadonly),
      () => edit(model),
    )
  }
  const connect = async (input: ConnectedModelInput) => {
    if (!props.config) return
    if (!props.isPythonCLI) {
      throw new Error('Quick connect requires the canonical providers.models configuration.')
    }
    await props.saveConfig(buildConnectedModelsConfig(props.config, props.models, input))
  }
  return { add, edit, view, connect }
}

function useModelDeletion(
  props: ConfigPageModelsSectionProps,
  referenceCounts: ReturnType<typeof getModelReferenceCounts>,
) {
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set())
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelNames, setModelNames] = useState<string[]>([])
  const blocker = (modelName: string) =>
    getModelDeleteBlocker(modelName, props.defaultModel, referenceCounts)
  const remove = async (names: string[]) => {
    if (!props.config || names.length === 0) return
    const blockedModel = names.find((modelName) => blocker(modelName))
    if (blockedModel) {
      setError(blocker(blockedModel))
      setModelNames([])
      return
    }
    setPending(true)
    setError(null)
    try {
      await props.saveConfig(
        buildDeletedModelsConfig(props.config, props.models, names, props.isPythonCLI),
      )
      setSelectedKeys((current) => {
        const next = new Set(current)
        for (const modelName of names) next.delete(modelName)
        return next
      })
      setModelNames([])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Failed to delete the selected models.')
    } finally {
      setPending(false)
    }
  }
  const requestOne = (model: NormalizedModel) => {
    const reason = blocker(model.name)
    if (reason) {
      setError(`${model.name}: ${reason}`)
      return
    }
    setError(null)
    setModelNames([model.name])
  }
  return {
    selectedKeys,
    setSelectedKeys,
    pending,
    error,
    setError,
    modelNames,
    setModelNames,
    blocker,
    remove,
    requestOne,
  }
}

function useReasoningFamilyInventory(props: ConfigPageModelsSectionProps) {
  const [search, setSearch] = useState('')
  const rows = filterReasoningFamilyRows(reasoningFamilyRows(props.reasoningFamilies), search)
  const view = (familyName: string) => {
    const family = props.reasoningFamilies[familyName]
    if (!family) return
    props.openViewModal(
      `Reasoning Family: ${familyName}`,
      reasoningFamilyViewSections(familyName, family),
    )
  }
  return { search, setSearch, rows, view }
}

function useEvaluationRecordInventory(props: ConfigPageModelsSectionProps) {
  const records = props.config?.evaluation?.records ?? []
  const manage = () => {
    if (!props.config) return
    props.openEditModal(
      'Evaluation Records',
      evaluationRecordsFormData(props.config),
      evaluationRecordsFields,
      async (data) => {
        if (!props.config) return
        await props.saveConfig(buildEvaluationRecordsConfig(props.config, data.records))
      },
      'edit',
    )
  }
  return { records, manage }
}

export function useConfigPageModelsSectionController(props: ConfigPageModelsSectionProps) {
  const filters = useModelInventoryFilters(props)
  const forms = useModelFormActions(props)
  const deletion = useModelDeletion(props, filters.referenceCounts)
  const reasoning = useReasoningFamilyInventory(props)
  const evaluations = useEvaluationRecordInventory(props)
  const [connectOpen, setConnectOpen] = useState(false)
  const liveVerification = useModelLiveVerification(props.config)
  const { catalog, error: catalogError } = useBuiltInModelCatalog()
  const toggleExpand = (model: NormalizedModel) => {
    props.onExpandedModelsChange((previous) => {
      const next = new Set(previous)
      if (next.has(model.name)) next.delete(model.name)
      else next.add(model.name)
      return next
    })
  }
  return {
    filters,
    forms,
    deletion,
    reasoning,
    evaluations,
    connect: { open: connectOpen, setOpen: setConnectOpen, catalog, catalogError },
    liveVerification,
    toggleExpand,
  }
}

export type ConfigPageModelsSectionController = ReturnType<
  typeof useConfigPageModelsSectionController
>
