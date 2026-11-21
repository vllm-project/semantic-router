import { useMemo, useState } from 'react'

import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { toast } from '../ui'
import type { ModelHealth } from '../utils/modelHealthApi'
import { buildEffectiveEvaluationGroups } from './configPageEffectiveEvaluations'
import type { ConnectedModelInput } from './ConfigPageConnectModelsDialog'
import {
  buildEvaluationRecordsConfig,
  evaluationRecordsFields,
  evaluationRecordsFormData,
} from './configPageEvaluationRecordsSupport'
import { cloneConfigData, ensureProvidersConfig } from './configPageCanonicalization'
import {
  buildAddedModelConfig,
  buildConnectedModelsConfig,
  buildDeletedModelsConfig,
  buildEditedModelConfig,
} from './configPageModelMutations'
import {
  filterModelInventoryFacets,
  getModelDeleteBlocker,
  getModelReferenceCounts,
  validateModelStructuredFields,
  validateNewModelName,
} from './configPageModelInventory'
import {
  MODEL_HEALTH_META,
  MODEL_HEALTH_ORDER,
  modelHealthStatus,
} from './configPageModelHealthPresentation'
import {
  MODEL_KIND_LABELS,
  modelDeploymentName,
  modelKind,
  modelProviderIds,
  providerDisplayName,
  type ModelKind,
} from './configPageModelPresentation'
import {
  editModelFormData,
  modelDialogFields,
  newModelFormData,
} from './configPageModelsSectionSupport'
import type { ConfigPageModelsSectionProps } from './configPageModelsSectionTypes'
import type { NormalizedModel } from './configPageSupport'
import { modelFormDataForSave } from './configPageModelFormSupport'
import { useBackendModelHealth } from './useBackendModelHealth'
import { useBackendModelTraffic } from './useBackendModelTraffic'
import { useModelLiveVerification } from './useModelLiveVerification'

const count = (models: NormalizedModel[], predicate: (model: NormalizedModel) => boolean) =>
  models.filter(predicate).length

function useModelInventoryFilters(
  props: ConfigPageModelsSectionProps,
  catalog: BuiltInModelCatalog,
  health: ReadonlyMap<string, ModelHealth>,
  healthAvailable: boolean,
) {
  const [providers, setProviders] = useState<string[]>([])
  const [kindFilter, setKindFilter] = useState<string[]>([])
  const [healthFilter, setHealthFilter] = useState<string[]>([])
  const deployments = useMemo(
    () =>
      new Map(
        props.models
          .map((model) => [model.name, modelDeploymentName(props.config, model.name)] as const)
          .filter(([, deployment]) => Boolean(deployment)),
      ),
    [props.config, props.models],
  )
  const referenceCounts = useMemo(() => getModelReferenceCounts(props.config), [props.config])
  const kinds = useMemo(
    () => new Map(props.models.map((model) => [model.name, modelKind(model, catalog)] as const)),
    [catalog, props.models],
  )
  const activeHealthFilter = useMemo(
    () => (healthAvailable ? healthFilter : []),
    [healthAvailable, healthFilter],
  )
  const models = useMemo(
    () =>
      filterModelInventoryFacets(props.models, {
        search: props.modelsSearch,
        providers,
        families: [],
        backends: [],
        roles: [],
        defaultModel: props.defaultModel,
        deployments,
      }).filter(
        (model) =>
          (kindFilter.length === 0 || kindFilter.includes(kinds.get(model.name) ?? 'chat')) &&
          (activeHealthFilter.length === 0 ||
            activeHealthFilter.includes(modelHealthStatus(health.get(model.name)))),
      ),
    [
      activeHealthFilter,
      deployments,
      health,
      kindFilter,
      kinds,
      props.defaultModel,
      props.models,
      props.modelsSearch,
      providers,
    ],
  )
  const providerOptions = useMemo(() => {
    const ids = [...new Set(props.models.flatMap(modelProviderIds))].sort()
    return ids.map((id) => ({
      value: id,
      label: providerDisplayName(id, catalog),
      count: count(props.models, (model) => modelProviderIds(model).includes(id)),
    }))
  }, [catalog, props.models])
  const kindOptions = useMemo(
    () =>
      (Object.entries(MODEL_KIND_LABELS) as Array<[ModelKind, string]>)
        .map(([value, label]) => ({
          value,
          label,
          count: count(props.models, (model) => kinds.get(model.name) === value),
        }))
        .filter((option) => option.count > 0),
    [kinds, props.models],
  )
  const healthOptions = useMemo(
    () =>
      MODEL_HEALTH_ORDER.map((value) => ({
        value,
        label: MODEL_HEALTH_META[value].label,
        count: count(props.models, (model) => modelHealthStatus(health.get(model.name)) === value),
      })).filter((option) => option.count > 0),
    [health, props.models],
  )
  const active = Boolean(
    props.modelsSearch.trim() || providers.length || kindFilter.length || activeHealthFilter.length,
  )
  const clear = () => {
    props.onModelsSearchChange('')
    setProviders([])
    setKindFilter([])
    setHealthFilter([])
  }
  return {
    providers,
    setProviders,
    providerOptions,
    kinds: kindFilter,
    setKinds: setKindFilter,
    kindOptions,
    kindByModel: kinds,
    health: activeHealthFilter,
    setHealth: setHealthFilter,
    healthOptions,
    deployments,
    referenceCounts,
    models,
    active,
    clear,
    resetKey: [props.modelsSearch, providers, kindFilter, activeHealthFilter].join('|'),
  }
}

function useModelFormActions(props: ConfigPageModelsSectionProps, catalog: BuiltInModelCatalog) {
  const reasoningFamilyNames = Object.keys(props.reasoningFamilies)
  const edit = (model: NormalizedModel) => {
    props.openEditModal(
      `Edit Model: ${model.name}`,
      editModelFormData(model),
      modelDialogFields(reasoningFamilyNames, 'edit', catalog),
      async (data) => {
        if (!props.config) return
        const submitted = modelFormDataForSave(data)
        validateModelStructuredFields(submitted)
        await props.saveConfig(buildEditedModelConfig(props.config, model, submitted))
        toast.success(`${model.name} saved`)
      },
      'edit',
    )
  }
  const add = () => {
    props.openEditModal(
      'Add New Model',
      newModelFormData(),
      modelDialogFields(reasoningFamilyNames, 'add', catalog),
      async (data) => {
        if (!props.config) return
        const submitted = modelFormDataForSave(data)
        validateModelStructuredFields(submitted)
        const modelName = validateNewModelName(submitted.model_name, props.models)
        await props.saveConfig(buildAddedModelConfig(props.config, modelName, submitted))
        toast.success(`${modelName} added`, {
          description: 'Add it to a decision to start sending it traffic.',
        })
      },
      'add',
    )
  }
  const connect = async (input: ConnectedModelInput) => {
    if (!props.config) return
    await props.saveConfig(buildConnectedModelsConfig(props.config, props.models, input))
    toast.success(
      input.modelIds.length === 1 ? 'Model connected' : `${input.modelIds.length} models connected`,
      { description: 'Add them to a decision to start sending traffic.' },
    )
  }
  return { add, edit, connect }
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
  const remove = async (names: string[], onDeleted?: (names: string[]) => void) => {
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
      await props.saveConfig(buildDeletedModelsConfig(props.config, props.models, names))
      setSelectedKeys((current) => {
        const next = new Set(current)
        for (const modelName of names) next.delete(modelName)
        return next
      })
      setModelNames([])
      onDeleted?.(names)
      toast.success(names.length === 1 ? `${names[0]} deleted` : `${names.length} models deleted`)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Failed to delete the selected models.')
      setModelNames([])
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

function useDefaultModelChange(props: ConfigPageModelsSectionProps) {
  const [target, setTarget] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const canBeDefault = (model: NormalizedModel) =>
    model.name !== props.defaultModel &&
    Boolean(props.config?.providers?.models?.some((entry) => entry.name === model.name))
  const confirm = async () => {
    if (!props.config || !target) return
    setPending(true)
    setError(null)
    try {
      const next = cloneConfigData(props.config)
      const providers = ensureProvidersConfig(next)
      providers.defaults = { ...providers.defaults, model: target }
      await props.saveConfig(next)
      toast.success(`${target} is now the default model`, {
        description: 'Requests that no decision matches go to it.',
      })
      setTarget(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The default model could not be changed.')
    } finally {
      setPending(false)
    }
  }
  return {
    target,
    request: (model: NormalizedModel) => {
      setError(null)
      setTarget(model.name)
    },
    cancel: () => {
      if (pending) return
      setTarget(null)
      setError(null)
    },
    confirm,
    pending,
    error,
    canBeDefault,
  }
}

function useEvaluationRecordInventory(
  props: ConfigPageModelsSectionProps,
  catalog: BuiltInModelCatalog,
) {
  const groups = useMemo(
    () => buildEffectiveEvaluationGroups(props.config, props.models, catalog),
    [props.config, props.models, catalog],
  )
  const manage = () => {
    if (!props.config) return
    props.openEditModal(
      'Evaluation Records',
      evaluationRecordsFormData(props.config),
      evaluationRecordsFields,
      async (data) => {
        if (!props.config) return
        await props.saveConfig(buildEvaluationRecordsConfig(props.config, data.records))
        toast.success('Evaluation records saved')
      },
      'edit',
    )
  }
  return { groups, manage }
}

export function useConfigPageModelsSectionController(props: ConfigPageModelsSectionProps) {
  const { catalog, error: catalogError, ready, loading, retry } = useBuiltInModelCatalog()
  const health = useBackendModelHealth(props.config)
  const traffic = useBackendModelTraffic()
  const filters = useModelInventoryFilters(
    props,
    catalog,
    health.byModel,
    health.phase !== 'unavailable',
  )
  const forms = useModelFormActions(props, catalog)
  const deletion = useModelDeletion(props, filters.referenceCounts)
  const defaults = useDefaultModelChange(props)
  const evaluations = useEvaluationRecordInventory(props, catalog)
  const liveVerification = useModelLiveVerification(props.config)
  return {
    catalog,
    filters,
    forms,
    deletion,
    defaults,
    evaluations: { ...evaluations, ready, loading, retry },
    catalogError,
    liveVerification,
    health,
    traffic,
  }
}

export type ConfigPageModelsSectionController = ReturnType<
  typeof useConfigPageModelsSectionController
>
