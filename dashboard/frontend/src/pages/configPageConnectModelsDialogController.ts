import {
  useEffect,
  useId,
  useMemo,
  useState,
  type Dispatch,
  type RefObject,
  type SetStateAction,
} from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  emptyConnectModelAdvancedValues,
  resolveConnectedModelName,
  type ConnectModelAdvancedValues,
} from './configPageConnectModelSupport'
import type { ConnectedModelInput } from './configPageConnectModelsDialogTypes'
import type { ModelPricing, ProviderReliability, RoutingModelCard } from './configPageSupport'
import { modelProviderPresetsFromCatalog, type ModelProviderPreset } from './modelProviderCatalog'

interface DiscoveryResponse {
  models?: unknown
  error?: unknown
}

type Stage = 'provider' | 'models'

interface DialogState {
  titleId: string
  stage: Stage
  setStage: Dispatch<SetStateAction<Stage>>
  search: string
  setSearch: Dispatch<SetStateAction<string>>
  provider: ModelProviderPreset | null
  setProvider: Dispatch<SetStateAction<ModelProviderPreset | null>>
  baseUrl: string
  setBaseUrl: Dispatch<SetStateAction<string>>
  apiKey: string
  setAPIKey: Dispatch<SetStateAction<string>>
  models: string[]
  setModels: Dispatch<SetStateAction<string[]>>
  selected: Set<string>
  setSelected: Dispatch<SetStateAction<Set<string>>>
  modelSearch: string
  setModelSearch: Dispatch<SetStateAction<string>>
  manualModel: string
  setManualModel: Dispatch<SetStateAction<string>>
  advanced: ConnectModelAdvancedValues
  setAdvanced: Dispatch<SetStateAction<ConnectModelAdvancedValues>>
  discovering: boolean
  setDiscovering: Dispatch<SetStateAction<boolean>>
  saving: boolean
  setSaving: Dispatch<SetStateAction<boolean>>
  error: string | null
  setError: Dispatch<SetStateAction<string | null>>
  busy: boolean
  dialogRef: RefObject<HTMLDivElement>
}

function useConnectDialogState(isOpen: boolean, onClose: () => void): DialogState {
  const titleId = useId()
  const [stage, setStage] = useState<Stage>('provider')
  const [search, setSearch] = useState('')
  const [provider, setProvider] = useState<ModelProviderPreset | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setAPIKey] = useState('')
  const [models, setModels] = useState<string[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [modelSearch, setModelSearch] = useState('')
  const [manualModel, setManualModel] = useState('')
  const [advanced, setAdvanced] = useState(emptyConnectModelAdvancedValues)
  const [discovering, setDiscovering] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const busy = discovering || saving
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen, onClose, dismissible: !busy })

  useEffect(() => {
    if (!isOpen) return
    setStage('provider')
    setSearch('')
    setProvider(null)
    setBaseUrl('')
    setAPIKey('')
    setModels([])
    setSelected(new Set())
    setModelSearch('')
    setManualModel('')
    setAdvanced(emptyConnectModelAdvancedValues())
    setError(null)
  }, [isOpen])

  return {
    titleId,
    stage,
    setStage,
    search,
    setSearch,
    provider,
    setProvider,
    baseUrl,
    setBaseUrl,
    apiKey,
    setAPIKey,
    models,
    setModels,
    selected,
    setSelected,
    modelSearch,
    setModelSearch,
    manualModel,
    setManualModel,
    advanced,
    setAdvanced,
    discovering,
    setDiscovering,
    saving,
    setSaving,
    error,
    setError,
    busy,
    dialogRef,
  }
}

function useConnectDialogDerived(
  state: DialogState,
  existingModelNames: string[],
  catalog: BuiltInModelCatalog,
) {
  const providerCatalog = useMemo(
    () => modelProviderPresetsFromCatalog(catalog.providers),
    [catalog.providers],
  )
  const visibleProviders = useMemo(() => {
    const query = state.search.trim().toLocaleLowerCase()
    if (!query) return providerCatalog
    return providerCatalog.filter((item) =>
      `${item.name} ${item.description}`.toLocaleLowerCase().includes(query),
    )
  }, [providerCatalog, state.search])
  const existing = useMemo(() => new Set(existingModelNames), [existingModelNames])
  const visibleModels = useMemo(() => {
    const query = state.modelSearch.trim().toLocaleLowerCase()
    if (!query) return state.models
    return state.models.filter((model) => model.toLocaleLowerCase().includes(query))
  }, [state.modelSearch, state.models])
  const resolvedModelNames = useMemo(() => {
    if (!state.provider) return new Map<string, string>()
    const occupied = new Set(existing)
    const resolved = new Map<string, string>()
    for (const model of state.models) {
      const name = resolveConnectedModelName(
        state.advanced.namePrefix,
        state.provider.id,
        model,
        occupied,
      )
      resolved.set(model, name)
      occupied.add(name)
    }
    return resolved
  }, [existing, state.advanced.namePrefix, state.models, state.provider])
  const catalogModels = useMemo(
    () => providerCatalogModels(catalog, state.provider?.id),
    [catalog, state.provider?.id],
  )
  const modelDisplayNames = useMemo(
    () => new Map(catalog.models.map((model) => [model.id, model.display_name])),
    [catalog.models],
  )
  return { visibleProviders, visibleModels, resolvedModelNames, catalogModels, modelDisplayNames }
}

function providerCatalogModels(catalog: BuiltInModelCatalog, providerID?: string) {
  const result = new Map<string, string>()
  if (!providerID) return result
  for (const offering of catalog.offerings) {
    if (
      offering.provider === providerID &&
      offering.lifecycle !== 'removed' &&
      !result.has(offering.provider_model_id)
    ) {
      result.set(offering.provider_model_id, offering.model)
    }
  }
  return result
}

export function useConnectModelsDialogController(
  isOpen: boolean,
  existingModelNames: string[],
  catalog: BuiltInModelCatalog,
  onClose: () => void,
  onImport: (input: ConnectedModelInput) => Promise<void>,
) {
  const state = useConnectDialogState(isOpen, onClose)
  const derived = useConnectDialogDerived(state, existingModelNames, catalog)
  return {
    ...state,
    ...derived,
    chooseProvider: (provider: ModelProviderPreset) => chooseProvider(state, provider),
    discover: () => discoverModels(state),
    addManualModel: () => addManualModel(state),
    submit: () => submitModels(state, derived, onImport, onClose),
  }
}

export type ConnectModelsDialogController = ReturnType<typeof useConnectModelsDialogController>

function chooseProvider(state: DialogState, provider: ModelProviderPreset): void {
  state.setProvider(provider)
  state.setBaseUrl(provider.baseUrl)
  state.setAPIKey('')
  state.setModels([])
  state.setSelected(new Set())
  state.setModelSearch('')
  state.setError(null)
  state.setStage('models')
}

async function discoverModels(state: DialogState): Promise<void> {
  const { provider } = state
  if (!provider || !state.baseUrl.trim()) {
    state.setError('Enter the provider base URL.')
    return
  }
  if (provider.baseUrl && provider.authStrategy !== 'none' && !state.apiKey.trim()) {
    state.setError('Enter your API key to connect this provider.')
    return
  }
  state.setDiscovering(true)
  state.setError(null)
  try {
    const response = await fetch('/api/models/discover', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        baseUrl: state.baseUrl.trim(),
        apiKey: state.apiKey.trim(),
        provider: provider.id,
      }),
    })
    const payload = (await response.json()) as DiscoveryResponse
    if (!response.ok) {
      throw new Error(
        typeof payload.error === 'string' ? payload.error : 'Models could not be loaded.',
      )
    }
    const discovered = Array.isArray(payload.models)
      ? payload.models.filter(
          (model): model is string => typeof model === 'string' && model.trim() !== '',
        )
      : []
    state.setModels(discovered)
    state.setSelected(new Set(discovered.length === 1 ? discovered : []))
    if (discovered.length === 0) state.setError('No models were returned. Add a model ID manually.')
  } catch (cause) {
    state.setModels([])
    state.setSelected(new Set())
    state.setError(cause instanceof Error ? cause.message : 'Models could not be loaded.')
  } finally {
    state.setDiscovering(false)
  }
}

function addManualModel(state: DialogState): void {
  const model = state.manualModel.trim()
  if (!model) return
  if (!state.models.includes(model)) state.setModels((current) => [...current, model].sort())
  state.setSelected((current) => new Set(current).add(model))
  state.setManualModel('')
  state.setError(null)
}

async function submitModels(
  state: DialogState,
  derived: ReturnType<typeof useConnectDialogDerived>,
  onImport: (input: ConnectedModelInput) => Promise<void>,
  onClose: () => void,
): Promise<void> {
  if (!state.provider || state.selected.size === 0) {
    state.setError('Choose at least one model.')
    return
  }
  state.setSaving(true)
  state.setError(null)
  try {
    await onImport({
      provider: state.provider,
      baseUrl: state.baseUrl.trim(),
      apiKey: state.apiKey.trim(),
      modelIds: [...state.selected],
      modelNames: Object.fromEntries(
        [...state.selected].map((modelID) => [
          modelID,
          derived.resolvedModelNames.get(modelID) ?? modelID,
        ]),
      ),
      catalogModels: Object.fromEntries(
        [...state.selected]
          .map((modelID) => [modelID, derived.catalogModels.get(modelID)] as const)
          .filter((entry): entry is readonly [string, string] => Boolean(entry[1])),
      ),
      ...advancedInput(state.advanced),
    })
    onClose()
  } catch (cause) {
    state.setError(cause instanceof Error ? cause.message : 'Models could not be added.')
  } finally {
    state.setSaving(false)
  }
}

const listValues = (value: string) => [
  ...new Set(
    value
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean),
  ),
]

const optionalNumber = (value: string): number | undefined => {
  if (!value.trim()) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function advancedInput(advanced: ConnectModelAdvancedValues) {
  const inputCost = optionalNumber(advanced.inputCost)
  const costConfigured = [
    advanced.inputCost,
    advanced.outputCost,
    advanced.cacheReadCost,
    advanced.cacheWriteCost,
  ].some((value) => value.trim())
  const pricing: ModelPricing | undefined = costConfigured
    ? {
        currency: 'USD',
        prompt_per_1m: inputCost,
        completion_per_1m: optionalNumber(advanced.outputCost),
        cached_input_per_1m: optionalNumber(advanced.cacheReadCost) ?? inputCost,
        cache_write_per_1m: optionalNumber(advanced.cacheWriteCost) ?? inputCost,
      }
    : undefined
  const reliability: ProviderReliability = {
    retry_count: optionalNumber(advanced.maxRetries),
    retry_on: advanced.retryOn.trim() || undefined,
    lb_policy: advanced.loadBalancing || undefined,
    health_check_path: advanced.healthCheckPath.trim() || undefined,
    health_check_interval: advanced.healthCheckInterval.trim() || undefined,
    health_check_timeout: advanced.healthCheckTimeout.trim() || undefined,
  }
  return {
    namePrefix: advanced.namePrefix,
    reasoningFamily: advanced.reasoningFamily || undefined,
    metadata: {
      description: advanced.description.trim() || undefined,
      modality: advanced.modality || undefined,
      param_size: advanced.parameterSize.trim() || undefined,
      context_window_size: optionalNumber(advanced.contextWindow),
      capabilities: listValues(advanced.capabilities),
      tags: listValues(advanced.tags),
    } satisfies Omit<RoutingModelCard, 'name'>,
    pricing,
    reliability: Object.values(reliability).some((value) => value !== undefined && value !== '')
      ? reliability
      : undefined,
  }
}
