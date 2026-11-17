import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, CornerDownLeft, Power } from 'lucide-react'
import AuthBackdrop from '../components/AuthBackdrop'
import { useReadonly } from '../contexts/ReadonlyContext'
import { useCapabilities, useCapability } from '../contexts/CapabilitiesContext'
import { isCapabilityUsable } from '../contexts/dashboardCapabilities'
import { useSetup } from '../contexts/SetupContext'
import { Button, Kbd, Progress, StatusDot } from '../ui'
import {
  activateSetupConfig,
  importRemoteSetupConfig,
  validateSetupConfig,
} from '../utils/setupApi'
import { ModelStepPanel, RoutingStarterPanel, SetupWizardStepRail } from './SetupWizardPanels'
import {
  ReviewActivatePanel,
  SetupActivationView,
  SetupSuccessView,
} from './SetupWizardReviewPanel'
import { SetupDecisionModelSection } from './SetupWizardDecisionModel'
import {
  SETUP_LAST_STEP,
  SETUP_STEP_DESCRIPTIONS,
  SETUP_STEP_LABELS,
  buildSetupConfig,
  createModelDraft,
  createSetupConfigCounts,
  createSetupRequestGuard,
  DEFAULT_REMOTE_SETUP_CONFIG_URL,
  fetchPresetDelta,
  fetchPresets,
  getStepOneErrors,
  isCpuOnlyPlatform,
  maskedSetupYaml,
  recommendedDecisionModels,
  upcomingCpuDecisionModels,
  removeSetupModel,
  restoreSetupModel,
  summarizeSetupConfig,
  switchSetupModelProvider,
  type ImportedSetupConfig,
  type ModelDraft,
  type PresetCatalogState,
  type PresetDelta,
  type PresetInfo,
  type PresetRequestState,
  type RemoteImportState,
  type RemovedModelSnapshot,
  type SetupActivationState,
  type SetupRoutingMode,
  type SetupStep,
  type SetupValidationState,
} from './setupWizardSupport'
import { type ProviderKind } from './setupWizardProviderCatalog'
import {
  DECISION_CATALOG_MODELS,
  DECISION_MODEL_OPTIONS,
  DEFAULT_DECISION_MODEL,
  configuredDecisionModel,
  withDecisionModel,
  type DecisionModelName,
} from './decisionModelSupport'

const PROVIDER_LOGO: Record<ProviderKind, string> = {
  vllm: 'vllm',
  ollama: 'ollama',
  'openai-compatible': 'openai',
  anthropic: 'anthropic',
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

/** Whether a config names its own decision model, which the wizard then keeps. */
function namesDecisionModel(config: Record<string, unknown> | null): boolean {
  const system = record(record(record(record(config).global).model_catalog).system)
  return Boolean(record(system.decision_model).deployment)
}

const SetupWizardPage: React.FC = () => {
  const navigate = useNavigate()
  const { setupState, refreshSetupState } = useSetup()
  const { isReadonly, isLoading: readonlyLoading, platform } = useReadonly()
  const { isLoading: capabilitiesLoading, error: capabilitiesError } = useCapabilities()
  const setupCapability = useCapability('setup')
  const activationBlockedReason = capabilitiesLoading
    ? 'Checking setup activation access…'
    : (capabilitiesError ??
      (!setupCapability
        ? 'Setup activation access was not reported.'
        : isCapabilityUsable(setupCapability)
          ? null
          : setupCapability.remediation || 'Setup activation is not available.'))
  const activationNotice =
    setupCapability?.state === 'degraded' ? setupCapability.remediation : null

  const [currentStep, setCurrentStep] = useState<SetupStep>(0)
  const [direction, setDirection] = useState(1)
  const [phase, setPhase] = useState<'wizard' | 'done'>('wizard')
  const [leaving, setLeaving] = useState(false)
  const [decisionModelChoice, setDecisionModelChoice] = useState<DecisionModelName | null>(null)
  const [models, setModels] = useState<ModelDraft[]>([createModelDraft(1)])
  const [defaultModelId, setDefaultModelId] = useState<string>('')
  const [routingMode, setRoutingMode] = useState<SetupRoutingMode>('scratch')
  const [remoteConfigUrl, setRemoteConfigUrl] = useState(DEFAULT_REMOTE_SETUP_CONFIG_URL)
  const [remoteImportState, setRemoteImportState] = useState<RemoteImportState>('idle')
  const [remoteImportError, setRemoteImportError] = useState<string | null>(null)
  const [importedRemoteConfig, setImportedRemoteConfig] = useState<ImportedSetupConfig | null>(null)
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [presetCatalogState, setPresetCatalogState] = useState<PresetCatalogState>('loading')
  const [presetCatalogError, setPresetCatalogError] = useState<string | null>(null)
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null)
  const [presetRequestState, setPresetRequestState] = useState<PresetRequestState>('idle')
  const [presetDelta, setPresetDelta] = useState<PresetDelta | null>(null)
  const [presetImportedConfig, setPresetImportedConfig] = useState<ImportedSetupConfig | null>(null)
  const [presetError, setPresetError] = useState<string | null>(null)
  const [stepOneAttempted, setStepOneAttempted] = useState(false)
  const [stepError, setStepError] = useState<string | null>(null)
  const [validationState, setValidationState] = useState<SetupValidationState>('idle')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [validatedConfig, setValidatedConfig] = useState<Record<string, unknown> | null>(null)
  const [validatedCounts, setValidatedCounts] = useState(createSetupConfigCounts())
  const [activationState, setActivationState] = useState<SetupActivationState>('idle')
  const [activationError, setActivationError] = useState<string | null>(null)
  const [removedModel, setRemovedModel] = useState<RemovedModelSnapshot | null>(null)
  const presetCatalogGuardRef = useRef(createSetupRequestGuard())
  const presetRequestGuardRef = useRef(createSetupRequestGuard())
  const remoteImportGuardRef = useRef(createSetupRequestGuard())
  const validationGuardRef = useRef(createSetupRequestGuard())

  const loadPresets = useCallback(async () => {
    const generation = presetCatalogGuardRef.current.begin()
    setPresetCatalogState('loading')
    setPresetCatalogError(null)
    try {
      const nextPresets = await fetchPresets()
      if (!presetCatalogGuardRef.current.isCurrent(generation)) return
      setPresets(nextPresets)
      setPresetCatalogState('ready')
    } catch (err) {
      if (!presetCatalogGuardRef.current.isCurrent(generation)) return
      setPresets([])
      setPresetCatalogState('error')
      setPresetCatalogError(
        err instanceof Error ? err.message : 'Failed to load starter architectures.',
      )
    }
  }, [])

  useEffect(() => {
    const requestGuard = presetCatalogGuardRef.current
    void loadPresets()
    return () => requestGuard.invalidate()
  }, [loadPresets])

  useEffect(
    () => () => {
      presetRequestGuardRef.current.invalidate()
      remoteImportGuardRef.current.invalidate()
      validationGuardRef.current.invalidate()
    },
    [],
  )

  useEffect(() => {
    if (models.length === 0) {
      setDefaultModelId('')
      return
    }
    if (!defaultModelId || !models.some((model) => model.id === defaultModelId)) {
      setDefaultModelId(models[0].id)
    }
  }, [models, defaultModelId])

  const stepOneErrors = getStepOneErrors(models, defaultModelId)
  const hasStepOneIssues = stepOneErrors.length > 0
  const shouldShowStepOneIssues = stepOneAttempted && hasStepOneIssues

  const resetReviewState = () => {
    validationGuardRef.current.invalidate()
    setValidationState('idle')
    setValidationError(null)
    setValidatedConfig(null)
    setValidatedCounts(createSetupConfigCounts())
    setActivationState('idle')
    setActivationError(null)
  }

  const invalidatePresetDraft = () => {
    presetRequestGuardRef.current.invalidate()
    setPresetRequestState('idle')
    setPresetError(null)
    setPresetDelta(null)
    setPresetImportedConfig(null)
  }

  let scratchConfig: Record<string, unknown> | null = null
  let scratchBuildError: string | null = null
  if (stepOneErrors.length === 0) {
    try {
      scratchConfig = buildSetupConfig(models, defaultModelId)
    } catch (err) {
      scratchBuildError = err instanceof Error ? err.message : 'Failed to build setup config.'
    }
  }

  const scratchCounts = summarizeSetupConfig(scratchConfig)
  const selectedPreset = presets.find((p) => p.id === selectedPresetId) ?? null
  const currentRouteLabel =
    routingMode === 'preset' && selectedPreset
      ? selectedPreset.label
      : routingMode === 'remote'
        ? 'From remote'
        : 'From scratch'
  const routeConfig =
    routingMode === 'preset'
      ? (presetImportedConfig?.config ?? null)
      : routingMode === 'remote'
        ? (importedRemoteConfig?.config ?? null)
        : scratchConfig

  // On a CPU-only deployment the small Vela 3.1 models are preferred when the
  // catalog has them; a route that names its own decision model keeps it.
  const cpuOnly = isCpuOnlyPlatform(platform)
  const recommendedModels = recommendedDecisionModels(
    DECISION_MODEL_OPTIONS,
    cpuOnly,
    DEFAULT_DECISION_MODEL,
  )
  const upcomingModels = cpuOnly ? upcomingCpuDecisionModels(DECISION_CATALOG_MODELS) : []
  const routeDecisionModel = configuredDecisionModel(routeConfig)
  const preferredDecisionModel =
    !namesDecisionModel(routeConfig) &&
    recommendedModels[0] &&
    recommendedModels[0].name !== DEFAULT_DECISION_MODEL
      ? recommendedModels[0].name
      : routeDecisionModel
  const selectedDecisionModel = decisionModelChoice ?? preferredDecisionModel
  const draftConfig =
    routeConfig && selectedDecisionModel !== routeDecisionModel
      ? withDecisionModel(routeConfig, selectedDecisionModel)
      : routeConfig
  const decisionModelLabel =
    DECISION_MODEL_OPTIONS.find((option) => option.name === selectedDecisionModel)?.label ??
    selectedDecisionModel
  const generatedCounts =
    routingMode === 'preset'
      ? (presetImportedConfig?.counts ?? createSetupConfigCounts())
      : routingMode === 'remote'
        ? (importedRemoteConfig?.counts ?? createSetupConfigCounts())
        : scratchCounts

  const previewYaml = maskedSetupYaml(validatedConfig ?? draftConfig)
  const validationSignature = draftConfig ? JSON.stringify(draftConfig) : ''

  const runValidation = useCallback(async (validationPayload: Record<string, unknown>) => {
    const generation = validationGuardRef.current.begin()
    setValidationState('validating')
    setValidationError(null)
    setActivationState('idle')
    setActivationError(null)
    try {
      const result = await validateSetupConfig(validationPayload)
      if (!validationGuardRef.current.isCurrent(generation)) return
      setValidatedConfig(result.config ?? validationPayload)
      setValidatedCounts({
        models: result.models,
        decisions: result.decisions,
        signals: result.signals,
        canActivate: result.canActivate,
      })
      setValidationState(result.valid ? 'valid' : 'error')
      if (!result.valid) setValidationError('The generated config needs fixes before activation.')
    } catch (err) {
      if (!validationGuardRef.current.isCurrent(generation)) return
      setValidatedConfig(null)
      setValidatedCounts(createSetupConfigCounts())
      setValidationState('error')
      setValidationError(err instanceof Error ? err.message : 'Setup validation failed.')
    }
  }, [])

  useEffect(() => {
    if (currentStep !== SETUP_LAST_STEP || !validationSignature) return
    // Scratch configs are rebuilt on every render, so key auto-validation off a
    // stable serialized payload instead of object identity.
    const validationPayload = JSON.parse(validationSignature) as Record<string, unknown>
    const requestGuard = validationGuardRef.current
    void runValidation(validationPayload)
    return () => requestGuard.invalidate()
  }, [currentStep, runValidation, validationSignature])

  const addModel = () => {
    setModels((prev) => [...prev, createModelDraft(prev.length + 1, prev)])
    setRemovedModel(null)
    invalidatePresetDraft()
    resetReviewState()
  }

  const updateModel = (id: string, field: keyof ModelDraft, value: string) => {
    setModels((prev) =>
      prev.map((model) => {
        if (model.id !== id) return model
        if (field === 'providerKind') return switchSetupModelProvider(model, value as ProviderKind)
        return { ...model, [field]: value }
      }),
    )
    setRemovedModel(null)
    setStepError(null)
    invalidatePresetDraft()
    resetReviewState()
  }

  const removeModel = (id: string) => {
    const result = removeSetupModel(models, id, defaultModelId)
    if (!result.removed) return
    setModels(result.models)
    setDefaultModelId(result.defaultModelId)
    setRemovedModel(result.removed)
    invalidatePresetDraft()
    resetReviewState()
  }

  const undoRemoveModel = () => {
    if (!removedModel) return
    const restored = restoreSetupModel(models, removedModel, defaultModelId)
    setModels(restored.models)
    setDefaultModelId(restored.defaultModelId)
    setRemovedModel(null)
    invalidatePresetDraft()
    resetReviewState()
  }

  const selectDefaultModel = (id: string) => {
    setDefaultModelId(id)
    setRemovedModel(null)
    resetReviewState()
  }

  const routingBlocked = (): string | null => {
    if (routingMode === 'remote' && !importedRemoteConfig) {
      const message =
        remoteImportState === 'importing'
          ? 'Wait for the remote config import to finish.'
          : 'Import a remote config before continuing.'
      setRemoteImportError(message)
      return message
    }
    if (routingMode === 'preset' && !presetImportedConfig) {
      const message =
        presetRequestState === 'loading' || presetRequestState === 'importing'
          ? 'Wait for the selected preset to finish preparing.'
          : 'Prepare the selected preset before continuing.'
      setPresetError(message)
      return message
    }
    return null
  }

  const goToStep = (step: SetupStep): boolean => {
    if (phase !== 'wizard') return false
    if (step > 0 && (hasStepOneIssues || scratchBuildError)) {
      setStepOneAttempted(true)
      setCurrentStep(0)
      setStepError('Finish the model setup before continuing.')
      return false
    }
    if (step >= 2) {
      const blocked = routingBlocked()
      if (blocked) {
        setCurrentStep(1)
        setStepError(blocked)
        return false
      }
    }
    setStepError(null)
    setDirection(step >= currentStep ? 1 : -1)
    setCurrentStep(step)
    return true
  }

  const handleNext = () => {
    if (currentStep < SETUP_LAST_STEP) goToStep((currentStep + 1) as SetupStep)
  }

  const handleBack = () => {
    setStepError(null)
    setDirection(-1)
    setCurrentStep((prev) => (prev === 0 ? prev : ((prev - 1) as SetupStep)))
  }

  const handleValidateAgain = async () => {
    if (!draftConfig) return
    await runValidation(draftConfig)
  }

  const handleSelectPreset = async (presetId: string) => {
    const generation = presetRequestGuardRef.current.begin()
    setSelectedPresetId(presetId)
    setPresetDelta(null)
    setPresetImportedConfig(null)
    setPresetRequestState('loading')
    setPresetError(null)
    setStepError(null)
    resetReviewState()

    const userModelNames = models.map((m) => m.name.trim()).filter((n) => n.length > 0)
    try {
      const delta = await fetchPresetDelta(presetId, userModelNames)
      if (!presetRequestGuardRef.current.isCurrent(generation)) return
      setPresetDelta(delta)
      setPresetRequestState('ready')
      setPresetError(null)

      if (delta.ready) {
        setPresetRequestState('importing')
        try {
          const result = await importRemoteSetupConfig(delta.recipe_url)
          if (!presetRequestGuardRef.current.isCurrent(generation)) return
          setPresetImportedConfig({
            config: result.config,
            sourceUrl: result.sourceUrl,
            counts: createSetupConfigCounts({
              models: result.models,
              decisions: result.decisions,
              signals: result.signals,
              canActivate: result.canActivate,
            }),
          })
          setPresetRequestState('imported')
          setPresetError(null)
        } catch (err) {
          if (!presetRequestGuardRef.current.isCurrent(generation)) return
          setPresetImportedConfig(null)
          setPresetRequestState('error')
          setPresetError(err instanceof Error ? err.message : 'Preset import failed.')
        }
      }
    } catch (err) {
      if (!presetRequestGuardRef.current.isCurrent(generation)) return
      setPresetDelta(null)
      setPresetImportedConfig(null)
      setPresetRequestState('error')
      setPresetError(
        err instanceof Error ? err.message : 'Failed to check the selected starter architecture.',
      )
    }
  }

  const handleImportPresetConfig = async () => {
    if (!presetDelta) return
    const generation = presetRequestGuardRef.current.begin()
    setPresetRequestState('importing')
    setPresetError(null)
    resetReviewState()
    try {
      const result = await importRemoteSetupConfig(presetDelta.recipe_url)
      if (!presetRequestGuardRef.current.isCurrent(generation)) return
      setPresetImportedConfig({
        config: result.config,
        sourceUrl: result.sourceUrl,
        counts: createSetupConfigCounts({
          models: result.models,
          decisions: result.decisions,
          signals: result.signals,
          canActivate: result.canActivate,
        }),
      })
      setPresetRequestState('imported')
      setPresetError(null)
    } catch (err) {
      if (!presetRequestGuardRef.current.isCurrent(generation)) return
      setPresetImportedConfig(null)
      setPresetRequestState('error')
      setPresetError(err instanceof Error ? err.message : 'Preset import failed.')
    }
  }

  const handleImportRemote = async () => {
    const trimmedUrl = remoteConfigUrl.trim()
    if (!trimmedUrl) {
      setRemoteImportState('error')
      setRemoteImportError('Paste a remote config URL before importing.')
      return
    }
    const generation = remoteImportGuardRef.current.begin()
    setRemoteImportState('importing')
    setRemoteImportError(null)
    setStepError(null)
    resetReviewState()
    try {
      const result = await importRemoteSetupConfig(trimmedUrl)
      if (!remoteImportGuardRef.current.isCurrent(generation)) return
      setImportedRemoteConfig({
        config: result.config,
        sourceUrl: result.sourceUrl,
        counts: createSetupConfigCounts({
          models: result.models,
          decisions: result.decisions,
          signals: result.signals,
          canActivate: result.canActivate,
        }),
      })
      setRemoteConfigUrl(result.sourceUrl)
      setRemoteImportState('imported')
      setRemoteImportError(null)
    } catch (err) {
      if (!remoteImportGuardRef.current.isCurrent(generation)) return
      setImportedRemoteConfig(null)
      setRemoteImportState('error')
      setRemoteImportError(err instanceof Error ? err.message : 'Remote import failed.')
    }
  }

  const handleSelectRoutingMode = (mode: SetupRoutingMode) => {
    setRoutingMode(mode)
    setRemoteImportError(null)
    setPresetError(null)
    setStepError(null)
    resetReviewState()
    if (mode !== 'remote') {
      remoteImportGuardRef.current.invalidate()
      if (remoteImportState === 'importing')
        setRemoteImportState(importedRemoteConfig ? 'imported' : 'idle')
    }
    if (mode !== 'preset') {
      presetRequestGuardRef.current.invalidate()
      if (presetRequestState === 'loading' || presetRequestState === 'importing') {
        setPresetRequestState(presetImportedConfig ? 'imported' : presetDelta ? 'ready' : 'idle')
      }
    }
  }

  const handleRemoteConfigUrlChange = (value: string) => {
    remoteImportGuardRef.current.invalidate()
    setRemoteConfigUrl(value)
    setRemoteImportError(null)
    if (importedRemoteConfig && value.trim() !== importedRemoteConfig.sourceUrl)
      setImportedRemoteConfig(null)
    if (
      remoteImportState === 'error' ||
      remoteImportState === 'importing' ||
      (importedRemoteConfig && value.trim() !== importedRemoteConfig.sourceUrl)
    ) {
      setRemoteImportState('idle')
    }
    resetReviewState()
  }

  const activateDisabled =
    validationState !== 'valid' ||
    !validatedCounts.canActivate ||
    activationState === 'activating' ||
    activationState === 'persisted' ||
    (!readonlyLoading && isReadonly) ||
    activationBlockedReason !== null

  const handleActivate = async () => {
    if (!draftConfig || validationState !== 'valid' || activateDisabled) return
    setActivationState('activating')
    setActivationError(null)
    try {
      const payload = validatedConfig ?? draftConfig
      const response = await activateSetupConfig(payload)
      if (response.status === 'persisted') {
        setActivationState('persisted')
        return
      }
      setActivationState('idle')
      setPhase('done')
    } catch (err) {
      setActivationState('error')
      setActivationError(err instanceof Error ? err.message : 'Setup activation failed.')
    }
  }

  const leaveSetup = async (to: string) => {
    setLeaving(true)
    await refreshSetupState()
    navigate(to, { replace: true })
  }

  const primary = () => {
    if (phase !== 'wizard') return
    if (currentStep < SETUP_LAST_STEP) handleNext()
    else void handleActivate()
  }
  const primaryRef = useRef(primary)
  primaryRef.current = primary

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        event.key !== 'Enter' ||
        event.defaultPrevented ||
        event.isComposing ||
        event.shiftKey ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey
      ) {
        return
      }
      const target = event.target as HTMLElement | null
      if (target?.closest('[data-choice]')) {
        event.preventDefault()
        primaryRef.current()
        return
      }
      if (
        target?.closest(
          'form, textarea, input, select, button, a, [role=dialog], [role=listbox], [role=menu], [role=combobox]',
        )
      )
        return
      event.preventDefault()
      primaryRef.current()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const finished = phase === 'done'
  const activating = activationState === 'activating'
  const percent = finished ? 100 : activating ? 90 : (currentStep / SETUP_STEP_LABELS.length) * 100
  const summaries = [
    hasStepOneIssues && stepOneAttempted
      ? 'Needs attention'
      : `${models.length} ${models.length === 1 ? 'model' : 'models'} connected`,
    currentRouteLabel,
    decisionModelLabel,
    finished
      ? 'Activated'
      : validationState === 'valid'
        ? 'Ready to activate'
        : 'Writes config.yaml',
  ]
  const modelNames =
    routingMode === 'scratch'
      ? models.map((model) => model.name.trim()).filter(Boolean)
      : (Array.isArray(record(draftConfig?.providers).models)
          ? (record(draftConfig?.providers).models as unknown[])
          : []
        )
          .map((model) => String(record(model).name ?? ''))
          .filter(Boolean)
  const providers =
    routingMode === 'scratch'
      ? [...new Set(models.map((model) => PROVIDER_LOGO[model.providerKind]))]
      : []
  const modelCount = modelNames.length || generatedCounts.models
  const defaultModel = String(
    record(record(draftConfig?.providers).defaults).model ?? modelNames[0] ?? 'auto',
  )
  const host = typeof window === 'undefined' ? 'localhost' : window.location.hostname
  const protocol = typeof window === 'undefined' ? 'http:' : window.location.protocol
  const baseUrl = setupState?.listenerPort
    ? `${protocol}//${host}:${setupState.listenerPort}/v1`
    : `${protocol}//${host}/v1`

  return (
    <AuthBackdrop>
      <motion.div
        initial={{ opacity: 0, y: 12, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="grid w-full max-w-[1080px] grid-cols-1 overflow-hidden rounded-2xl border border-border bg-card/95 shadow-xl backdrop-blur-xl md:h-[min(780px,calc(100dvh-150px))] md:min-h-[600px] md:grid-cols-[264px_minmax(0,1fr)]"
      >
        <aside className="hidden min-h-0 flex-col border-r border-border bg-surface-subtle/70 p-5 md:flex">
          <span className="px-1 text-xs font-medium text-subtle-foreground">First-run setup</span>
          <h1 className="mt-1 px-1 text-[17px] font-semibold tracking-[-0.02em]">
            Set up your router
          </h1>
          <div className="mt-5 px-1">
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">
                {finished
                  ? 'Complete'
                  : activating
                    ? 'Activating'
                    : `Step ${currentStep + 1} of ${SETUP_STEP_LABELS.length}`}
              </span>
              <span className="font-medium tabular text-foreground/80">{Math.round(percent)}%</span>
            </div>
            <Progress
              value={percent}
              tone={finished ? 'success' : 'brand'}
              aria-label="Setup progress"
            />
          </div>
          <div className="mt-5">
            <SetupWizardStepRail
              currentStep={currentStep}
              finished={finished}
              summaries={summaries}
              onGoToStep={goToStep}
            />
          </div>
          <div className="mt-auto rounded-xl border border-border bg-surface px-3 py-2.5">
            <span className="flex items-center gap-2 text-xs font-medium">
              <StatusDot tone={finished ? 'success' : 'info'} pulse />
              {finished ? 'Routing live traffic' : 'Setup mode'}
            </span>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              {finished
                ? 'Change anything later in Settings.'
                : 'The router serves no traffic until you activate a configuration.'}
            </p>
          </div>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-col">
          <div className="border-b border-border px-5 py-3 md:hidden">
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="font-medium">
                {finished ? 'Activation' : SETUP_STEP_LABELS[currentStep][1]}
              </span>
              <span className="text-muted-foreground">
                {finished ? '' : `Step ${currentStep + 1} of ${SETUP_STEP_LABELS.length}`}
              </span>
            </div>
            <Progress value={percent} />
          </div>

          {activating ? (
            <SetupActivationView />
          ) : finished ? (
            <SetupSuccessView
              baseUrl={baseUrl}
              defaultModel={defaultModel}
              summary={`${modelCount} ${modelCount === 1 ? 'model' : 'models'} · ${currentRouteLabel} · ${decisionModelLabel}`}
              leaving={leaving}
              onTest={() => void leaveSetup('/get-started')}
              onDashboard={() => void leaveSetup('/')}
            />
          ) : (
            <>
              <div className="relative min-h-0 flex-1 overflow-hidden">
                <AnimatePresence mode="wait" initial={false} custom={direction}>
                  <motion.div
                    key={currentStep}
                    custom={direction}
                    variants={{
                      enter: (dir: number) => ({ opacity: 0, x: dir * 16 }),
                      center: { opacity: 1, x: 0 },
                      exit: (dir: number) => ({ opacity: 0, x: dir * -16 }),
                    }}
                    initial="enter"
                    animate="center"
                    exit="exit"
                    transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
                    className="flex h-full flex-col"
                    id={`setup-step-${currentStep}-panel`}
                    role="region"
                    aria-labelledby={`setup-step-${currentStep}-button`}
                  >
                    <header className="px-6 pb-4 pt-6 sm:px-8">
                      <h2 className="text-lg font-semibold tracking-[-0.02em]">
                        {SETUP_STEP_LABELS[currentStep][1]}
                      </h2>
                      <p className="mt-1 max-w-xl text-[13px] text-muted-foreground">
                        {SETUP_STEP_DESCRIPTIONS[currentStep]}
                      </p>
                    </header>
                    <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6 pt-1 sm:px-8">
                      {currentStep === 0 ? (
                        <ModelStepPanel
                          models={models}
                          defaultModelId={defaultModelId}
                          shouldShowStepOneIssues={shouldShowStepOneIssues}
                          stepOneErrors={stepOneErrors}
                          stepOneAttempted={stepOneAttempted}
                          draftBuildError={scratchBuildError}
                          removedModel={removedModel?.model ?? null}
                          onAddModel={addModel}
                          onUpdateModel={updateModel}
                          onRemoveModel={removeModel}
                          onUndoRemoveModel={undoRemoveModel}
                          onSelectDefaultModel={selectDefaultModel}
                        />
                      ) : currentStep === 1 ? (
                        <RoutingStarterPanel
                          routingMode={routingMode}
                          remoteConfigUrl={remoteConfigUrl}
                          remoteImportState={remoteImportState}
                          remoteImportError={remoteImportError}
                          importedConfig={importedRemoteConfig}
                          counts={generatedCounts}
                          presets={presets}
                          presetCatalogState={presetCatalogState}
                          presetCatalogError={presetCatalogError}
                          selectedPresetId={selectedPresetId}
                          presetRequestState={presetRequestState}
                          presetDelta={presetDelta}
                          presetImportedConfig={presetImportedConfig}
                          presetError={presetError}
                          onSelectRoutingMode={handleSelectRoutingMode}
                          onChangeRemoteConfigUrl={handleRemoteConfigUrlChange}
                          onImportRemoteConfig={() => void handleImportRemote()}
                          onSelectPreset={(id) => void handleSelectPreset(id)}
                          onImportPresetConfig={() => void handleImportPresetConfig()}
                          onRetryPresets={() => void loadPresets()}
                        />
                      ) : currentStep === 2 ? (
                        <SetupDecisionModelSection
                          value={selectedDecisionModel}
                          options={DECISION_MODEL_OPTIONS}
                          recommended={recommendedModels}
                          upcoming={upcomingModels.map((option) => option.label)}
                          cpuOnly={cpuOnly}
                          onChange={(name) => {
                            setDecisionModelChoice(name)
                            resetReviewState()
                          }}
                        />
                      ) : (
                        <ReviewActivatePanel
                          routeLabel={currentRouteLabel}
                          decisionModelLabel={decisionModelLabel}
                          modelNames={modelNames}
                          providers={providers}
                          listenerPort={setupState?.listenerPort}
                          validationState={validationState}
                          validationError={validationError}
                          activationState={activationState}
                          activationError={activationError}
                          validatedCounts={validatedCounts}
                          modelsCount={generatedCounts.models}
                          generatedDecisions={generatedCounts.decisions}
                          generatedSignals={generatedCounts.signals}
                          previewYaml={previewYaml}
                          readonlyLoading={readonlyLoading}
                          isReadonly={isReadonly}
                          activationBlockedReason={activationBlockedReason}
                          activationNotice={activationNotice}
                          onValidateAgain={() => void handleValidateAgain()}
                          onEdit={(step) => goToStep(step)}
                        />
                      )}
                    </div>
                  </motion.div>
                </AnimatePresence>
              </div>
              <footer className="flex items-center gap-3 border-t border-border bg-surface-subtle/60 px-6 py-3 sm:px-8">
                <Button variant="ghost" onClick={handleBack} disabled={currentStep === 0}>
                  <ArrowLeft />
                  Back
                </Button>
                <AnimatePresence mode="wait" initial={false}>
                  {stepError ? (
                    <motion.p
                      key="error"
                      role="alert"
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: [0, -4, 4, -2, 0] }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.35 }}
                      className="truncate text-[13px] text-destructive-foreground"
                    >
                      {stepError}
                    </motion.p>
                  ) : (
                    <motion.p
                      key="hint"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="hidden items-center gap-1.5 text-xs text-subtle-foreground sm:flex"
                    >
                      Press
                      <Kbd>
                        <CornerDownLeft className="size-3" />
                      </Kbd>
                      to continue
                    </motion.p>
                  )}
                </AnimatePresence>
                {currentStep < SETUP_LAST_STEP ? (
                  <Button className="ml-auto" onClick={handleNext}>
                    Continue
                    <ArrowRight />
                  </Button>
                ) : (
                  <Button
                    className="ml-auto"
                    onClick={() => void handleActivate()}
                    disabled={activateDisabled}
                  >
                    <Power />
                    {activationState === 'persisted'
                      ? 'Saved; rollout required'
                      : 'Activate router'}
                  </Button>
                )}
              </footer>
            </>
          )}
        </section>
      </motion.div>
    </AuthBackdrop>
  )
}

export default SetupWizardPage
