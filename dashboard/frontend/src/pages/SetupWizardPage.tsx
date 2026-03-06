import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ColorBends from '../components/ColorBends'
import { useReadonly } from '../contexts/ReadonlyContext'
import { useSetup } from '../contexts/SetupContext'
import {
  countSignals,
  getRoutingPreset,
  routingPresets,
  type RoutingPresetId,
} from '../presets/routingPresets'
import { markOnboardingPending } from '../utils/onboarding'
import { activateSetupConfig, validateSetupConfig } from '../utils/setupApi'
import styles from './SetupWizardPage.module.css'

type SetupStep = 0 | 1 | 2
type ProviderKind = 'vllm' | 'openai-compatible' | 'anthropic'

interface ModelDraft {
  id: string
  name: string
  providerKind: ProviderKind
  baseUrl: string
  accessKey: string
  endpointName: string
}

interface BuiltModel {
  name: string
  endpoints: Array<{
    name: string
    weight: number
    endpoint: string
    protocol: 'http' | 'https'
  }>
  access_key?: string
  api_format?: 'anthropic'
}

const PROVIDER_OPTIONS: Array<{
  id: ProviderKind
  label: string
  description: string
  placeholder: string
}> = [
  {
    id: 'vllm',
    label: 'Local vLLM',
    description: 'Best for first-run with a local or self-hosted OpenAI-compatible endpoint.',
    placeholder: 'http://vllm-gpt-oss-120b:8000',
  },
  {
    id: 'openai-compatible',
    label: 'OpenAI-compatible API',
    description: 'Works for hosted endpoints that expose the OpenAI chat/completions surface.',
    placeholder: 'https://api.openai.com',
  },
  {
    id: 'anthropic',
    label: 'Anthropic Messages API',
    description: 'Uses Anthropic-compatible request translation inside the router.',
    placeholder: 'https://api.anthropic.com',
  },
]

function createModelDraft(seed: number): ModelDraft {
  return {
    id: `model-${Date.now()}-${seed}`,
    name: 'openai/gpt-oss-120b',
    providerKind: 'vllm',
    baseUrl: 'http://vllm-gpt-oss-120b:8000',
    accessKey: '',
    endpointName: 'primary',
  }
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function inferProtocol(endpoint: string, providerKind: ProviderKind): 'http' | 'https' {
  if (providerKind === 'anthropic') {
    return 'https'
  }

  if (
    endpoint.startsWith('localhost') ||
    endpoint.startsWith('127.0.0.1') ||
    endpoint.startsWith('0.0.0.0') ||
    endpoint.startsWith('host.docker.internal')
  ) {
    return 'http'
  }

  if (endpoint.includes(':80')) {
    return 'http'
  }

  return 'https'
}

function parseBaseUrl(rawValue: string, providerKind: ProviderKind): {
  protocol: 'http' | 'https'
  endpoint: string
} {
  const trimmed = rawValue.trim().replace(/\/$/, '')
  if (!trimmed) {
    throw new Error('Model base URL is required.')
  }

  const normalized = trimmed.includes('://')
    ? trimmed
    : `${inferProtocol(trimmed, providerKind)}://${trimmed}`

  let parsed: URL
  try {
    parsed = new URL(normalized)
  } catch {
    throw new Error(`Invalid model endpoint: ${rawValue}`)
  }

  const protocol = parsed.protocol.replace(':', '')
  if (protocol !== 'http' && protocol !== 'https') {
    throw new Error(`Unsupported protocol for model endpoint: ${rawValue}`)
  }

  const path = parsed.pathname && parsed.pathname !== '/' ? parsed.pathname.replace(/\/$/, '') : ''
  return {
    protocol,
    endpoint: `${parsed.host}${path}`,
  }
}

function getStepOneErrors(models: ModelDraft[], defaultModelId: string): string[] {
  const errors: string[] = []

  if (models.length === 0) {
    errors.push('Add at least one model before continuing.')
    return errors
  }

  const names = new Set<string>()
  let hasDefault = false

  models.forEach((model, index) => {
    const position = index + 1
    const trimmedName = model.name.trim()

    if (!trimmedName) {
      errors.push(`Model ${position} is missing a model name.`)
    } else {
      const normalizedName = trimmedName.toLowerCase()
      if (names.has(normalizedName)) {
        errors.push(`Model name "${trimmedName}" is duplicated.`)
      }
      names.add(normalizedName)
    }

    if (!model.baseUrl.trim()) {
      errors.push(`Model ${position} is missing a base URL.`)
    } else {
      try {
        parseBaseUrl(model.baseUrl, model.providerKind)
      } catch (err) {
        errors.push(err instanceof Error ? err.message : `Model ${position} has an invalid base URL.`)
      }
    }

    if (model.id === defaultModelId) {
      hasDefault = true
    }
  })

  if (!hasDefault) {
    errors.push('Choose a default model before continuing.')
  }

  return errors
}

function mergeSignals(
  presetSignals?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  if (!presetSignals) {
    return undefined
  }

  const mergedEntries = Object.entries(presetSignals).filter(([, value]) => Array.isArray(value) && value.length > 0)
  if (mergedEntries.length === 0) {
    return undefined
  }

  return Object.fromEntries(mergedEntries)
}

function buildSetupConfig(
  models: ModelDraft[],
  defaultModelId: string,
  selectedPresetId: RoutingPresetId | null,
): Record<string, unknown> {
  const builtModels: BuiltModel[] = models.map((model, index) => {
    const { protocol, endpoint } = parseBaseUrl(model.baseUrl, model.providerKind)
    const endpointName = model.endpointName.trim() || `${slugify(model.name) || `model-${index + 1}`}-primary`

    return {
      name: model.name.trim(),
      endpoints: [
        {
          name: endpointName,
          weight: 100,
          endpoint,
          protocol,
        },
      ],
      access_key: model.accessKey.trim() || undefined,
      api_format: model.providerKind === 'anthropic' ? 'anthropic' : undefined,
    }
  })

  const defaultModel = builtModels.find((model) => {
    const draft = models.find((item) => item.id === defaultModelId)
    return draft?.name.trim() === model.name
  })

  if (!defaultModel) {
    throw new Error('Default model selection is invalid.')
  }

  const preset = selectedPresetId ? getRoutingPreset(selectedPresetId) : null
  const presetFragment = preset ? preset.build(defaultModel.name) : null

  const catchAllDecision = {
    name: 'default-route',
    description: 'Generated during setup to route all requests to the default model.',
    priority: 100,
    rules: {
      operator: 'AND',
      conditions: [],
    },
    modelRefs: [
      {
        model: defaultModel.name,
        use_reasoning: false,
      },
    ],
  }

  const config: Record<string, unknown> = {
    providers: {
      models: builtModels,
      default_model: defaultModel.name,
    },
    decisions: [...(presetFragment?.decisions ?? []), catchAllDecision],
  }

  const signals = mergeSignals(presetFragment?.signals as Record<string, unknown> | undefined)
  if (signals) {
    config.signals = signals
  }

  return config
}

function maskSecrets(config: Record<string, unknown> | null): string {
  if (!config) {
    return ''
  }

  const serialized = JSON.stringify(
    config,
    (key, value) => {
      if (key === 'access_key' && typeof value === 'string' && value.length > 0) {
        return '••••••••'
      }
      return value
    },
    2,
  )

  return serialized
}

const SetupWizardPage: React.FC = () => {
  const navigate = useNavigate()
  const { setupState, refreshSetupState } = useSetup()
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()

  const [currentStep, setCurrentStep] = useState<SetupStep>(0)
  const [models, setModels] = useState<ModelDraft[]>([createModelDraft(1)])
  const [defaultModelId, setDefaultModelId] = useState<string>('')
  const [selectedPresetId, setSelectedPresetId] = useState<RoutingPresetId | null>(null)
  const [stepOneAttempted, setStepOneAttempted] = useState(false)
  const [validationState, setValidationState] = useState<'idle' | 'validating' | 'valid' | 'error'>('idle')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [validatedConfig, setValidatedConfig] = useState<Record<string, unknown> | null>(null)
  const [validatedCounts, setValidatedCounts] = useState({ models: 0, decisions: 0, canActivate: false })
  const [activationState, setActivationState] = useState<'idle' | 'activating' | 'error'>('idle')
  const [activationError, setActivationError] = useState<string | null>(null)

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

  let draftConfig: Record<string, unknown> | null = null
  let draftBuildError: string | null = null
  const selectedPreset = selectedPresetId ? getRoutingPreset(selectedPresetId) : null
  const currentRouteLabel = selectedPreset?.label || 'Default catch-all'
  if (stepOneErrors.length === 0) {
    try {
      draftConfig = buildSetupConfig(models, defaultModelId, selectedPresetId)
    } catch (err) {
      draftBuildError = err instanceof Error ? err.message : 'Failed to build setup config.'
    }
  }
  const generatedSignals = countSignals(draftConfig?.signals as Parameters<typeof countSignals>[0] | undefined)
  const generatedDecisions = Array.isArray(draftConfig?.decisions) ? draftConfig.decisions.length : 0

  const previewSource = maskSecrets(validatedConfig ?? draftConfig)
  const validationSignature = draftConfig ? JSON.stringify(draftConfig) : ''

  useEffect(() => {
    if (currentStep !== 2 || !draftConfig) {
      return
    }

    let cancelled = false

    const runValidation = async () => {
      setValidationState('validating')
      setValidationError(null)
      setActivationError(null)

      try {
        const result = await validateSetupConfig(draftConfig)
        if (cancelled) {
          return
        }

        setValidatedConfig(result.config ?? draftConfig)
        setValidatedCounts({
          models: result.models,
          decisions: result.decisions,
          canActivate: result.canActivate,
        })
        setValidationState(result.valid ? 'valid' : 'error')
      } catch (err) {
        if (cancelled) {
          return
        }

        setValidatedConfig(null)
        setValidatedCounts({ models: 0, decisions: 0, canActivate: false })
        setValidationState('error')
        setValidationError(err instanceof Error ? err.message : 'Setup validation failed.')
      }
    }

    void runValidation()

    return () => {
      cancelled = true
    }
  }, [currentStep, validationSignature])

  const addModel = () => {
    setModels((prev) => [...prev, createModelDraft(prev.length + 1)])
  }

  const updateModel = (id: string, field: keyof ModelDraft, value: string) => {
    setModels((prev) =>
      prev.map((model) => {
        if (model.id !== id) {
          return model
        }

        if (field === 'providerKind') {
          const nextProvider = value as ProviderKind
          const nextPlaceholder = PROVIDER_OPTIONS.find((option) => option.id === nextProvider)?.placeholder
          return {
            ...model,
            providerKind: nextProvider,
            baseUrl: model.baseUrl.trim() ? model.baseUrl : nextPlaceholder || model.baseUrl,
          }
        }

        return { ...model, [field]: value }
      }),
    )

    setValidationState('idle')
    setValidationError(null)
    setValidatedConfig(null)
  }

  const removeModel = (id: string) => {
    setModels((prev) => prev.filter((model) => model.id !== id))
    setValidationState('idle')
    setValidationError(null)
    setValidatedConfig(null)
  }

  const goToStep = (step: SetupStep) => {
    if (currentStep === 0 && step > 0 && (hasStepOneIssues || draftBuildError)) {
      setStepOneAttempted(true)
      return
    }
    setCurrentStep(step)
  }

  const handleNext = () => {
    if (currentStep === 0) {
      if (hasStepOneIssues || draftBuildError) {
        setStepOneAttempted(true)
        return
      }

      setCurrentStep(1)
      return
    }

    setCurrentStep((prev) => (prev === 2 ? prev : ((prev + 1) as SetupStep)))
  }

  const handleBack = () => {
    setCurrentStep((prev) => (prev === 0 ? prev : ((prev - 1) as SetupStep)))
  }

  const handleValidateAgain = async () => {
    if (!draftConfig) {
      return
    }

    setValidationState('validating')
    setValidationError(null)

    try {
      const result = await validateSetupConfig(draftConfig)
      setValidatedConfig(result.config ?? draftConfig)
      setValidatedCounts({
        models: result.models,
        decisions: result.decisions,
        canActivate: result.canActivate,
      })
      setValidationState(result.valid ? 'valid' : 'error')
    } catch (err) {
      setValidatedConfig(null)
      setValidatedCounts({ models: 0, decisions: 0, canActivate: false })
      setValidationState('error')
      setValidationError(err instanceof Error ? err.message : 'Setup validation failed.')
    }
  }

  const handleActivate = async () => {
    if (!draftConfig || validationState !== 'valid') {
      return
    }

    setActivationState('activating')
    setActivationError(null)

    try {
      const payload = validatedConfig ?? draftConfig
      await activateSetupConfig(payload)
      markOnboardingPending()
      await refreshSetupState()
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setActivationState('error')
      setActivationError(err instanceof Error ? err.message : 'Setup activation failed.')
    }
  }

  const renderRouteSummary = () => (
    <div className={styles.routeSummary}>
      <span className={styles.routeSummaryLabel}>Routing starter</span>
      <span className={styles.routeSummaryValue}>{currentRouteLabel}</span>
    </div>
  )

  const renderStepContent = () => {
    if (currentStep === 0) {
      return (
        <div className={styles.stepBody}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionHeaderMain}>
              <h2 className={styles.sectionTitle}>Connect your first model</h2>
              <p className={styles.sectionDescription}>
                Start by registering one or more models. Routing can stay simple for now; setup only
                needs enough information to create a valid baseline config.
              </p>
              {renderRouteSummary()}
            </div>
            <div className={styles.sectionHeaderAside}>
              <button className={styles.secondaryButton} onClick={addModel}>
                Add model
              </button>
            </div>
          </div>

          <div className={styles.modelList}>
            {models.map((model, index) => {
              const providerMeta = PROVIDER_OPTIONS.find((option) => option.id === model.providerKind)
              const hasNameError = shouldShowStepOneIssues && !model.name.trim()
              const hasBaseUrlError =
                shouldShowStepOneIssues &&
                (!model.baseUrl.trim() || (() => {
                  try {
                    parseBaseUrl(model.baseUrl, model.providerKind)
                    return false
                  } catch {
                    return true
                  }
                })())

              return (
                <div key={model.id} className={styles.modelCard}>
                  <div className={styles.modelCardHeader}>
                    <div>
                      <div className={styles.modelCardEyebrow}>Model {index + 1}</div>
                      <h3 className={styles.modelCardTitle}>{model.name.trim() || 'New model draft'}</h3>
                    </div>
                    <div className={styles.modelCardActions}>
                      <label className={styles.defaultToggle}>
                        <input
                          type="radio"
                          name="default-model"
                          checked={defaultModelId === model.id}
                          onChange={() => setDefaultModelId(model.id)}
                        />
                        <span>Default</span>
                      </label>
                      <button
                        className={styles.ghostButton}
                        onClick={() => removeModel(model.id)}
                        disabled={models.length === 1}
                      >
                        Remove
                      </button>
                    </div>
                  </div>

                  <div className={styles.formGrid}>
                    <label className={`${styles.field} ${hasNameError ? styles.fieldError : ''}`}>
                      <span className={styles.fieldLabel}>Model name</span>
                      <input
                        value={model.name}
                        onChange={(event) => updateModel(model.id, 'name', event.target.value)}
                        placeholder="openai/gpt-oss-120b"
                        aria-invalid={hasNameError}
                      />
                      {hasNameError && <span className={styles.fieldErrorText}>Model name is required.</span>}
                    </label>

                    <label className={styles.field}>
                      <span className={styles.fieldLabel}>Provider</span>
                      <select
                        value={model.providerKind}
                        onChange={(event) => updateModel(model.id, 'providerKind', event.target.value)}
                      >
                        {PROVIDER_OPTIONS.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className={`${styles.field} ${styles.fieldWide} ${hasBaseUrlError ? styles.fieldError : ''}`}>
                      <span className={styles.fieldLabel}>Base URL or host</span>
                      <input
                        value={model.baseUrl}
                        onChange={(event) => updateModel(model.id, 'baseUrl', event.target.value)}
                        placeholder={providerMeta?.placeholder}
                        aria-invalid={hasBaseUrlError}
                      />
                      <span className={styles.fieldHint}>
                        {providerMeta?.description} You can enter a full URL like{' '}
                        <code>{providerMeta?.placeholder}</code> or a host such as <code>localhost:8000/v1</code>.
                      </span>
                      {hasBaseUrlError && (
                        <span className={styles.fieldErrorText}>Enter a valid base URL or host before continuing.</span>
                      )}
                    </label>

                    <label className={styles.field}>
                      <span className={styles.fieldLabel}>Endpoint label</span>
                      <input
                        value={model.endpointName}
                        onChange={(event) => updateModel(model.id, 'endpointName', event.target.value)}
                        placeholder="primary"
                      />
                    </label>

                    <label className={styles.field}>
                      <span className={styles.fieldLabel}>Access key</span>
                      <input
                        value={model.accessKey}
                        onChange={(event) => updateModel(model.id, 'accessKey', event.target.value)}
                        placeholder="Optional API key"
                        type="password"
                      />
                    </label>
                  </div>
                </div>
              )
            })}
          </div>

          {(shouldShowStepOneIssues || (stepOneAttempted && draftBuildError)) && (
            <div className={styles.errorPanel}>
              <div className={styles.errorTitle}>Finish the model setup before continuing</div>
              <ul className={styles.errorList}>
                {shouldShowStepOneIssues && stepOneErrors.map((error) => (
                  <li key={error}>{error}</li>
                ))}
                {stepOneAttempted && draftBuildError && <li>{draftBuildError}</li>}
              </ul>
            </div>
          )}
        </div>
      )
    }

    if (currentStep === 1) {
      return (
        <div className={styles.stepBody}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionHeaderMain}>
              <h2 className={styles.sectionTitle}>Choose a routing starter</h2>
              <p className={styles.sectionDescription}>
                Pick one preset below or keep `No preset` to activate with the default catch-all
                route only.
              </p>
              {renderRouteSummary()}
            </div>
          </div>

          <div className={styles.presetSection}>
            <div className={styles.presetSectionHeader}>
              <div>
                <h3 className={styles.presetSectionTitle}>Routing options</h3>
                <p className={styles.presetSectionDescription}>
                  Presets add reusable signals and decisions. `No preset` keeps the setup minimal
                  and activates only the default fallback route.
                </p>
              </div>
              {selectedPreset && <span className={styles.presetSummaryBadge}>{selectedPreset.label}</span>}
            </div>

            <div className={styles.presetGrid}>
              <button
                className={`${styles.presetCard} ${!selectedPresetId ? styles.presetCardActive : ''}`}
                onClick={() => {
                  setSelectedPresetId(null)
                  setValidationState('idle')
                  setValidationError(null)
                  setValidatedConfig(null)
                }}
              >
                <div className={styles.presetCardHeader}>
                  <h4 className={styles.presetCardTitle}>No preset</h4>
                  <span className={styles.presetCardMeta}>Default</span>
                </div>
                <p className={styles.presetCardDescription}>
                  Activate the default catch-all route now and build the rest of the routing tree later.
                </p>
              </button>

              {routingPresets.map((preset) => {
                const fragment = preset.build('preview-model')
                const isActive = selectedPresetId === preset.id

                return (
                  <button
                    key={preset.id}
                    className={`${styles.presetCard} ${isActive ? styles.presetCardActive : ''}`}
                    onClick={() => {
                      setSelectedPresetId(preset.id)
                      setValidationState('idle')
                      setValidationError(null)
                      setValidatedConfig(null)
                    }}
                  >
                    <div className={styles.presetCardHeader}>
                      <h4 className={styles.presetCardTitle}>{preset.label}</h4>
                      <span className={styles.presetCardMeta}>
                        {countSignals(fragment.signals)} signals · {fragment.decisions.length} decisions
                      </span>
                    </div>
                    <p className={styles.presetCardDescription}>{preset.description}</p>
                  </button>
                )
              })}
            </div>
          </div>

          <div className={styles.reviewStats}>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Models ready</span>
              <span className={styles.reviewStatValue}>{models.length}</span>
            </div>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Generated decisions</span>
              <span className={styles.reviewStatValue}>{generatedDecisions}</span>
            </div>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Generated signals</span>
              <span className={styles.reviewStatValue}>{generatedSignals}</span>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className={styles.stepBody}>
        <div className={styles.sectionHeader}>
          <div className={styles.sectionHeaderMain}>
            <h2 className={styles.sectionTitle}>Review and activate</h2>
            <p className={styles.sectionDescription}>
              The dashboard validates the generated config before activation. Activation writes the
              resulting YAML to <code>config.yaml</code> and exits setup mode.
            </p>
            {renderRouteSummary()}
          </div>
          <div className={styles.sectionHeaderAside}>
            <button className={styles.secondaryButton} onClick={() => void handleValidateAgain()}>
              Revalidate
            </button>
          </div>
        </div>

        <div className={styles.reviewGrid}>
          <div className={styles.reviewCard}>
            <div className={styles.reviewCardHeader}>
              <h3 className={styles.reviewCardTitle}>Activation summary</h3>
              <span
                className={`${styles.statusPill} ${
                  validationState === 'valid'
                    ? styles.statusSuccess
                    : validationState === 'error'
                      ? styles.statusError
                      : styles.statusPending
                }`}
              >
                {validationState === 'validating' && 'Validating'}
                {validationState === 'valid' && 'Ready'}
                {validationState === 'error' && 'Needs fixes'}
                {validationState === 'idle' && 'Pending'}
              </span>
            </div>

            <div className={styles.reviewStats}>
              <div className={styles.reviewStat}>
                <span className={styles.reviewStatLabel}>Listener</span>
                <span className={styles.reviewStatValue}>
                  {setupState?.listenerPort ? `:${setupState.listenerPort}` : 'Bootstrap'}
                </span>
              </div>
              <div className={styles.reviewStat}>
                <span className={styles.reviewStatLabel}>Models</span>
                <span className={styles.reviewStatValue}>
                  {validatedCounts.models || models.length}
                </span>
              </div>
              <div className={styles.reviewStat}>
                <span className={styles.reviewStatLabel}>Decisions</span>
                <span className={styles.reviewStatValue}>
                  {validatedCounts.decisions || generatedDecisions}
                </span>
              </div>
            </div>

            {validationError && (
              <div className={styles.errorPanel}>
                <div className={styles.errorTitle}>Validation failed</div>
                <p className={styles.errorText}>{validationError}</p>
              </div>
            )}

            {activationError && (
              <div className={styles.errorPanel}>
                <div className={styles.errorTitle}>Activation failed</div>
                <p className={styles.errorText}>{activationError}</p>
              </div>
            )}

            {!readonlyLoading && isReadonly && (
              <div className={styles.readonlyPanel}>
                This dashboard is running in read-only mode. Review is available, but activation is disabled.
              </div>
            )}
          </div>

          <div className={styles.previewCard}>
            <div className={styles.previewHeader}>
              <h3 className={styles.reviewCardTitle}>Generated config preview</h3>
              <span className={styles.previewHint}>Secrets are masked in preview.</span>
            </div>
            <pre className={styles.previewCode}>{previewSource}</pre>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <div className={styles.backgroundEffect}>
        <ColorBends
          colors={['#76b900', '#8fd400', '#1f2c10', '#d5f080']}
          rotation={20}
          speed={0.2}
          scale={1}
          frequency={1}
          warpStrength={1}
          mouseInfluence={1}
          parallax={0.5}
          noise={0.08}
          transparent
          autoRotate={0.8}
        />
      </div>

      <div className={styles.content}>
        <div className={styles.hero}>
          <div className={styles.heroHeader}>
            <div className={styles.heroBadge}>First-run setup</div>
          </div>
          <div className={styles.heroTitleRow}>
            <div className={styles.heroLogoWrap} aria-hidden="true">
              <img className={styles.heroLogo} src="/vllm.png" alt="" />
            </div>
            <h1 className={styles.heroTitle}>Configure a model first. Routing can follow.</h1>
          </div>
          <p className={styles.heroDescription}>
            Extract signals. Compose decisions. Route the best model.
          </p>
        </div>

        <div className={styles.stepper}>
          {[
            ['1', 'Connect model'],
            ['2', 'Choose routing'],
            ['3', 'Review & activate'],
          ].map(([index, label], stepIndex) => {
            const numericStep = stepIndex as SetupStep
            const isActive = currentStep === numericStep
            const isDone = currentStep > numericStep

            return (
              <button
                key={label}
                className={`${styles.stepButton} ${isActive ? styles.stepButtonActive : ''} ${isDone ? styles.stepButtonDone : ''}`}
                onClick={() => goToStep(numericStep)}
              >
                <span className={styles.stepNumber}>{index}</span>
                <span className={styles.stepLabel}>{label}</span>
              </button>
            )
          })}
        </div>

        <div className={styles.panel}>
          {renderStepContent()}

          <div className={styles.footer}>
            <div className={styles.footerActions}>
              {currentStep > 0 && (
                <button className={styles.secondaryButton} onClick={handleBack}>
                  Back
                </button>
              )}
              {currentStep < 2 && (
                <button className={styles.primaryButton} onClick={handleNext}>
                  Next
                </button>
              )}
              {currentStep === 2 && (
                <button
                  className={styles.primaryButton}
                  onClick={() => void handleActivate()}
                  disabled={
                    validationState !== 'valid' ||
                    !validatedCounts.canActivate ||
                    activationState === 'activating' ||
                    (!readonlyLoading && isReadonly)
                  }
                >
                  {activationState === 'activating' ? 'Activating…' : 'Activate'}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SetupWizardPage
