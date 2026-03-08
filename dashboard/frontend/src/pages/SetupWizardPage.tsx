import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ColorBends from '../components/ColorBends'
import { useReadonly } from '../contexts/ReadonlyContext'
import { useSetup } from '../contexts/SetupContext'
import {
  countSignals,
  getRoutingPreset,
  type RoutingPresetId,
} from '../presets/routingPresets'
import { markOnboardingPending } from '../utils/onboarding'
import { activateSetupConfig, validateSetupConfig } from '../utils/setupApi'
import {
  ModelStepPanel,
  RoutingStarterPanel,
  SetupWizardStepper,
} from './SetupWizardPanels'
import { ReviewActivatePanel } from './SetupWizardReviewPanel'
import {
  buildSetupConfig,
  createModelDraft,
  getStepOneErrors,
  maskSecrets,
  PROVIDER_OPTIONS,
  type ModelDraft,
  type ProviderKind,
  type SetupActivationState,
  type SetupStep,
  type SetupValidationState,
} from './setupWizardSupport'
import styles from './SetupWizardPage.module.css'

const SetupWizardPage: React.FC = () => {
  const navigate = useNavigate()
  const { setupState, refreshSetupState } = useSetup()
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()

  const [currentStep, setCurrentStep] = useState<SetupStep>(0)
  const [models, setModels] = useState<ModelDraft[]>([createModelDraft(1)])
  const [defaultModelId, setDefaultModelId] = useState<string>('')
  const [selectedPresetId, setSelectedPresetId] = useState<RoutingPresetId | null>(null)
  const [stepOneAttempted, setStepOneAttempted] = useState(false)
  const [validationState, setValidationState] = useState<SetupValidationState>('idle')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [validatedConfig, setValidatedConfig] = useState<Record<string, unknown> | null>(null)
  const [validatedCounts, setValidatedCounts] = useState({ models: 0, decisions: 0, canActivate: false })
  const [activationState, setActivationState] = useState<SetupActivationState>('idle')
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

        <SetupWizardStepper currentStep={currentStep} onGoToStep={goToStep} />

        <div className={styles.panel}>
          {currentStep === 0 && (
            <ModelStepPanel
              currentRouteLabel={currentRouteLabel}
              models={models}
              defaultModelId={defaultModelId}
              shouldShowStepOneIssues={shouldShowStepOneIssues}
              stepOneErrors={stepOneErrors}
              stepOneAttempted={stepOneAttempted}
              draftBuildError={draftBuildError}
              onAddModel={addModel}
              onUpdateModel={updateModel}
              onRemoveModel={removeModel}
              onSelectDefaultModel={setDefaultModelId}
            />
          )}
          {currentStep === 1 && (
            <RoutingStarterPanel
              currentRouteLabel={currentRouteLabel}
              selectedPresetId={selectedPresetId}
              selectedPresetLabel={selectedPreset?.label}
              modelsCount={models.length}
              generatedDecisions={generatedDecisions}
              generatedSignals={generatedSignals}
              onSelectPreset={(presetId) => {
                setSelectedPresetId(presetId)
                setValidationState('idle')
                setValidationError(null)
                setValidatedConfig(null)
              }}
            />
          )}
          {currentStep === 2 && (
            <ReviewActivatePanel
              currentRouteLabel={currentRouteLabel}
              listenerPort={setupState?.listenerPort}
              validationState={validationState}
              validationError={validationError}
              activationError={activationError}
              validatedCounts={validatedCounts}
              modelsCount={models.length}
              generatedDecisions={generatedDecisions}
              previewSource={previewSource}
              readonlyLoading={readonlyLoading}
              isReadonly={isReadonly}
              onValidateAgain={() => void handleValidateAgain()}
            />
          )}

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
