import type { RoutingPresetId } from '../presets/routingPresets'
import { countSignals, routingPresets } from '../presets/routingPresets'
import styles from './SetupWizardPage.module.css'
import {
  parseBaseUrl,
  PROVIDER_OPTIONS,
  SETUP_STEP_LABELS,
  type ModelDraft,
  type SetupStep,
} from './setupWizardSupport'

interface RouteSummaryProps {
  currentRouteLabel: string
}

interface SetupWizardStepperProps {
  currentStep: SetupStep
  onGoToStep: (step: SetupStep) => void
}

interface ModelStepPanelProps {
  currentRouteLabel: string
  models: ModelDraft[]
  defaultModelId: string
  shouldShowStepOneIssues: boolean
  stepOneErrors: string[]
  stepOneAttempted: boolean
  draftBuildError: string | null
  onAddModel: () => void
  onUpdateModel: (id: string, field: keyof ModelDraft, value: string) => void
  onRemoveModel: (id: string) => void
  onSelectDefaultModel: (id: string) => void
}

interface RoutingStarterPanelProps {
  currentRouteLabel: string
  selectedPresetId: RoutingPresetId | null
  selectedPresetLabel?: string
  modelsCount: number
  generatedDecisions: number
  generatedSignals: number
  onSelectPreset: (presetId: RoutingPresetId | null) => void
}

export function SetupRouteSummary({ currentRouteLabel }: RouteSummaryProps) {
  return (
    <div className={styles.routeSummary}>
      <span className={styles.routeSummaryLabel}>Routing starter</span>
      <span className={styles.routeSummaryValue}>{currentRouteLabel}</span>
    </div>
  )
}

export function SetupWizardStepper({ currentStep, onGoToStep }: SetupWizardStepperProps) {
  return (
    <div className={styles.stepper}>
      {SETUP_STEP_LABELS.map(([index, label], stepIndex) => {
        const numericStep = stepIndex as SetupStep
        const isActive = currentStep === numericStep
        const isDone = currentStep > numericStep

        return (
          <button
            key={label}
            className={`${styles.stepButton} ${isActive ? styles.stepButtonActive : ''} ${isDone ? styles.stepButtonDone : ''}`}
            onClick={() => onGoToStep(numericStep)}
          >
            <span className={styles.stepNumber}>{index}</span>
            <span className={styles.stepLabel}>{label}</span>
          </button>
        )
      })}
    </div>
  )
}

export function ModelStepPanel({
  currentRouteLabel,
  models,
  defaultModelId,
  shouldShowStepOneIssues,
  stepOneErrors,
  stepOneAttempted,
  draftBuildError,
  onAddModel,
  onUpdateModel,
  onRemoveModel,
  onSelectDefaultModel,
}: ModelStepPanelProps) {
  return (
    <div className={styles.stepBody}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeaderMain}>
          <h2 className={styles.sectionTitle}>Connect your first model</h2>
          <p className={styles.sectionDescription}>
            Start by registering one or more models. Routing can stay simple for now; setup only
            needs enough information to create a valid baseline config.
          </p>
          <SetupRouteSummary currentRouteLabel={currentRouteLabel} />
        </div>
        <div className={styles.sectionHeaderAside}>
          <button className={styles.secondaryButton} onClick={onAddModel}>
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
                      onChange={() => onSelectDefaultModel(model.id)}
                    />
                    <span>Default</span>
                  </label>
                  <button
                    className={styles.ghostButton}
                    onClick={() => onRemoveModel(model.id)}
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
                    onChange={(event) => onUpdateModel(model.id, 'name', event.target.value)}
                    placeholder="openai/gpt-oss-120b"
                    aria-invalid={hasNameError}
                  />
                  {hasNameError && <span className={styles.fieldErrorText}>Model name is required.</span>}
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Provider</span>
                  <select
                    value={model.providerKind}
                    onChange={(event) => onUpdateModel(model.id, 'providerKind', event.target.value)}
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
                    onChange={(event) => onUpdateModel(model.id, 'baseUrl', event.target.value)}
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
                    onChange={(event) => onUpdateModel(model.id, 'endpointName', event.target.value)}
                    placeholder="primary"
                  />
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Access key</span>
                  <input
                    value={model.accessKey}
                    onChange={(event) => onUpdateModel(model.id, 'accessKey', event.target.value)}
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

export function RoutingStarterPanel({
  currentRouteLabel,
  selectedPresetId,
  selectedPresetLabel,
  modelsCount,
  generatedDecisions,
  generatedSignals,
  onSelectPreset,
}: RoutingStarterPanelProps) {
  return (
    <div className={styles.stepBody}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeaderMain}>
          <h2 className={styles.sectionTitle}>Choose a routing starter</h2>
          <p className={styles.sectionDescription}>
            Pick one preset below or keep `No preset` to activate with the default catch-all route only.
          </p>
          <SetupRouteSummary currentRouteLabel={currentRouteLabel} />
        </div>
      </div>

      <div className={styles.presetSection}>
        <div className={styles.presetSectionHeader}>
          <div>
            <h3 className={styles.presetSectionTitle}>Routing options</h3>
            <p className={styles.presetSectionDescription}>
              Presets add reusable signals and decisions. `No preset` keeps the setup minimal and activates only the
              default fallback route.
            </p>
          </div>
          {selectedPresetLabel && <span className={styles.presetSummaryBadge}>{selectedPresetLabel}</span>}
        </div>

        <div className={styles.presetGrid}>
          <button
            className={`${styles.presetCard} ${!selectedPresetId ? styles.presetCardActive : ''}`}
            onClick={() => onSelectPreset(null)}
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
                onClick={() => onSelectPreset(preset.id)}
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
          <span className={styles.reviewStatValue}>{modelsCount}</span>
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
