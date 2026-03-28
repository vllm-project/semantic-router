import styles from "./SetupWizardPage.module.css";
import {
  DEFAULT_REMOTE_SETUP_CONFIG_URL,
  parseBaseUrl,
  PROVIDER_OPTIONS,
  SETUP_STEP_LABELS,
  type ImportedSetupConfig,
  type ModelDraft,
  type RemoteImportState,
  type SetupConfigCounts,
  type SetupRoutingMode,
  type SetupStep,
} from "./setupWizardSupport";

interface RouteSummaryProps {
  currentRouteLabel: string;
}

interface SetupWizardStepperProps {
  currentStep: SetupStep;
  onGoToStep: (step: SetupStep) => void;
}

interface ModelStepPanelProps {
  currentRouteLabel: string;
  models: ModelDraft[];
  defaultModelId: string;
  shouldShowStepOneIssues: boolean;
  stepOneErrors: string[];
  stepOneAttempted: boolean;
  draftBuildError: string | null;
  onAddModel: () => void;
  onUpdateModel: (id: string, field: keyof ModelDraft, value: string) => void;
  onRemoveModel: (id: string) => void;
  onSelectDefaultModel: (id: string) => void;
}

interface RoutingStarterPanelProps {
  currentRouteLabel: string;
  routingMode: SetupRoutingMode;
  remoteConfigUrl: string;
  remoteImportState: RemoteImportState;
  remoteImportError: string | null;
  importedConfig: ImportedSetupConfig | null;
  counts: SetupConfigCounts;
  onSelectRoutingMode: (mode: SetupRoutingMode) => void;
  onChangeRemoteConfigUrl: (value: string) => void;
  onImportRemoteConfig: () => void;
}

export function SetupRouteSummary({ currentRouteLabel }: RouteSummaryProps) {
  return (
    <div className={styles.routeSummary}>
      <span className={styles.routeSummaryLabel}>Routing mode</span>
      <span className={styles.routeSummaryValue}>{currentRouteLabel}</span>
    </div>
  );
}

export function SetupWizardStepper({
  currentStep,
  onGoToStep,
}: SetupWizardStepperProps) {
  return (
    <div className={styles.stepper}>
      {SETUP_STEP_LABELS.map(([index, label], stepIndex) => {
        const numericStep = stepIndex as SetupStep;
        const isActive = currentStep === numericStep;
        const isDone = currentStep > numericStep;

        return (
          <button
            key={label}
            className={`${styles.stepButton} ${isActive ? styles.stepButtonActive : ""} ${isDone ? styles.stepButtonDone : ""}`}
            onClick={() => onGoToStep(numericStep)}
          >
            <span className={styles.stepNumber}>{index}</span>
            <span className={styles.stepLabel}>{label}</span>
          </button>
        );
      })}
    </div>
  );
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
            Start by registering one or more models. Routing can stay simple for
            now; setup only needs enough information to create a valid baseline
            config.
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
          const providerMeta = PROVIDER_OPTIONS.find(
            (option) => option.id === model.providerKind,
          );
          const hasNameError = shouldShowStepOneIssues && !model.name.trim();
          const hasBaseUrlError =
            shouldShowStepOneIssues &&
            (!model.baseUrl.trim() ||
              (() => {
                try {
                  parseBaseUrl(model.baseUrl, model.providerKind);
                  return false;
                } catch {
                  return true;
                }
              })());

          return (
            <div key={model.id} className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <div>
                  <div className={styles.modelCardEyebrow}>
                    Model {index + 1}
                  </div>
                  <h3 className={styles.modelCardTitle}>
                    {model.name.trim() || "New model draft"}
                  </h3>
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
                <label
                  className={`${styles.field} ${hasNameError ? styles.fieldError : ""}`}
                >
                  <span className={styles.fieldLabel}>Model name</span>
                  <input
                    value={model.name}
                    onChange={(event) =>
                      onUpdateModel(model.id, "name", event.target.value)
                    }
                    placeholder="qwen/qwen3.5-rocm"
                    aria-invalid={hasNameError}
                  />
                  {hasNameError && (
                    <span className={styles.fieldErrorText}>
                      Model name is required.
                    </span>
                  )}
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Provider</span>
                  <select
                    value={model.providerKind}
                    onChange={(event) =>
                      onUpdateModel(
                        model.id,
                        "providerKind",
                        event.target.value,
                      )
                    }
                  >
                    {PROVIDER_OPTIONS.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label
                  className={`${styles.field} ${styles.fieldWide} ${hasBaseUrlError ? styles.fieldError : ""}`}
                >
                  <span className={styles.fieldLabel}>Base URL or host</span>
                  <input
                    value={model.baseUrl}
                    onChange={(event) =>
                      onUpdateModel(model.id, "baseUrl", event.target.value)
                    }
                    placeholder={providerMeta?.placeholder}
                    aria-invalid={hasBaseUrlError}
                  />
                  <span className={styles.fieldHint}>
                    {providerMeta?.description} You can enter a full URL like{" "}
                    <code>{providerMeta?.placeholder}</code> or a host such as{" "}
                    <code>localhost:8000/v1</code>.
                  </span>
                  {hasBaseUrlError && (
                    <span className={styles.fieldErrorText}>
                      Enter a valid base URL or host before continuing.
                    </span>
                  )}
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Endpoint label</span>
                  <input
                    value={model.endpointName}
                    onChange={(event) =>
                      onUpdateModel(
                        model.id,
                        "endpointName",
                        event.target.value,
                      )
                    }
                    placeholder="primary"
                  />
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Access key</span>
                  <input
                    value={model.accessKey}
                    onChange={(event) =>
                      onUpdateModel(model.id, "accessKey", event.target.value)
                    }
                    placeholder="Optional API key"
                    type="password"
                  />
                </label>
              </div>
            </div>
          );
        })}
      </div>

      {(shouldShowStepOneIssues || (stepOneAttempted && draftBuildError)) && (
        <div className={styles.errorPanel}>
          <div className={styles.errorTitle}>
            Finish the model setup before continuing
          </div>
          <ul className={styles.errorList}>
            {shouldShowStepOneIssues &&
              stepOneErrors.map((error) => <li key={error}>{error}</li>)}
            {stepOneAttempted && draftBuildError && <li>{draftBuildError}</li>}
          </ul>
        </div>
      )}
    </div>
  );
}

export function RoutingStarterPanel({
  currentRouteLabel,
  routingMode,
  remoteConfigUrl,
  remoteImportState,
  remoteImportError,
  importedConfig,
  counts,
  onSelectRoutingMode,
  onChangeRemoteConfigUrl,
  onImportRemoteConfig,
}: RoutingStarterPanelProps) {
  const isScratchMode = routingMode === "scratch";
  const isRemoteMode = routingMode === "remote";
  const isImporting = remoteImportState === "importing";

  return (
    <div className={styles.stepBody}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeaderMain}>
          <h2 className={styles.sectionTitle}>
            Choose how routing should begin
          </h2>
          <p className={styles.sectionDescription}>
            Keep setup minimal with a default catch-all route, or import a
            remote config and carry its models, decisions, and signals into
            review.
          </p>
          <SetupRouteSummary currentRouteLabel={currentRouteLabel} />
        </div>
      </div>

      <div className={styles.presetSection}>
        <div className={styles.presetSectionHeader}>
          <div>
            <h3 className={styles.presetSectionTitle}>Routing options</h3>
            <p className={styles.presetSectionDescription}>
              `From scratch` keeps the setup on a default catch-all. `From
              remote` imports a full config file from a URL and uses that
              imported draft for review and activation.
            </p>
          </div>
          <span className={styles.presetSummaryBadge}>{currentRouteLabel}</span>
        </div>

        <div className={styles.presetGrid}>
          <button
            className={`${styles.presetCard} ${isScratchMode ? styles.presetCardActive : ""}`}
            onClick={() => onSelectRoutingMode("scratch")}
          >
            <div className={styles.presetCardHeader}>
              <h4 className={styles.presetCardTitle}>From scratch</h4>
              <span className={styles.presetCardMeta}>Default catch-all</span>
            </div>
            <p className={styles.presetCardDescription}>
              Build the first router config from the model you connected in step
              one, then evolve the routing tree after activation.
            </p>
          </button>

          <button
            className={`${styles.presetCard} ${isRemoteMode ? styles.presetCardActive : ""}`}
            onClick={() => onSelectRoutingMode("remote")}
          >
            <div className={styles.presetCardHeader}>
              <h4 className={styles.presetCardTitle}>From remote</h4>
              <span className={styles.presetCardMeta}>
                {importedConfig
                  ? `${counts.models} models · ${counts.decisions} decisions`
                  : "Import config.yaml"}
              </span>
            </div>
            <p className={styles.presetCardDescription}>
              Paste a direct YAML URL, fetch the config, and reuse its existing
              routing graph instead of starting from a blank baseline.
            </p>
          </button>
        </div>

        {isRemoteMode && (
          <div className={styles.remoteImportPanel}>
            <label
              className={`${styles.field} ${styles.fieldWide} ${remoteImportError ? styles.fieldError : ""}`}
            >
              <span className={styles.fieldLabel}>Remote config URL</span>
              <input
                value={remoteConfigUrl}
                onChange={(event) =>
                  onChangeRemoteConfigUrl(event.target.value)
                }
                placeholder={DEFAULT_REMOTE_SETUP_CONFIG_URL}
              />
              <span className={styles.fieldHint}>
                Paste a direct YAML link. The wizard fetches the file, parses
                the config, and moves that imported draft into the review step.
              </span>
              {remoteImportError && (
                <span className={styles.fieldErrorText}>
                  {remoteImportError}
                </span>
              )}
            </label>

            <div className={styles.remoteImportActions}>
              <button
                className={styles.secondaryButton}
                onClick={onImportRemoteConfig}
                disabled={isImporting}
              >
                {isImporting ? "Importing…" : "Import"}
              </button>
            </div>

            {importedConfig && (
              <div className={styles.remoteImportSummary}>
                <div className={styles.remoteImportSummaryHeader}>
                  <h4 className={styles.presetCardTitle}>
                    Remote config ready
                  </h4>
                  <span className={styles.presetCardMeta}>
                    {counts.models} models · {counts.decisions} decisions ·{" "}
                    {counts.signals} signals
                  </span>
                </div>
                <p className={styles.remoteImportSource}>
                  {importedConfig.sourceUrl}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      <div className={styles.reviewStats}>
        <div className={styles.reviewStat}>
          <span className={styles.reviewStatLabel}>Models ready</span>
          <span className={styles.reviewStatValue}>{counts.models}</span>
        </div>
        <div className={styles.reviewStat}>
          <span className={styles.reviewStatLabel}>Generated decisions</span>
          <span className={styles.reviewStatValue}>{counts.decisions}</span>
        </div>
        <div className={styles.reviewStat}>
          <span className={styles.reviewStatLabel}>Generated signals</span>
          <span className={styles.reviewStatValue}>{counts.signals}</span>
        </div>
      </div>
    </div>
  );
}
