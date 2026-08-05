import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import ConfirmDialog from "../components/ConfirmDialog";
import styles from "./SetupWizardPage.module.css";
import SetupWizardPresetChecklist from "./SetupWizardPresetChecklist";
import {
  DEFAULT_REMOTE_SETUP_CONFIG_URL,
  filterSetupModels,
  getModelDraftFieldErrors,
  paginateSetupModels,
  PROVIDER_OPTIONS,
  SETUP_MODELS_PER_PAGE,
  SETUP_STEP_LABELS,
  type ImportedSetupConfig,
  type ModelDraft,
  type PresetCatalogState,
  type PresetDelta,
  type PresetInfo,
  type PresetRequestState,
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
  onGoToStep: (step: SetupStep) => boolean;
}

interface ModelStepPanelProps {
  currentRouteLabel: string;
  models: ModelDraft[];
  defaultModelId: string;
  shouldShowStepOneIssues: boolean;
  stepOneErrors: string[];
  stepOneAttempted: boolean;
  draftBuildError: string | null;
  removedModel: ModelDraft | null;
  onAddModel: () => void;
  onUpdateModel: (id: string, field: keyof ModelDraft, value: string) => void;
  onRemoveModel: (id: string) => void;
  onUndoRemoveModel: () => void;
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
  presets: PresetInfo[];
  presetCatalogState: PresetCatalogState;
  presetCatalogError: string | null;
  selectedPresetId: string | null;
  presetRequestState: PresetRequestState;
  presetDelta: PresetDelta | null;
  presetImportedConfig: ImportedSetupConfig | null;
  presetError: string | null;
  onSelectRoutingMode: (mode: SetupRoutingMode) => void;
  onChangeRemoteConfigUrl: (value: string) => void;
  onImportRemoteConfig: () => void;
  onSelectPreset: (id: string) => void;
  onImportPresetConfig: () => void;
  onRetryPresets: () => void;
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
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const focusStep = (step: number) => {
    buttonRefs.current[step]?.focus();
  };

  const handleStepKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    stepIndex: number,
  ) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      return;
    }

    event.preventDefault();
    if (event.key === "Home") {
      focusStep(0);
    } else if (event.key === "End") {
      focusStep(SETUP_STEP_LABELS.length - 1);
    } else {
      const direction = event.key === "ArrowRight" ? 1 : -1;
      focusStep(
        (stepIndex + direction + SETUP_STEP_LABELS.length) %
          SETUP_STEP_LABELS.length,
      );
    }
  };

  const activateStep = (step: SetupStep) => {
    if (!onGoToStep(step)) {
      window.requestAnimationFrame(() => focusStep(currentStep));
    }
  };

  return (
    <nav className={styles.stepper} aria-label="Setup progress">
      <ol className={styles.stepList}>
        {SETUP_STEP_LABELS.map(([index, label], stepIndex) => {
          const numericStep = stepIndex as SetupStep;
          const isActive = currentStep === numericStep;
          const isDone = currentStep > numericStep;

          return (
            <li key={label} className={styles.stepItem}>
              <button
                ref={(element) => {
                  buttonRefs.current[stepIndex] = element;
                }}
                id={`setup-step-${numericStep}-button`}
                type="button"
                className={`${styles.stepButton} ${isActive ? styles.stepButtonActive : ""} ${isDone ? styles.stepButtonDone : ""}`}
                aria-current={isActive ? "step" : undefined}
                aria-controls={`setup-step-${numericStep}-panel`}
                aria-label={`Step ${index}: ${label}${isDone ? ", complete" : ""}`}
                tabIndex={isActive ? 0 : -1}
                onKeyDown={(event) => handleStepKeyDown(event, stepIndex)}
                onClick={() => activateStep(numericStep)}
              >
                <span className={styles.stepNumber}>{isDone ? "✓" : index}</span>
                <span className={styles.stepLabel}>{label}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
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
  removedModel,
  onAddModel,
  onUpdateModel,
  onRemoveModel,
  onUndoRemoveModel,
  onSelectDefaultModel,
}: ModelStepPanelProps) {
  const [modelQuery, setModelQuery] = useState("");
  const [modelPageNumber, setModelPageNumber] = useState(1);
  const [pendingRemoveModel, setPendingRemoveModel] =
    useState<ModelDraft | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const validationFocusHandledRef = useRef(false);
  const fieldErrorsByModelId = getModelDraftFieldErrors(models);
  const filteredModels = filterSetupModels(models, modelQuery);
  const modelPage = paginateSetupModels(filteredModels, modelPageNumber);

  useEffect(() => {
    if (modelPage.page !== modelPageNumber) {
      setModelPageNumber(modelPage.page);
    }
  }, [modelPage.page, modelPageNumber]);

  useEffect(() => {
    const shouldFocusValidation =
      shouldShowStepOneIssues || Boolean(stepOneAttempted && draftBuildError);
    if (!shouldFocusValidation) {
      validationFocusHandledRef.current = false;
      return;
    }
    if (validationFocusHandledRef.current) {
      return;
    }
    validationFocusHandledRef.current = true;
    const currentFieldErrors = getModelDraftFieldErrors(models);
    const firstInvalidModelIndex = models.findIndex((model) => {
      const errors = currentFieldErrors[model.id];
      return Boolean(errors?.name || errors?.baseUrl);
    });
    if (firstInvalidModelIndex >= 0) {
      setModelQuery("");
      setModelPageNumber(
        Math.floor(firstInvalidModelIndex / SETUP_MODELS_PER_PAGE) + 1,
      );
    }
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        panelRef.current
          ?.querySelector<HTMLElement>('[aria-invalid="true"]')
          ?.focus();
      });
    });
  }, [draftBuildError, models, shouldShowStepOneIssues, stepOneAttempted]);

  const handleAddModel = () => {
    setModelQuery("");
    setModelPageNumber(
      Math.max(1, Math.ceil((models.length + 1) / SETUP_MODELS_PER_PAGE)),
    );
    onAddModel();
  };

  return (
    <div ref={panelRef} className={styles.stepBody}>
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
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={handleAddModel}
          >
            Add model
          </button>
        </div>
      </div>

      <div className={styles.modelCollectionToolbar}>
        <label className={styles.modelSearchField}>
          <span className={styles.fieldLabel}>Find a model</span>
          <input
            type="search"
            value={modelQuery}
            onChange={(event) => {
              setModelQuery(event.target.value);
              setModelPageNumber(1);
            }}
            placeholder="Search name, provider, or endpoint"
          />
        </label>
        <p className={styles.modelCount} aria-live="polite">
          {modelQuery.trim()
            ? `${filteredModels.length} of ${models.length} models`
            : `${models.length} model${models.length === 1 ? "" : "s"}`}
        </p>
      </div>

      {removedModel && (
        <div className={styles.undoNotice} role="status" aria-live="polite">
          <span>
            Removed <strong>{removedModel.name || "model draft"}</strong>.
          </span>
          <button
            type="button"
            className={styles.undoButton}
            onClick={onUndoRemoveModel}
          >
            Undo
          </button>
        </div>
      )}

      <div className={styles.modelList} aria-label="Connected models">
        {modelPage.items.map((model) => {
          const index = models.findIndex((candidate) => candidate.id === model.id);
          const providerMeta = PROVIDER_OPTIONS.find(
            (option) => option.id === model.providerKind,
          );
          const fieldErrors = fieldErrorsByModelId[model.id] ?? {};
          const nameError = shouldShowStepOneIssues
            ? fieldErrors.name
            : undefined;
          const baseUrlError = shouldShowStepOneIssues
            ? fieldErrors.baseUrl
            : undefined;
          const hasNameError = Boolean(nameError);
          const hasBaseUrlError = Boolean(baseUrlError);

          return (
            <article key={model.id} className={styles.modelCard}>
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
                    type="button"
                    className={styles.ghostButton}
                    aria-label={`Remove model ${model.name.trim() || index + 1}`}
                    onClick={() => setPendingRemoveModel(model)}
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
                    aria-describedby={
                      hasNameError ? `${model.id}-name-error` : undefined
                    }
                  />
                  {hasNameError && (
                    <span
                      id={`${model.id}-name-error`}
                      className={styles.fieldErrorText}
                    >
                      {nameError}
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
                    aria-describedby={
                      hasBaseUrlError ? `${model.id}-base-url-error` : undefined
                    }
                  />
                  <span className={styles.fieldHint}>
                    {providerMeta?.description} You can enter a full URL like{" "}
                    <code>{providerMeta?.placeholder}</code> or a host such as{" "}
                    <code>localhost:8000/v1</code>.
                  </span>
                  {hasBaseUrlError && (
                    <span
                      id={`${model.id}-base-url-error`}
                      className={styles.fieldErrorText}
                    >
                      {baseUrlError}
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
            </article>
          );
        })}

        {modelPage.total === 0 && (
          <div className={styles.modelEmptyState} role="status">
            No models match “{modelQuery.trim()}”. Clear the search to return to
            the full inventory.
          </div>
        )}
      </div>

      {modelPage.pageCount > 1 && (
        <nav className={styles.modelPagination} aria-label="Model pages">
          <button
            type="button"
            className={styles.ghostButton}
            onClick={() => setModelPageNumber((page) => Math.max(1, page - 1))}
            disabled={modelPage.page === 1}
          >
            Previous
          </button>
          <span aria-live="polite">
            Page {modelPage.page} of {modelPage.pageCount}
          </span>
          <button
            type="button"
            className={styles.ghostButton}
            onClick={() =>
              setModelPageNumber((page) => Math.min(modelPage.pageCount, page + 1))
            }
            disabled={modelPage.page === modelPage.pageCount}
          >
            Next models
          </button>
        </nav>
      )}

      {(shouldShowStepOneIssues || (stepOneAttempted && draftBuildError)) && (
        <div className={styles.errorPanel} role="alert">
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

      <ConfirmDialog
        isOpen={Boolean(pendingRemoveModel)}
        eyebrow="Model inventory"
        title="Remove this model?"
        description={
          <p>
            {pendingRemoveModel?.name || "This model draft"} will be removed from
            the setup configuration. You can undo immediately after removal.
          </p>
        }
        confirmLabel="Remove model"
        tone="danger"
        onCancel={() => setPendingRemoveModel(null)}
        onConfirm={() => {
          if (pendingRemoveModel) {
            onRemoveModel(pendingRemoveModel.id);
          }
          setPendingRemoveModel(null);
        }}
      />
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
  presets,
  presetCatalogState,
  presetCatalogError,
  selectedPresetId,
  presetRequestState,
  presetDelta,
  presetImportedConfig,
  presetError,
  onSelectRoutingMode,
  onChangeRemoteConfigUrl,
  onImportRemoteConfig,
  onSelectPreset,
  onImportPresetConfig,
  onRetryPresets,
}: RoutingStarterPanelProps) {
  const isScratchMode = routingMode === "scratch";
  const isRemoteMode = routingMode === "remote";
  const isPresetMode = routingMode === "preset";
  const isImporting = remoteImportState === "importing";

  return (
    <div className={styles.stepBody}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeaderMain}>
          <h2 className={styles.sectionTitle}>
            Choose how routing should begin
          </h2>
          <p className={styles.sectionDescription}>
            Pick a goal-oriented mode, keep setup minimal with a default
            catch-all route, or import a remote config.
          </p>
          <SetupRouteSummary currentRouteLabel={currentRouteLabel} />
        </div>
      </div>

      <div className={styles.presetSection}>
        <div className={styles.presetSectionHeader}>
          <div>
            <h3 className={styles.presetSectionTitle}>Routing options</h3>
            <p className={styles.presetSectionDescription}>
              Choose a built-in mode to get a curated routing config, start from
              scratch with a default catch-all, or import a full config from a
              URL.
            </p>
          </div>
          <span className={styles.presetSummaryBadge}>{currentRouteLabel}</span>
        </div>

        <div className={styles.presetGrid}>
          <button
            type="button"
            className={`${styles.presetCard} ${isScratchMode ? styles.presetCardActive : ""}`}
            aria-pressed={isScratchMode}
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

          {presets.map((preset) => {
            const isActive =
              isPresetMode && selectedPresetId === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                className={`${styles.presetCard} ${isActive ? styles.presetCardActive : ""}`}
                aria-pressed={isActive}
                onClick={() => {
                  onSelectRoutingMode("preset");
                  onSelectPreset(preset.id);
                }}
              >
                <div className={styles.presetCardHeader}>
                  <h4 className={styles.presetCardTitle}>{preset.label}</h4>
                  <span className={styles.presetCardMeta}>
                    {preset.required_models.length} model
                    {preset.required_models.length === 1 ? "" : "s"}
                  </span>
                </div>
                <p className={styles.presetCardDescription}>
                  {preset.summary}
                </p>
              </button>
            );
          })}

          <button
            type="button"
            className={`${styles.presetCard} ${isRemoteMode ? styles.presetCardActive : ""}`}
            aria-pressed={isRemoteMode}
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

        {presetCatalogState === "loading" && (
          <div className={styles.asyncNotice} role="status">
            Loading starter architectures…
          </div>
        )}

        {presetCatalogState === "error" && (
          <div className={styles.errorPanel} role="alert">
            <div className={styles.errorTitle}>Starter architectures unavailable</div>
            <p className={styles.errorText}>{presetCatalogError}</p>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={onRetryPresets}
            >
              Retry presets
            </button>
          </div>
        )}

        {isPresetMode && presetRequestState === "loading" && (
          <div className={styles.asyncNotice} role="status">
            Checking the selected architecture against your model inventory…
          </div>
        )}

        {isPresetMode &&
          selectedPresetId &&
          presetRequestState === "idle" &&
          !presetDelta && (
            <div className={styles.asyncNotice} role="status">
              <span>The model inventory changed. Recheck this preset before continuing.</span>
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => onSelectPreset(selectedPresetId)}
              >
                Recheck preset
              </button>
            </div>
          )}

        {isPresetMode && presetDelta && (
          <div
            className={styles.remoteImportPanel}
            aria-busy={presetRequestState === "importing"}
          >
            <SetupWizardPresetChecklist
              presetDelta={presetDelta}
              presetImportedConfig={presetImportedConfig}
              counts={counts}
              presetError={presetError}
              presetRequestState={presetRequestState}
              onImportPresetConfig={onImportPresetConfig}
            />
          </div>
        )}

        {isPresetMode && !presetDelta && presetError && (
          <div className={styles.remoteImportPanel}>
            <p className={styles.fieldErrorText} role="alert">
              {presetError}
            </p>
            {selectedPresetId && (
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => onSelectPreset(selectedPresetId)}
              >
                Retry selected preset
              </button>
            )}
          </div>
        )}

        {isRemoteMode && (
          <div className={styles.remoteImportPanel} aria-busy={isImporting}>
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
                aria-invalid={Boolean(remoteImportError)}
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
                type="button"
                className={styles.secondaryButton}
                onClick={onImportRemoteConfig}
                disabled={isImporting}
              >
                {isImporting
                  ? "Importing…"
                  : remoteImportState === "error"
                    ? "Retry import"
                    : "Import"}
              </button>
            </div>

            {isImporting && (
              <div className={styles.asyncNotice} role="status">
                Fetching and parsing the remote configuration…
              </div>
            )}

            {importedConfig && (
              <div className={styles.remoteImportSummary} role="status">
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
