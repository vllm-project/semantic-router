import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useReadonly } from "../contexts/ReadonlyContext";
import { useSetup } from "../contexts/SetupContext";
import { markOnboardingPending } from "../utils/onboarding";
import {
  activateSetupConfig,
  importRemoteSetupConfig,
  validateSetupConfig,
} from "../utils/setupApi";
import {
  ModelStepPanel,
  RoutingStarterPanel,
  SetupWizardStepper,
} from "./SetupWizardPanels";
import { ReviewActivatePanel } from "./SetupWizardReviewPanel";
import SetupWizardBackground from "./SetupWizardBackground";
import {
  buildSetupConfig,
  createModelDraft,
  createSetupRequestGuard,
  createSetupConfigCounts,
  DEFAULT_REMOTE_SETUP_CONFIG_URL,
  fetchPresetDelta,
  fetchPresets,
  getStepOneErrors,
  maskSecrets,
  PROVIDER_OPTIONS,
  removeSetupModel,
  restoreSetupModel,
  summarizeSetupConfig,
  type ImportedSetupConfig,
  type ModelDraft,
  type PresetCatalogState,
  type PresetDelta,
  type PresetInfo,
  type PresetRequestState,
  type ProviderKind,
  type RemoteImportState,
  type RemovedModelSnapshot,
  type SetupActivationState,
  type SetupRoutingMode,
  type SetupStep,
  type SetupValidationState,
} from "./setupWizardSupport";
import styles from "./SetupWizardPage.module.css";

const SetupWizardPage: React.FC = () => {
  const navigate = useNavigate();
  const { setupState, refreshSetupState } = useSetup();
  const { isReadonly, isLoading: readonlyLoading } = useReadonly();

  const [currentStep, setCurrentStep] = useState<SetupStep>(0);
  const [models, setModels] = useState<ModelDraft[]>([createModelDraft(1)]);
  const [defaultModelId, setDefaultModelId] = useState<string>("");
  const [routingMode, setRoutingMode] = useState<SetupRoutingMode>("scratch");
  const [remoteConfigUrl, setRemoteConfigUrl] = useState(
    DEFAULT_REMOTE_SETUP_CONFIG_URL,
  );
  const [remoteImportState, setRemoteImportState] =
    useState<RemoteImportState>("idle");
  const [remoteImportError, setRemoteImportError] = useState<string | null>(
    null,
  );
  const [importedRemoteConfig, setImportedRemoteConfig] =
    useState<ImportedSetupConfig | null>(null);
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [presetCatalogState, setPresetCatalogState] =
    useState<PresetCatalogState>("loading");
  const [presetCatalogError, setPresetCatalogError] = useState<string | null>(
    null,
  );
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const [presetRequestState, setPresetRequestState] =
    useState<PresetRequestState>("idle");
  const [presetDelta, setPresetDelta] = useState<PresetDelta | null>(null);
  const [presetImportedConfig, setPresetImportedConfig] =
    useState<ImportedSetupConfig | null>(null);
  const [presetError, setPresetError] = useState<string | null>(null);
  const [stepOneAttempted, setStepOneAttempted] = useState(false);
  const [validationState, setValidationState] =
    useState<SetupValidationState>("idle");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validatedConfig, setValidatedConfig] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [validatedCounts, setValidatedCounts] = useState(
    createSetupConfigCounts(),
  );
  const [activationState, setActivationState] =
    useState<SetupActivationState>("idle");
  const [activationError, setActivationError] = useState<string | null>(null);
  const [removedModel, setRemovedModel] =
    useState<RemovedModelSnapshot | null>(null);
  const presetCatalogGuardRef = useRef(createSetupRequestGuard());
  const presetRequestGuardRef = useRef(createSetupRequestGuard());
  const remoteImportGuardRef = useRef(createSetupRequestGuard());
  const validationGuardRef = useRef(createSetupRequestGuard());

  const loadPresets = useCallback(async () => {
    const generation = presetCatalogGuardRef.current.begin();
    setPresetCatalogState("loading");
    setPresetCatalogError(null);

    try {
      const nextPresets = await fetchPresets();
      if (!presetCatalogGuardRef.current.isCurrent(generation)) {
        return;
      }
      setPresets(nextPresets);
      setPresetCatalogState("ready");
    } catch (err) {
      if (!presetCatalogGuardRef.current.isCurrent(generation)) {
        return;
      }
      setPresets([]);
      setPresetCatalogState("error");
      setPresetCatalogError(
        err instanceof Error ? err.message : "Failed to load starter architectures.",
      );
    }
  }, []);

  useEffect(() => {
    const requestGuard = presetCatalogGuardRef.current;
    void loadPresets();
    return () => requestGuard.invalidate();
  }, [loadPresets]);

  useEffect(
    () => () => {
      presetRequestGuardRef.current.invalidate();
      remoteImportGuardRef.current.invalidate();
      validationGuardRef.current.invalidate();
    },
    [],
  );

  useEffect(() => {
    if (models.length === 0) {
      setDefaultModelId("");
      return;
    }

    if (
      !defaultModelId ||
      !models.some((model) => model.id === defaultModelId)
    ) {
      setDefaultModelId(models[0].id);
    }
  }, [models, defaultModelId]);

  const stepOneErrors = getStepOneErrors(models, defaultModelId);
  const hasStepOneIssues = stepOneErrors.length > 0;
  const shouldShowStepOneIssues = stepOneAttempted && hasStepOneIssues;

  const resetReviewState = () => {
    validationGuardRef.current.invalidate();
    setValidationState("idle");
    setValidationError(null);
    setValidatedConfig(null);
    setValidatedCounts(createSetupConfigCounts());
    setActivationState("idle");
    setActivationError(null);
  };

  const invalidatePresetDraft = () => {
    presetRequestGuardRef.current.invalidate();
    setPresetRequestState("idle");
    setPresetError(null);
    setPresetDelta(null);
    setPresetImportedConfig(null);
  };

  let scratchConfig: Record<string, unknown> | null = null;
  let scratchBuildError: string | null = null;
  if (stepOneErrors.length === 0) {
    try {
      scratchConfig = buildSetupConfig(models, defaultModelId);
    } catch (err) {
      scratchBuildError =
        err instanceof Error ? err.message : "Failed to build setup config.";
    }
  }

  const scratchCounts = summarizeSetupConfig(scratchConfig);

  const selectedPreset = presets.find((p) => p.id === selectedPresetId) ?? null;
  const currentRouteLabel =
    routingMode === "preset" && selectedPreset
      ? selectedPreset.label
      : routingMode === "remote"
        ? "From remote"
        : "From scratch";
  const draftConfig =
    routingMode === "preset"
      ? (presetImportedConfig?.config ?? null)
      : routingMode === "remote"
        ? (importedRemoteConfig?.config ?? null)
        : scratchConfig;
  const generatedCounts =
    routingMode === "preset"
      ? (presetImportedConfig?.counts ?? createSetupConfigCounts())
      : routingMode === "remote"
        ? (importedRemoteConfig?.counts ?? createSetupConfigCounts())
        : scratchCounts;

  const previewSource = maskSecrets(validatedConfig ?? draftConfig);
  const validationSignature = draftConfig ? JSON.stringify(draftConfig) : "";

  const runValidation = useCallback(
    async (validationPayload: Record<string, unknown>) => {
      const generation = validationGuardRef.current.begin();
      setValidationState("validating");
      setValidationError(null);
      setActivationState("idle");
      setActivationError(null);

      try {
        const result = await validateSetupConfig(validationPayload);
        if (!validationGuardRef.current.isCurrent(generation)) {
          return;
        }

        setValidatedConfig(result.config ?? validationPayload);
        setValidatedCounts({
          models: result.models,
          decisions: result.decisions,
          signals: result.signals,
          canActivate: result.canActivate,
        });
        setValidationState(result.valid ? "valid" : "error");
        if (!result.valid) {
          setValidationError("The generated config needs fixes before activation.");
        }
      } catch (err) {
        if (!validationGuardRef.current.isCurrent(generation)) {
          return;
        }

        setValidatedConfig(null);
        setValidatedCounts(createSetupConfigCounts());
        setValidationState("error");
        setValidationError(
          err instanceof Error ? err.message : "Setup validation failed.",
        );
      }
    },
    [],
  );

  useEffect(() => {
    if (currentStep !== 2 || !validationSignature) {
      return;
    }

    // Scratch configs are rebuilt on every render, so key auto-validation off a
    // stable serialized payload instead of object identity.
    const validationPayload = JSON.parse(validationSignature) as Record<
      string,
      unknown
    >;
    const requestGuard = validationGuardRef.current;

    void runValidation(validationPayload);

    return () => {
      requestGuard.invalidate();
    };
  }, [currentStep, runValidation, validationSignature]);

  const addModel = () => {
    setModels((prev) => [...prev, createModelDraft(prev.length + 1, prev)]);
    setRemovedModel(null);
    invalidatePresetDraft();
    resetReviewState();
  };

  const updateModel = (id: string, field: keyof ModelDraft, value: string) => {
    setModels((prev) =>
      prev.map((model) => {
        if (model.id !== id) {
          return model;
        }

        if (field === "providerKind") {
          const nextProvider = value as ProviderKind;
          const nextPlaceholder = PROVIDER_OPTIONS.find(
            (option) => option.id === nextProvider,
          )?.placeholder;
          return {
            ...model,
            providerKind: nextProvider,
            baseUrl: model.baseUrl.trim()
              ? model.baseUrl
              : nextPlaceholder || model.baseUrl,
          };
        }

        return { ...model, [field]: value };
      }),
    );

    setRemovedModel(null);
    invalidatePresetDraft();
    resetReviewState();
  };

  const removeModel = (id: string) => {
    const result = removeSetupModel(models, id, defaultModelId);
    if (!result.removed) {
      return;
    }

    setModels(result.models);
    setDefaultModelId(result.defaultModelId);
    setRemovedModel(result.removed);
    invalidatePresetDraft();
    resetReviewState();
  };

  const undoRemoveModel = () => {
    if (!removedModel) {
      return;
    }

    const restored = restoreSetupModel(models, removedModel, defaultModelId);
    setModels(restored.models);
    setDefaultModelId(restored.defaultModelId);
    setRemovedModel(null);
    invalidatePresetDraft();
    resetReviewState();
  };

  const selectDefaultModel = (id: string) => {
    setDefaultModelId(id);
    setRemovedModel(null);
    resetReviewState();
  };

  const isStep2Blocked = () => {
    if (routingMode === "remote" && !importedRemoteConfig) {
      setRemoteImportError(
        remoteImportState === "importing"
          ? "Wait for the remote config import to finish."
          : "Import a remote config before continuing.",
      );
      return true;
    }
    if (routingMode === "preset" && !presetImportedConfig) {
      setPresetError(
        presetRequestState === "loading" || presetRequestState === "importing"
          ? "Wait for the selected preset to finish preparing."
          : "Prepare the selected preset before continuing.",
      );
      return true;
    }
    return false;
  };

  const goToStep = (step: SetupStep): boolean => {
    if (step > 0 && (hasStepOneIssues || scratchBuildError)) {
      setStepOneAttempted(true);
      return false;
    }

    if (step === 2 && isStep2Blocked()) {
      return false;
    }

    setCurrentStep(step);
    return true;
  };

  const handleNext = () => {
    if (currentStep < 2) {
      goToStep((currentStep + 1) as SetupStep);
    }
  };

  const handleBack = () => {
    setCurrentStep((prev) => (prev === 0 ? prev : ((prev - 1) as SetupStep)));
  };

  const handleValidateAgain = async () => {
    if (!draftConfig) {
      return;
    }

    await runValidation(draftConfig);
  };

  const handleSelectPreset = async (presetId: string) => {
    const generation = presetRequestGuardRef.current.begin();
    setSelectedPresetId(presetId);
    setPresetDelta(null);
    setPresetImportedConfig(null);
    setPresetRequestState("loading");
    setPresetError(null);
    resetReviewState();

    const userModelNames = models
      .map((m) => m.name.trim())
      .filter((n) => n.length > 0);
    try {
      const delta = await fetchPresetDelta(presetId, userModelNames);
      if (!presetRequestGuardRef.current.isCurrent(generation)) {
        return;
      }
      setPresetDelta(delta);
      setPresetRequestState("ready");
      setPresetError(null);

      if (delta.ready) {
        setPresetRequestState("importing");
        try {
          const result = await importRemoteSetupConfig(delta.recipe_url);
          if (!presetRequestGuardRef.current.isCurrent(generation)) {
            return;
          }
          setPresetImportedConfig({
            config: result.config,
            sourceUrl: result.sourceUrl,
            counts: createSetupConfigCounts({
              models: result.models,
              decisions: result.decisions,
              signals: result.signals,
              canActivate: result.canActivate,
            }),
          });
          setPresetRequestState("imported");
          setPresetError(null);
        } catch (err) {
          if (!presetRequestGuardRef.current.isCurrent(generation)) {
            return;
          }
          setPresetImportedConfig(null);
          setPresetRequestState("error");
          setPresetError(
            err instanceof Error ? err.message : "Preset import failed.",
          );
        }
      }
    } catch (err) {
      if (!presetRequestGuardRef.current.isCurrent(generation)) {
        return;
      }
      setPresetDelta(null);
      setPresetImportedConfig(null);
      setPresetRequestState("error");
      setPresetError(
        err instanceof Error
          ? err.message
          : "Failed to check the selected starter architecture.",
      );
    }
  };

  const handleImportPresetConfig = async () => {
    if (!presetDelta) {
      return;
    }
    const generation = presetRequestGuardRef.current.begin();
    setPresetRequestState("importing");
    setPresetError(null);
    resetReviewState();
    try {
      const result = await importRemoteSetupConfig(presetDelta.recipe_url);
      if (!presetRequestGuardRef.current.isCurrent(generation)) {
        return;
      }
      setPresetImportedConfig({
        config: result.config,
        sourceUrl: result.sourceUrl,
        counts: createSetupConfigCounts({
          models: result.models,
          decisions: result.decisions,
          signals: result.signals,
          canActivate: result.canActivate,
        }),
      });
      setPresetRequestState("imported");
      setPresetError(null);
    } catch (err) {
      if (!presetRequestGuardRef.current.isCurrent(generation)) {
        return;
      }
      setPresetImportedConfig(null);
      setPresetRequestState("error");
      setPresetError(
        err instanceof Error ? err.message : "Preset import failed.",
      );
    }
  };

  const handleImportRemote = async () => {
    const trimmedUrl = remoteConfigUrl.trim();
    if (!trimmedUrl) {
      setRemoteImportState("error");
      setRemoteImportError("Paste a remote config URL before importing.");
      return;
    }

    const generation = remoteImportGuardRef.current.begin();
    setRemoteImportState("importing");
    setRemoteImportError(null);
    resetReviewState();

    try {
      const result = await importRemoteSetupConfig(trimmedUrl);
      if (!remoteImportGuardRef.current.isCurrent(generation)) {
        return;
      }
      setImportedRemoteConfig({
        config: result.config,
        sourceUrl: result.sourceUrl,
        counts: createSetupConfigCounts({
          models: result.models,
          decisions: result.decisions,
          signals: result.signals,
          canActivate: result.canActivate,
        }),
      });
      setRemoteConfigUrl(result.sourceUrl);
      setRemoteImportState("imported");
      setRemoteImportError(null);
    } catch (err) {
      if (!remoteImportGuardRef.current.isCurrent(generation)) {
        return;
      }
      setImportedRemoteConfig(null);
      setRemoteImportState("error");
      setRemoteImportError(
        err instanceof Error ? err.message : "Remote import failed.",
      );
    }
  };

  const handleSelectRoutingMode = (mode: SetupRoutingMode) => {
    setRoutingMode(mode);
    setRemoteImportError(null);
    setPresetError(null);
    resetReviewState();

    if (mode !== "remote") {
      remoteImportGuardRef.current.invalidate();
      if (remoteImportState === "importing") {
        setRemoteImportState(importedRemoteConfig ? "imported" : "idle");
      }
    }

    if (mode !== "preset") {
      presetRequestGuardRef.current.invalidate();
      if (
        presetRequestState === "loading" ||
        presetRequestState === "importing"
      ) {
        setPresetRequestState(
          presetImportedConfig ? "imported" : presetDelta ? "ready" : "idle",
        );
      }
    }
  };

  const handleRemoteConfigUrlChange = (value: string) => {
    remoteImportGuardRef.current.invalidate();
    setRemoteConfigUrl(value);
    setRemoteImportError(null);
    if (
      importedRemoteConfig &&
      value.trim() !== importedRemoteConfig.sourceUrl
    ) {
      setImportedRemoteConfig(null);
    }
    if (
      remoteImportState === "error" ||
      remoteImportState === "importing" ||
      (importedRemoteConfig && value.trim() !== importedRemoteConfig.sourceUrl)
    ) {
      setRemoteImportState("idle");
    }
    resetReviewState();
  };

  const handleActivate = async () => {
    if (!draftConfig || validationState !== "valid") {
      return;
    }

    setActivationState("activating");
    setActivationError(null);

    try {
      const payload = validatedConfig ?? draftConfig;
      await activateSetupConfig(payload);
      markOnboardingPending();
      await refreshSetupState();
      navigate("/dashboard", { replace: true });
    } catch (err) {
      setActivationState("error");
      setActivationError(
        err instanceof Error ? err.message : "Setup activation failed.",
      );
    }
  };

  return (
    <div className={styles.page}>
      <SetupWizardBackground />

      <div className={styles.content}>
        <div className={styles.hero}>
          <div className={styles.heroHeader}>
            <div className={styles.heroBadge}>Mixture-of-Models setup</div>
          </div>
          <div className={styles.heroTitleRow}>
            <div className={styles.heroLogoWrap} aria-hidden="true">
              <img className={styles.heroLogo} src="/vllm.png" alt="" />
            </div>
            <h1 className={styles.heroTitle}>
              Build your first Mixture-of-Models.
            </h1>
          </div>
          <p className={styles.heroDescription}>
            Connect heterogeneous LLMs, then compose how they route and work together.
          </p>
        </div>

        <SetupWizardStepper currentStep={currentStep} onGoToStep={goToStep} />

        <div
          id={`setup-step-${currentStep}-panel`}
          className={styles.panel}
          role="region"
          aria-labelledby={`setup-step-${currentStep}-button`}
        >
          {currentStep === 0 && (
            <ModelStepPanel
              currentRouteLabel={currentRouteLabel}
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
          )}
          {currentStep === 1 && (
            <RoutingStarterPanel
              currentRouteLabel={currentRouteLabel}
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
          )}
          {currentStep === 2 && (
            <ReviewActivatePanel
              currentRouteLabel={currentRouteLabel}
              listenerPort={setupState?.listenerPort}
              validationState={validationState}
              validationError={validationError}
              activationState={activationState}
              activationError={activationError}
              validatedCounts={validatedCounts}
              modelsCount={generatedCounts.models}
              generatedDecisions={generatedCounts.decisions}
              generatedSignals={generatedCounts.signals}
              previewSource={previewSource}
              readonlyLoading={readonlyLoading}
              isReadonly={isReadonly}
              onValidateAgain={() => void handleValidateAgain()}
            />
          )}

          <div className={styles.footer}>
            <div className={styles.footerActions}>
              {currentStep > 0 && (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={handleBack}
                >
                  Back
                </button>
              )}
              {currentStep < 2 && (
                <button
                  type="button"
                  className={styles.primaryButton}
                  onClick={handleNext}
                >
                  Next
                </button>
              )}
              {currentStep === 2 && (
                <button
                  type="button"
                  className={styles.primaryButton}
                  onClick={() => void handleActivate()}
                  aria-busy={activationState === "activating"}
                  disabled={
                    validationState !== "valid" ||
                    !validatedCounts.canActivate ||
                    activationState === "activating" ||
                    (!readonlyLoading && isReadonly)
                  }
                >
                  {activationState === "activating"
                    ? "Activating…"
                    : "Activate"}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SetupWizardPage;
