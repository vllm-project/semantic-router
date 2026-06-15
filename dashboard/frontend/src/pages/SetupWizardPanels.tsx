import styles from "./SetupWizardPage.module.css";
import {
  DEFAULT_REMOTE_SETUP_CONFIG_URL,
  getModelDraftFieldErrors,
  PROVIDER_OPTIONS,
  SETUP_STEP_LABELS,
  type ImportedSetupConfig,
  type ModelDraft,
  type PresetDelta,
  type PresetInfo,
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
  presets: PresetInfo[];
  selectedPresetId: string | null;
  presetDelta: PresetDelta | null;
  presetImportedConfig: ImportedSetupConfig | null;
  presetError: string | null;
  onSelectRoutingMode: (mode: SetupRoutingMode) => void;
  onChangeRemoteConfigUrl: (value: string) => void;
  onImportRemoteConfig: () => void;
  onSelectPreset: (id: string) => void;
  onImportPresetConfig: () => void;
}

function PresetModelChecklist({
  presetDelta,
  presetImportedConfig,
  counts,
  presetError,
  onImportPresetConfig,
}: {
  presetDelta: PresetDelta;
  presetImportedConfig: ImportedSetupConfig | null;
  counts: SetupConfigCounts;
  presetError: string | null;
  onImportPresetConfig: () => void;
}) {
  const total =
    presetDelta.configured_models.length + presetDelta.missing_models.length;
  const readyCount = presetDelta.configured_models.length;
  const pct = total > 0 ? Math.round((readyCount / total) * 100) : 0;

  return (
    <div className={styles.remoteImportSummary}>
      <div className={styles.remoteImportSummaryHeader}>
        <h4 className={styles.presetCardTitle}>
          {presetDelta.ready ? "Ready to activate" : "Model requirements"}
        </h4>
        <span className={styles.presetCardMeta}>
          {readyCount}/{total} models ready
        </span>
      </div>

      <div
        style={{
          height: 4,
          borderRadius: 2,
          background: "var(--color-border, #333)",
          marginBottom: 12,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            borderRadius: 2,
            background: presetDelta.ready
              ? "var(--color-success, #22c55e)"
              : "var(--color-warning, #eab308)",
            transition: "width 0.3s ease",
          }}
        />
      </div>

      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {presetDelta.configured_models.map((name) => (
          <li
            key={name}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "4px 0",
              color: "var(--color-success, #22c55e)",
            }}
          >
            <span style={{ fontSize: 16, lineHeight: 1 }}>&#10003;</span>
            <span style={{ color: "var(--color-text, #e5e5e5)" }}>
              <strong>{name}</strong>
            </span>
            <span
              style={{
                fontSize: 12,
                color: "var(--color-text-muted, #888)",
              }}
            >
              configured
            </span>
          </li>
        ))}
        {presetDelta.missing_models.map((m) => (
          <li
            key={m.name}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "4px 0",
              color: "var(--color-warning, #eab308)",
            }}
          >
            <span style={{ fontSize: 16, lineHeight: 1 }}>&#9675;</span>
            <span style={{ color: "var(--color-text, #e5e5e5)" }}>
              <strong>{m.name}</strong>
            </span>
            <span
              style={{
                fontSize: 12,
                color: "var(--color-text-muted, #888)",
              }}
            >
              {m.role} &mdash; add in step 1
            </span>
          </li>
        ))}
      </ul>

      {!presetDelta.ready && (
        <p className={styles.fieldHint} style={{ marginTop: 12 }}>
          Missing models will use placeholder endpoints. You can update them
          after activation from the config page.
        </p>
      )}

      {presetError && (
        <p className={styles.fieldErrorText} style={{ marginTop: 8 }}>
          {presetError}
        </p>
      )}

      {!presetImportedConfig && (
        <div className={styles.remoteImportActions}>
          <button
            className={styles.secondaryButton}
            onClick={onImportPresetConfig}
          >
            {presetDelta.ready
              ? "Import preset config"
              : "Import with placeholders"}
          </button>
        </div>
      )}

      {presetImportedConfig && (
        <div
          style={{
            marginTop: 12,
            padding: "8px 12px",
            borderRadius: 6,
            background: "var(--color-success-bg, rgba(34,197,94,0.1))",
            border: "1px solid var(--color-success, #22c55e)",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span style={{ color: "var(--color-success, #22c55e)", fontSize: 16 }}>
            &#10003;
          </span>
          <span>
            <strong>Preset config ready</strong> &mdash; {counts.models} models
            &middot; {counts.decisions} decisions &middot; {counts.signals}{" "}
            signals
          </span>
        </div>
      )}
    </div>
  );
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
  const fieldErrorsByModelId = getModelDraftFieldErrors(models);

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
                  />
                  <span className={styles.fieldHint}>
                    {providerMeta?.description} You can enter a full URL like{" "}
                    <code>{providerMeta?.placeholder}</code> or a host such as{" "}
                    <code>localhost:8000/v1</code>.
                  </span>
                  {hasBaseUrlError && (
                    <span className={styles.fieldErrorText}>
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
  presets,
  selectedPresetId,
  presetDelta,
  presetImportedConfig,
  presetError,
  onSelectRoutingMode,
  onChangeRemoteConfigUrl,
  onImportRemoteConfig,
  onSelectPreset,
  onImportPresetConfig,
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

          {presets.map((preset) => {
            const isActive =
              isPresetMode && selectedPresetId === preset.id;
            return (
              <button
                key={preset.id}
                className={`${styles.presetCard} ${isActive ? styles.presetCardActive : ""}`}
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

        {isPresetMode && presetDelta && (
          <div className={styles.remoteImportPanel}>
            <PresetModelChecklist
              presetDelta={presetDelta}
              presetImportedConfig={presetImportedConfig}
              counts={counts}
              presetError={presetError}
              onImportPresetConfig={onImportPresetConfig}
            />
          </div>
        )}

        {isPresetMode && !presetDelta && presetError && (
          <div className={styles.remoteImportPanel}>
            <p className={styles.fieldErrorText}>{presetError}</p>
          </div>
        )}

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
