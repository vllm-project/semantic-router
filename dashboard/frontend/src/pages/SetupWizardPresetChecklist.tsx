import styles from "./SetupWizardPage.module.css";
import type {
  ImportedSetupConfig,
  PresetDelta,
  PresetRequestState,
  SetupConfigCounts,
} from "./setupWizardSupport";

interface SetupWizardPresetChecklistProps {
  presetDelta: PresetDelta;
  presetImportedConfig: ImportedSetupConfig | null;
  counts: SetupConfigCounts;
  presetError: string | null;
  presetRequestState: PresetRequestState;
  onImportPresetConfig: () => void;
}

export default function SetupWizardPresetChecklist({
  presetDelta,
  presetImportedConfig,
  counts,
  presetError,
  presetRequestState,
  onImportPresetConfig,
}: SetupWizardPresetChecklistProps) {
  const total =
    presetDelta.configured_models.length + presetDelta.missing_models.length;
  const readyCount = presetDelta.configured_models.length;
  const isImporting = presetRequestState === "importing";

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

      <progress
        className={styles.presetProgress}
        value={readyCount}
        max={Math.max(1, total)}
        aria-label={`${readyCount} of ${total} required models configured`}
      />

      <ul className={styles.presetModelList}>
        {presetDelta.configured_models.map((name) => (
          <li key={name} className={styles.presetModelItem}>
            <span className={styles.presetModelReadyIcon} aria-hidden="true">
              &#10003;
            </span>
            <span className={styles.presetModelName}>
              <strong>{name}</strong>
            </span>
            <span className={styles.presetModelMeta}>configured</span>
          </li>
        ))}
        {presetDelta.missing_models.map((model) => (
          <li key={model.name} className={styles.presetModelItem}>
            <span className={styles.presetModelMissingIcon} aria-hidden="true">
              &#9675;
            </span>
            <span className={styles.presetModelName}>
              <strong>{model.name}</strong>
            </span>
            <span className={styles.presetModelMeta}>
              {model.role} &mdash; add in step 1
            </span>
          </li>
        ))}
      </ul>

      {!presetDelta.ready && (
        <p className={`${styles.fieldHint} ${styles.presetChecklistHint}`}>
          Missing models will use placeholder endpoints. You can update them
          after activation from the config page.
        </p>
      )}

      {presetError && (
        <p className={styles.fieldErrorText} role="alert">
          {presetError}
        </p>
      )}

      {!presetImportedConfig && (
        <div className={styles.remoteImportActions}>
          <button
            className={styles.secondaryButton}
            onClick={onImportPresetConfig}
            disabled={isImporting}
          >
            {isImporting
              ? "Importing preset…"
              : presetError
                ? "Retry preset import"
                : presetDelta.ready
                  ? "Import preset config"
                  : "Import with placeholders"}
          </button>
        </div>
      )}

      {presetImportedConfig && (
        <div className={styles.presetReadyNotice} role="status">
          <span className={styles.presetModelReadyIcon} aria-hidden="true">
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
