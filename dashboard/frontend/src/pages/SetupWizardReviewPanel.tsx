import styles from "./SetupWizardPage.module.css";
import { SetupRouteSummary } from "./SetupWizardPanels";
import type { SetupValidationState } from "./setupWizardSupport";

interface ReviewActivatePanelProps {
  currentRouteLabel: string;
  listenerPort?: number;
  validationState: SetupValidationState;
  validationError: string | null;
  activationError: string | null;
  validatedCounts: {
    models: number;
    decisions: number;
    signals: number;
    canActivate: boolean;
  };
  modelsCount: number;
  generatedDecisions: number;
  generatedSignals: number;
  previewSource: string;
  readonlyLoading: boolean;
  isReadonly: boolean;
  onValidateAgain: () => void;
}

export function ReviewActivatePanel({
  currentRouteLabel,
  listenerPort,
  validationState,
  validationError,
  activationError,
  validatedCounts,
  modelsCount,
  generatedDecisions,
  generatedSignals,
  previewSource,
  readonlyLoading,
  isReadonly,
  onValidateAgain,
}: ReviewActivatePanelProps) {
  return (
    <div className={styles.stepBody}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeaderMain}>
          <h2 className={styles.sectionTitle}>Review and activate</h2>
          <p className={styles.sectionDescription}>
            The dashboard validates the generated config before activation.
            Activation writes the resulting YAML to <code>config.yaml</code> and
            exits setup mode.
          </p>
          <SetupRouteSummary currentRouteLabel={currentRouteLabel} />
        </div>
        <div className={styles.sectionHeaderAside}>
          <button className={styles.secondaryButton} onClick={onValidateAgain}>
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
                validationState === "valid"
                  ? styles.statusSuccess
                  : validationState === "error"
                    ? styles.statusError
                    : styles.statusPending
              }`}
            >
              {validationState === "validating" && "Validating"}
              {validationState === "valid" && "Ready"}
              {validationState === "error" && "Needs fixes"}
              {validationState === "idle" && "Pending"}
            </span>
          </div>

          <div className={styles.reviewStats}>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Listener</span>
              <span className={styles.reviewStatValue}>
                {listenerPort ? `:${listenerPort}` : "Bootstrap"}
              </span>
            </div>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Models</span>
              <span className={styles.reviewStatValue}>
                {validatedCounts.models || modelsCount}
              </span>
            </div>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Decisions</span>
              <span className={styles.reviewStatValue}>
                {validatedCounts.decisions || generatedDecisions}
              </span>
            </div>
            <div className={styles.reviewStat}>
              <span className={styles.reviewStatLabel}>Signals</span>
              <span className={styles.reviewStatValue}>
                {validatedCounts.signals || generatedSignals}
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
              This dashboard is running in read-only mode. Review is available,
              but activation is disabled.
            </div>
          )}
        </div>

        <div className={styles.previewCard}>
          <div className={styles.previewHeader}>
            <h3 className={styles.reviewCardTitle}>Generated config preview</h3>
            <span className={styles.previewHint}>
              Secrets are masked in preview.
            </span>
          </div>
          <pre className={styles.previewCode}>{previewSource}</pre>
        </div>
      </div>
    </div>
  );
}
