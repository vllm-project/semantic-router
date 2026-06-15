import React from "react";

import type { BuilderNLStagedDraft, EditorMode } from "@/types/dsl";

import styles from "./builderPageNaturalLanguagePanel.module.css";

interface BuilderNLDraftPreviewOption {
  key: "dsl" | "yaml" | "crd" | "baseYaml";
  label: string;
  content: string;
  emptyText: string;
}

interface BuilderNaturalLanguageResultCardProps {
  stagedDraft: BuilderNLStagedDraft | null;
  draftPreviewOptions: BuilderNLDraftPreviewOption[];
  activeDraftPreview?: BuilderNLDraftPreviewOption;
  copiedPreview: string | null;
  onPreviewTabChange: (tab: BuilderNLDraftPreviewOption["key"]) => void;
  onCopyPreview: () => void;
  onApplyDraft: () => void;
  onDiscardDraft: () => void;
  onModeSwitch: (mode: EditorMode) => void;
}

const BuilderNaturalLanguageResultCard: React.FC<
  BuilderNaturalLanguageResultCardProps
> = ({
  stagedDraft,
  draftPreviewOptions,
  activeDraftPreview,
  copiedPreview,
  onPreviewTabChange,
  onCopyPreview,
  onApplyDraft,
  onDiscardDraft,
  onModeSwitch,
}) => {
  const stagedDiagnostics = stagedDraft?.validation.diagnostics ?? [];
  const stagedWarningCount = stagedDiagnostics.filter(
    (item) => item.level !== "error",
  ).length;
  const stagedReviewWarnings = stagedDraft?.review.warnings ?? [];
  const stagedReviewChecks = stagedDraft?.review.checks ?? [];

  return (
    <section className={styles.card}>
      <div className={styles.sectionHeader}>
        <div>
          <h3 className={styles.cardTitle}>Generated outputs</h3>
          <p className={styles.sectionHint}>
            Inspect the final DSL and compiled outputs first, then decide whether
            the staged result is ready to apply into Builder.
          </p>
        </div>
        {stagedDraft ? (
          <button
            className={styles.ghostBtn}
            onClick={onDiscardDraft}
            type="button"
          >
            Discard
          </button>
        ) : null}
      </div>

      {stagedDraft ? (
        <>
          <div
            className={`${styles.resultStatusBanner} ${
              stagedDraft.validation.ready
                ? styles.resultStatusReady
                : styles.resultStatusWarn
            }`}
          >
            <div>
              <div className={styles.resultLabel}>Builder handoff</div>
              <div className={styles.resultStatusTitle}>
                {stagedDraft.validation.ready
                  ? "Repository checks passed"
                  : "Repository review still has findings"}
              </div>
              <div className={styles.resultStatusText}>
                {stagedDraft.review.summary || stagedDraft.summary || "Draft generated."}
              </div>
            </div>
            <div className={styles.resultStatusMeta}>
              <span className={styles.badgeMuted}>
                {stagedDraft.validation.errorCount} validation error
                {stagedDraft.validation.errorCount === 1 ? "" : "s"}
              </span>
              {stagedWarningCount > 0 ? (
                <span className={styles.badgeMuted}>
                  {stagedWarningCount} warning{stagedWarningCount === 1 ? "" : "s"}
                </span>
              ) : null}
            </div>
          </div>

          <div className={styles.resultLayout}>
            <div className={styles.resultMain}>
              <div className={styles.previewHeader}>
                <div>
                  <div className={styles.resultLabel}>Final outputs</div>
                  <div className={styles.previewCaption}>
                    The shared nlgen result is shown here exactly as Builder
                    staged it, plus the local YAML and CRD compilation results.
                  </div>
                </div>
                <div className={styles.previewActions}>
                  <div className={styles.previewTabRow}>
                    {draftPreviewOptions.map((option) => (
                      <button
                        className={
                          option.key === activeDraftPreview?.key
                            ? styles.segmentActive
                            : styles.segment
                        }
                        key={option.key}
                        onClick={() => onPreviewTabChange(option.key)}
                        type="button"
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  <button
                    className={styles.ghostBtn}
                    disabled={!activeDraftPreview?.content}
                    onClick={onCopyPreview}
                    type="button"
                  >
                    {copiedPreview === activeDraftPreview?.key
                      ? "Copied"
                      : `Copy ${activeDraftPreview?.label ?? "output"}`}
                  </button>
                </div>
              </div>

              {activeDraftPreview?.content ? (
                <pre className={styles.preview}>{activeDraftPreview.content}</pre>
              ) : (
                <div className={styles.previewEmpty}>
                  {activeDraftPreview?.emptyText ||
                    "No staged draft output is available yet."}
                </div>
              )}

              {stagedDraft.suggestedTestQuery ? (
                <div className={styles.resultBlock}>
                  <div className={styles.resultLabel}>Suggested test prompt</div>
                  <div className={styles.testQuery}>
                    {stagedDraft.suggestedTestQuery}
                  </div>
                </div>
              ) : null}
            </div>

            <div className={styles.resultSidebar}>
              <div className={styles.resultMetaCard}>
                <div className={styles.resultLabel}>Original request</div>
                <div className={styles.requestPreview}>{stagedDraft.prompt}</div>
              </div>

              <div className={styles.resultMetaCard}>
                <div className={styles.resultLabel}>Builder apply gate</div>
                <div
                  className={
                    stagedDraft.review.ready ? styles.statusOk : styles.statusWarn
                  }
                >
                  {stagedDraft.review.ready
                    ? "Ready to apply"
                    : "Manual review recommended"}
                </div>
                <div className={styles.statusMeta}>{stagedDraft.review.summary}</div>
              </div>

              {stagedDraft.validation.compileError ? (
                <div className={styles.resultMetaCard}>
                  <div className={styles.resultLabel}>Compile error</div>
                  <div className={styles.errorText}>
                    {stagedDraft.validation.compileError}
                  </div>
                </div>
              ) : null}

              {stagedDiagnostics.length > 0 ? (
                <div className={styles.resultMetaCard}>
                  <div className={styles.resultLabel}>Validation findings</div>
                  <ul className={styles.list}>
                    {stagedDiagnostics.slice(0, 8).map((item, index) => (
                      <li key={`${item.message}-${index}`}>
                        <strong>{item.level}</strong>: {item.message}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {stagedReviewWarnings.length > 0 ? (
                <div className={styles.resultMetaCard}>
                  <div className={styles.resultLabel}>Review warnings</div>
                  <ul className={styles.list}>
                    {stagedReviewWarnings.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {stagedReviewChecks.length > 0 ? (
                <div className={styles.resultMetaCard}>
                  <div className={styles.resultLabel}>Checks completed</div>
                  <ul className={styles.listMuted}>
                    {stagedReviewChecks.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </div>

          <div className={styles.applyRow}>
            <button className={styles.primaryBtn} onClick={onApplyDraft} type="button">
              {stagedDraft.validation.ready
                ? "Apply staged DSL to Builder"
                : "Open staged DSL in Builder"}
            </button>
            <button
              className={styles.secondaryBtn}
              onClick={() => onModeSwitch("dsl")}
              type="button"
            >
              Open live Builder draft
            </button>
          </div>
          <div className={styles.applyNote}>
            Applying the staged DSL keeps the current Builder base YAML intact,
            including global, provider, and listener settings.
          </div>
        </>
      ) : (
        <div className={styles.emptyState}>
          <div className={styles.emptyTitle}>No generated DSL yet</div>
          <div className={styles.emptyText}>
            Run generation to inspect the final DSL, compiled YAML, and Builder
            handoff state before you touch the live Builder draft.
          </div>
        </div>
      )}
    </section>
  );
};

export { BuilderNaturalLanguageResultCard };
