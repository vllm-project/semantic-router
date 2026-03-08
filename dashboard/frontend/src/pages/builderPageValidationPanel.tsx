import React from "react";

import type { Diagnostic } from "@/types/dsl";

import styles from "./BuilderPage.module.css";

interface BuilderValidationPanelProps {
  diagnostics: Diagnostic[];
  validationOpen: boolean;
  errorDiags: Diagnostic[];
  warnDiags: Diagnostic[];
  constraintDiags: Diagnostic[];
  onToggle: () => void;
  onApplyFix: (diag: Diagnostic, newText: string) => void;
}

const BuilderValidationPanel: React.FC<BuilderValidationPanelProps> = ({
  diagnostics,
  validationOpen,
  errorDiags,
  warnDiags,
  constraintDiags,
  onToggle,
  onApplyFix,
}) => {
  if (diagnostics.length === 0) {
    return null;
  }

  return (
    <div className={styles.validationPanel}>
      <div className={styles.validationHeader} onClick={onToggle}>
        <span className={styles.validationTitle}>
          <svg
            width="14"
            height="14"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path
              d="M3 8.5l3 3 7-7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Validation
        </span>
        <span className={styles.validationCounts}>
          {errorDiags.length > 0 && (
            <span className={styles.valCountError}>
              {errorDiags.length} error{errorDiags.length !== 1 ? "s" : ""}
            </span>
          )}
          {warnDiags.length > 0 && (
            <span className={styles.valCountWarn}>
              {warnDiags.length} warning{warnDiags.length !== 1 ? "s" : ""}
            </span>
          )}
          {constraintDiags.length > 0 && (
            <span className={styles.valCountConstraint}>
              {constraintDiags.length} constraint
              {constraintDiags.length !== 1 ? "s" : ""}
            </span>
          )}
        </span>
        <svg
          className={`${styles.validationChevron} ${validationOpen ? styles.validationChevronOpen : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path
            d="M4 6l4 4 4-4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      {validationOpen && (
        <div className={styles.validationBody}>
          {errorDiags.length > 0 && (
            <div className={styles.valGroup}>
              <div className={styles.valGroupTitle}>
                <svg
                  className={styles.valIconError}
                  viewBox="0 0 16 16"
                  fill="currentColor"
                >
                  <circle cx="8" cy="8" r="7" />
                  <path
                    d="M5.5 5.5l5 5M10.5 5.5l-5 5"
                    stroke="#fff"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
                Error ({errorDiags.length})
              </div>
              {errorDiags.map((d, i) => (
                <div key={i} className={styles.valItem}>
                  <span className={styles.valMessage}>
                    Ln {d.line}, Col {d.column}: {d.message}
                  </span>
                  {d.fixes?.map((fix, fi) => (
                    <button
                      key={fi}
                      className={styles.valFixBtn}
                      onClick={() => onApplyFix(d, fix.newText)}
                      title={fix.description}
                    >
                      Fix
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
          {warnDiags.length > 0 && (
            <div className={styles.valGroup}>
              <div className={styles.valGroupTitle}>
                <svg
                  className={styles.valIconWarn}
                  viewBox="0 0 16 16"
                  fill="currentColor"
                >
                  <path d="M8 1l7 13H1L8 1z" />
                  <path
                    d="M8 6v3M8 11v1"
                    stroke="#000"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
                Warning ({warnDiags.length})
              </div>
              {warnDiags.map((d, i) => (
                <div key={i} className={styles.valItem}>
                  <span className={styles.valMessage}>
                    Ln {d.line}, Col {d.column}: {d.message}
                  </span>
                  {d.fixes?.map((fix, fi) => (
                    <button
                      key={fi}
                      className={styles.valFixBtn}
                      onClick={() => onApplyFix(d, fix.newText)}
                      title={fix.description}
                    >
                      Fix
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
          {constraintDiags.length > 0 && (
            <div className={styles.valGroup}>
              <div className={styles.valGroupTitle}>
                <svg
                  className={styles.valIconConstraint}
                  viewBox="0 0 16 16"
                  fill="currentColor"
                >
                  <circle cx="8" cy="8" r="7" />
                  <path
                    d="M8 5v4M8 11v1"
                    stroke="#000"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
                Constraint ({constraintDiags.length})
              </div>
              {constraintDiags.map((d, i) => (
                <div key={i} className={styles.valItem}>
                  <span className={styles.valMessage}>
                    Ln {d.line}, Col {d.column}: {d.message}
                  </span>
                  {d.fixes?.map((fix, fi) => (
                    <button
                      key={fi}
                      className={styles.valFixBtn}
                      onClick={() => onApplyFix(d, fix.newText)}
                      title={fix.description}
                    >
                      Fix
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export { BuilderValidationPanel };
