import React from "react";

import type { EditorMode } from "@/types/dsl";

import styles from "./BuilderPage.module.css";

interface BuilderToolbarProps {
  dirty: boolean;
  mode: EditorMode;
  wasmReady: boolean;
  wasmError: string | null;
  dslSource: string;
  loading: boolean;
  deploying: boolean;
  guideOpen: boolean;
  outputPanelOpen: boolean;
  onModeSwitch: (mode: EditorMode) => void;
  onImport: () => void;
  onCompile: () => void;
  onRequestDeploy: () => void;
  onFormat: () => void;
  onValidate: () => void;
  onToggleGuide: () => void;
  onToggleOutput: () => void;
  onReset: () => void;
}

const BuilderToolbar: React.FC<BuilderToolbarProps> = ({
  dirty,
  mode,
  wasmReady,
  wasmError,
  dslSource,
  loading,
  deploying,
  guideOpen,
  outputPanelOpen,
  onModeSwitch,
  onImport,
  onCompile,
  onRequestDeploy,
  onFormat,
  onValidate,
  onToggleGuide,
  onToggleOutput,
  onReset,
}) => {
  return (
    <div className={styles.toolbar}>
      <div className={styles.toolbarTitle}>
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <rect x="2" y="2" width="5" height="5" rx="1" />
          <rect x="9" y="2" width="5" height="5" rx="1" />
          <rect x="2" y="9" width="5" height="5" rx="1" />
          <rect x="9" y="9" width="5" height="5" rx="1" />
        </svg>
        Config Builder
        {dirty && (
          <span style={{ color: "var(--color-text-muted)", fontWeight: 400 }}>
            (unsaved)
          </span>
        )}
      </div>

      <span className={styles.divider} />

      <div className={styles.modeSwitcher}>
        <button
          className={mode === "visual" ? styles.modeBtnActive : styles.modeBtn}
          onClick={() => onModeSwitch("visual")}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="1" y="1" width="6" height="6" rx="1" />
            <rect x="9" y="1" width="6" height="6" rx="1" />
            <rect x="1" y="9" width="6" height="6" rx="1" />
            <rect x="9" y="9" width="6" height="6" rx="1" />
          </svg>
          Visual
        </button>
        <button
          className={mode === "dsl" ? styles.modeBtnActive : styles.modeBtn}
          onClick={() => onModeSwitch("dsl")}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M2 3h12M2 8h8M2 13h10" strokeLinecap="round" />
          </svg>
          DSL
        </button>
        <button
          className={styles.modeBtn}
          disabled
          title="Natural Language mode — coming soon"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M2 4h12M2 8h9M2 12h6" strokeLinecap="round" />
            <circle cx="13" cy="11" r="2" />
          </svg>
          NL
        </button>
      </div>

      <span className={styles.divider} />

      {wasmError ? (
        <span className={styles.statusError}>
          <span className={styles.dot} /> WASM Error
        </span>
      ) : wasmReady ? (
        <span className={styles.statusReady}>
          <span className={styles.dot} /> Ready
        </span>
      ) : (
        <span className={styles.statusLoading}>
          <span className={styles.dotPulse} /> Loading WASM…
        </span>
      )}

      <div className={styles.toolbarRight}>
        <button
          className={styles.toolbarBtnPrimary}
          onClick={mode !== "nl" ? onImport : undefined}
          disabled={!wasmReady || mode === "nl"}
          title={
            mode === "nl"
              ? "Import Config is not available in NL mode (coming soon)"
              : "Import router config"
          }
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path
              d="M8 2v8M5 7l3 3 3-3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M2 11v2a1 1 0 001 1h10a1 1 0 001-1v-2"
              strokeLinecap="round"
            />
          </svg>
          Import
        </button>
        <button
          className={styles.toolbarBtnPrimary}
          onClick={onCompile}
          disabled={!wasmReady || !dslSource.trim() || loading}
          title="Compile (Ctrl+Enter)"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M4 2l8 6-8 6V2z" fill="currentColor" />
          </svg>
          {loading ? "Compiling…" : "Compile"}
        </button>
        <button
          className={styles.toolbarBtnDeploy}
          onClick={onRequestDeploy}
          disabled={!wasmReady || !dslSource.trim() || loading || deploying}
          title="Deploy config to router"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path
              d="M8 2v8M5 7l3 3 3-3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M2 12v1a1 1 0 001 1h10a1 1 0 001-1v-1"
              strokeLinecap="round"
            />
          </svg>
          {deploying ? "Deploying…" : "Deploy"}
        </button>
        <button
          className={styles.toolbarBtn}
          onClick={onFormat}
          disabled={!wasmReady || !dslSource.trim()}
          title="Format DSL"
        >
          Format
        </button>
        <button
          className={styles.toolbarBtn}
          onClick={onValidate}
          disabled={!wasmReady || !dslSource.trim()}
          title="Validate"
        >
          Validate
        </button>
        <span className={styles.divider} />
        <button
          className={guideOpen ? styles.toolbarBtnActive : styles.toolbarBtn}
          onClick={onToggleGuide}
          title={guideOpen ? "Close DSL Guide" : "Open DSL Guide"}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path
              d="M2 2h9a2 2 0 012 2v10l-3-2H2V2z"
              strokeLinejoin="round"
            />
            <path d="M5 6h5M5 9h3" strokeLinecap="round" />
          </svg>
          Guide
        </button>
        <button
          className={
            outputPanelOpen ? styles.toolbarBtnActive : styles.toolbarBtn
          }
          onClick={onToggleOutput}
          title={outputPanelOpen ? "Hide Output Panel" : "Show Output Panel"}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="1" y="1" width="14" height="14" rx="2" />
            <path d="M10 1v14" />
          </svg>
          Output
        </button>
        <span className={styles.divider} />
        <button className={styles.toolbarBtnDanger} onClick={onReset} title="Reset">
          Reset
        </button>
      </div>
    </div>
  );
};

export { BuilderToolbar };
