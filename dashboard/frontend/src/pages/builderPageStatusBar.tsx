import React from "react";

import type { EditorMode } from "@/types/dsl";

import styles from "./BuilderPage.module.css";

interface BuilderStatusBarProps {
  isValid: boolean;
  errorCount: number;
  modelCount: number;
  signalCount: number;
  routeCount: number;
  pluginCount: number;
  lineCount: number;
  mode: EditorMode;
}

const BuilderStatusBar: React.FC<BuilderStatusBarProps> = ({
  isValid,
  errorCount,
  modelCount,
  signalCount,
  routeCount,
  pluginCount,
  lineCount,
  mode,
}) => {
  return (
    <div className={styles.statusBar}>
      <div
        className={`${styles.statusItem} ${isValid ? styles.statusValid : styles.statusInvalid}`}
      >
        {isValid ? (
          <svg
            className={styles.statusCheckmark}
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              d="M3 8.5l3 3 7-7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ) : (
          <svg
            className={styles.statusCheckmark}
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              d="M4 4l8 8M12 4l-8 8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
        {isValid ? "Valid" : `${errorCount} error${errorCount !== 1 ? "s" : ""}`}
      </div>
      <div className={styles.statusItem}>Models: {modelCount}</div>
      <div className={styles.statusItem}>Signals: {signalCount}</div>
      <div className={styles.statusItem}>Routes: {routeCount}</div>
      <div className={styles.statusItem}>Plugins: {pluginCount}</div>
      {mode === "dsl" && <div className={styles.statusItem}>Lines: {lineCount}</div>}
      <div className={styles.statusItem}>
        Mode: {mode === "visual" ? "Visual" : mode === "dsl" ? "DSL" : "NL"}
      </div>
    </div>
  );
};

export { BuilderStatusBar };
