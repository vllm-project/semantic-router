import React, { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { DiffEditor } from "@monaco-editor/react";

import type { DeployResult, DeployStep } from "@/types/dsl";

import styles from "./BuilderPage.module.css";

const DEPLOY_STEP_ORDER: DeployStep[] = [
  "validating",
  "backing_up",
  "writing",
  "reloading",
  "done",
];

const DeployStepItem: React.FC<{
  step: DeployStep;
  current: DeployStep | null;
  label: string;
}> = ({ step, current, label }) => {
  const currentIdx = current ? DEPLOY_STEP_ORDER.indexOf(current) : -1;
  const stepIdx = DEPLOY_STEP_ORDER.indexOf(step);
  const isDone =
    current === "done" || (currentIdx > stepIdx && current !== "error");
  const isActive = current === step;
  const isError = current === "error" && isActive;

  return (
    <div
      className={`${styles.deployStepItem} ${isDone ? styles.deployStepDone : ""} ${isActive ? styles.deployStepActive : ""} ${isError ? styles.deployStepError : ""}`}
    >
      <span className={styles.deployStepIcon}>
        {isDone ? (
          <svg
            width="12"
            height="12"
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
        ) : isActive && !isError ? (
          <span className={styles.deployStepSpinner} />
        ) : isError ? (
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
          </svg>
        ) : (
          <span className={styles.deployStepPending} />
        )}
      </span>
      <span className={styles.deployStepLabel}>{label}</span>
    </div>
  );
};

interface BuilderDeployConfirmModalProps {
  open: boolean;
  loading: boolean;
  error: string | null;
  currentYaml: string;
  mergedYaml: string;
  onClose: () => void;
  onConfirm: () => void;
}

const BuilderDeployConfirmModal: React.FC<BuilderDeployConfirmModalProps> = ({
  open,
  loading,
  error,
  currentYaml,
  mergedYaml,
  onClose,
  onConfirm,
}) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const diffEditorRef = useRef<any>(null);
  const [diffChangeCount, setDiffChangeCount] = useState(0);

  if (!open) return null;

  return createPortal(
    <div className={styles.modalOverlay} onClick={onClose}>
      <div
        className={styles.deployDiffModal}
        onClick={(event) => event.stopPropagation()}
      >
        <div className={styles.deployModalHeader}>
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-warning)"
            strokeWidth="2"
          >
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span>Deploy to Router — Config Diff</span>
          <div style={{ flex: 1 }} />
          <span className={styles.deployDiffLabels}>
            <span className={styles.deployDiffLabelOld}>Current</span>
            <span
              style={{
                margin: "0 0.25rem",
                color: "var(--color-text-muted)",
              }}
            >
              &rarr;
            </span>
            <span className={styles.deployDiffLabelNew}>After Deploy</span>
          </span>
        </div>
        {!loading && !error && (
          <div className={styles.deployDiffNav}>
            <button
              className={styles.deployDiffNavBtn}
              title="Previous Change (↑)"
              onClick={() => {
                const editor = diffEditorRef.current;
                if (!editor) return;
                const navigation = editor.getLineChanges();
                if (!navigation || navigation.length === 0) return;
                const modifiedEditor = editor.getModifiedEditor();
                const currentLine =
                  modifiedEditor.getPosition()?.lineNumber ?? 1;
                for (let index = navigation.length - 1; index >= 0; index -= 1) {
                  const startLine =
                    navigation[index].modifiedStartLineNumber ||
                    navigation[index].originalStartLineNumber;
                  if (startLine < currentLine) {
                    modifiedEditor.revealLineInCenter(startLine);
                    modifiedEditor.setPosition({ lineNumber: startLine, column: 1 });
                    return;
                  }
                }
                const last = navigation[navigation.length - 1];
                const lastLine =
                  last.modifiedStartLineNumber || last.originalStartLineNumber;
                modifiedEditor.revealLineInCenter(lastLine);
                modifiedEditor.setPosition({ lineNumber: lastLine, column: 1 });
              }}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path
                  d="M4 10l4-4 4 4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
            <button
              className={styles.deployDiffNavBtn}
              title="Next Change (↓)"
              onClick={() => {
                const editor = diffEditorRef.current;
                if (!editor) return;
                const navigation = editor.getLineChanges();
                if (!navigation || navigation.length === 0) return;
                const modifiedEditor = editor.getModifiedEditor();
                const currentLine =
                  modifiedEditor.getPosition()?.lineNumber ?? 1;
                for (let index = 0; index < navigation.length; index += 1) {
                  const startLine =
                    navigation[index].modifiedStartLineNumber ||
                    navigation[index].originalStartLineNumber;
                  if (startLine > currentLine) {
                    modifiedEditor.revealLineInCenter(startLine);
                    modifiedEditor.setPosition({ lineNumber: startLine, column: 1 });
                    return;
                  }
                }
                const first = navigation[0];
                const firstLine =
                  first.modifiedStartLineNumber || first.originalStartLineNumber;
                modifiedEditor.revealLineInCenter(firstLine);
                modifiedEditor.setPosition({ lineNumber: firstLine, column: 1 });
              }}
            >
              <svg
                width="14"
                height="14"
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
            </button>
            <span className={styles.deployDiffNavInfo}>
              {diffChangeCount === 0
                ? "No changes"
                : `${diffChangeCount} change${diffChangeCount > 1 ? "s" : ""}`}
            </span>
          </div>
        )}
        <div className={styles.deployDiffBody}>
          {loading && (
            <div className={styles.deployDiffLoading}>
              <div className={styles.spinner} />
              Loading config diff...
            </div>
          )}
          {error && (
            <div className={styles.deployDiffError}>
              Failed to load preview: {error}
            </div>
          )}
          {!loading && !error && (
            <DiffEditor
              original={currentYaml}
              modified={mergedYaml}
              language="yaml"
              theme="vs-dark"
              onMount={(editor) => {
                diffEditorRef.current = editor;
                const updateCount = () => {
                  const changes = editor.getLineChanges();
                  setDiffChangeCount(changes?.length ?? 0);
                };
                const timer = setTimeout(updateCount, 500);
                try {
                  editor.onDidUpdateDiff(updateCount);
                } catch {
                  /* older Monaco */
                }
                return () => clearTimeout(timer);
              }}
              options={{
                readOnly: true,
                renderSideBySide: true,
                minimap: { enabled: true },
                scrollBeyondLastLine: false,
                fontSize: 12,
                lineNumbers: "on",
                wordWrap: "on",
                renderOverviewRuler: true,
                renderIndicators: true,
                contextmenu: false,
                scrollbar: {
                  verticalScrollbarSize: 8,
                  horizontalScrollbarSize: 8,
                },
              }}
            />
          )}
        </div>
        <div className={styles.deployModalFooter}>
          <p
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
              margin: 0,
              flex: 1,
            }}
          >
            A backup of the current config will be created before deployment.
          </p>
          <button className={styles.toolbarBtn} onClick={onClose}>
            Cancel
          </button>
          <button
            className={styles.toolbarBtnDeploy}
            onClick={onConfirm}
            disabled={loading || !!error}
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
            Deploy Now
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
};

interface BuilderDeployToastProps {
  deploying: boolean;
  deployStep: DeployStep | null;
  deployResult: DeployResult | null;
  onDismiss: () => void;
}

const BuilderDeployToast: React.FC<BuilderDeployToastProps> = ({
  deploying,
  deployStep,
  deployResult,
  onDismiss,
}) => {
  if (!deploying && !deployResult) return null;

  return createPortal(
    <div className={styles.deployToast}>
      {deploying && (
        <div className={styles.deployProgress}>
          <div className={styles.deployStepList}>
            <DeployStepItem
              step="validating"
              current={deployStep}
              label="Validating config"
            />
            <DeployStepItem
              step="backing_up"
              current={deployStep}
              label="Creating backup"
            />
            <DeployStepItem
              step="writing"
              current={deployStep}
              label="Writing config"
            />
            <DeployStepItem
              step="reloading"
              current={deployStep}
              label="Reloading runtime"
            />
          </div>
        </div>
      )}
      {deployResult && !deploying && (
        <div
          className={
            deployResult.status === "success"
              ? styles.deployResultSuccess
              : styles.deployResultError
          }
        >
          <div className={styles.deployResultIcon}>
            {deployResult.status === "success" ? (
              <svg
                width="16"
                height="16"
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
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
              </svg>
            )}
          </div>
          <span className={styles.deployResultMsg}>{deployResult.message}</span>
          <button className={styles.deployResultDismiss} onClick={onDismiss}>
            <svg
              width="12"
              height="12"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      )}
    </div>,
    document.body,
  );
};

const BuilderDragOverlay: React.FC<{ active: boolean }> = ({ active }) => {
  if (!active) return null;
  return createPortal(<div className={styles.dragOverlay} />, document.body);
};

export { BuilderDeployConfirmModal, BuilderDeployToast, BuilderDragOverlay };
