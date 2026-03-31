import React, { useCallback, useMemo, useState } from "react";

import styles from "./BuilderPage.module.css";

type OutputTab = "yaml" | "crd" | "dsl";

interface BuilderOutputPanelProps {
  open: boolean;
  width: number;
  yamlOutput: string;
  crdOutput: string;
  dslSource: string;
  dslTabLabel?: string;
  compileError: string | null;
  onDragStart: (event: React.MouseEvent) => void;
  onOpen: () => void;
  onClose: () => void;
}

const BuilderOutputPanel: React.FC<BuilderOutputPanelProps> = ({
  open,
  width,
  yamlOutput,
  crdOutput,
  dslSource,
  dslTabLabel = "DSL",
  compileError,
  onDragStart,
  onOpen,
  onClose,
}) => {
  const [outputTab, setOutputTab] = useState<OutputTab>("yaml");
  const [copied, setCopied] = useState(false);

  const outputContent = useMemo(() => {
    if (outputTab === "yaml") return yamlOutput || "";
    if (outputTab === "crd") return crdOutput || "";
    return dslSource || "";
  }, [outputTab, yamlOutput, crdOutput, dslSource]);

  const handleCopyOutput = useCallback(async () => {
    if (!outputContent) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(outputContent);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = outputContent;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = outputContent;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [outputContent]);

  if (!open) {
    return (
      <button
        className={styles.outputPanelToggle}
        onClick={onOpen}
        title="Show Output Panel"
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
            d="M10 2l-4 6 4 6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    );
  }

  return (
    <>
      <div className={styles.resizeHandle} onMouseDown={onDragStart}>
        <div className={styles.resizeHandleLine} />
      </div>
      <div className={styles.outputPanel} style={{ width }}>
        <div className={styles.outputPanelTabs}>
          <button
            className={
              outputTab === "yaml"
                ? styles.outputPanelTabActive
                : styles.outputPanelTab
            }
            onClick={() => setOutputTab("yaml")}
          >
            YAML
          </button>
          <button
            className={
              outputTab === "crd"
                ? styles.outputPanelTabActive
                : styles.outputPanelTab
            }
            onClick={() => setOutputTab("crd")}
          >
            CRD
          </button>
          <button
            className={
              outputTab === "dsl"
                ? styles.outputPanelTabActive
                : styles.outputPanelTab
            }
            onClick={() => setOutputTab("dsl")}
          >
            {dslTabLabel}
          </button>
          <div
            style={{
              marginLeft: "auto",
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-xs)",
            }}
          >
            {outputContent && (
              <button
                className={styles.outputPanelCopyBtn}
                onClick={handleCopyOutput}
                title="Copy to clipboard"
              >
                {copied ? (
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="var(--color-success)"
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
                    width="12"
                    height="12"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <rect x="5" y="5" width="9" height="9" rx="1" />
                    <path d="M2 11V2h9" strokeLinecap="round" />
                  </svg>
                )}
                {copied ? "Copied!" : "Copy"}
              </button>
            )}
            <button
              className={styles.outputPanelCloseBtn}
              onClick={onClose}
              title="Close panel"
            >
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
        </div>
        <div className={styles.outputPanelContent}>
          {compileError && (
            <div className={styles.outputPanelError}>{compileError}</div>
          )}
          {outputContent ? (
            <pre className={styles.outputPanelCode}>{outputContent}</pre>
          ) : (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>&#9889;</div>
              <div>
                {outputTab === "dsl" ? (
                  `${dslTabLabel} source is empty`
                ) : (
                  <>
                    Press <strong>Compile</strong> to generate{" "}
                    {outputTab.toUpperCase()} output
                  </>
                )}
              </div>
              <div
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                }}
              >
                Ctrl+Enter to compile
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export { BuilderOutputPanel };
