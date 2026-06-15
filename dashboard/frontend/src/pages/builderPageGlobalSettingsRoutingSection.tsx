import React from "react";

import type { DSLFieldObject, DSLFieldValue } from "@/types/dsl";
import styles from "./BuilderPage.module.css";
import type { EditableListener } from "./builderPageGlobalSettingsSupport";
import { getBool, getNum, getStr } from "./builderPageGlobalSettingsSupport";

interface GlobalSettingsRoutingSectionProps {
  local: DSLFieldObject;
  collapsedSections: Record<string, boolean>;
  modelSelection: DSLFieldObject;
  reasoningFamilies: DSLFieldObject;
  looper: DSLFieldObject;
  listeners: EditableListener[];
  onToggleSection: (key: string) => void;
  onSetField: (key: string, value: DSLFieldValue) => void;
  onSetNestedField: (
    parentKey: string,
    childKey: string,
    value: DSLFieldValue,
  ) => void;
  onUpdateListener: (
    index: number,
    field: keyof EditableListener,
    value: string | number,
  ) => void;
  onAddListener: () => void;
  onRemoveListener: (index: number) => void;
}

const GlobalSettingsRoutingSection: React.FC<
  GlobalSettingsRoutingSectionProps
> = ({
  local,
  collapsedSections,
  modelSelection,
  reasoningFamilies,
  looper,
  listeners,
  onToggleSection,
  onSetField,
  onSetNestedField,
  onUpdateListener,
  onAddListener,
  onRemoveListener,
}) => {
  return (
    <div className={styles.gsSection}>
      <div
        className={styles.gsSectionHeader}
        onClick={() => onToggleSection("routing")}
      >
        <svg
          className={styles.gsSectionChevron}
          data-open={!collapsedSections["routing"]}
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M3 2l4 3-4 3" />
        </svg>
        <span className={styles.gsSectionTitle}>Routing</span>
      </div>
      {!collapsedSections["routing"] && (
        <div className={styles.gsSectionBody}>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Listeners</label>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.5rem",
                width: "100%",
              }}
            >
              {listeners.length === 0 && (
                <div
                  style={{
                    fontSize: "var(--text-xs)",
                    color: "var(--color-text-muted)",
                  }}
                >
                  No listeners configured yet. Add at least one listener to
                  expose the router.
                </div>
              )}
              {listeners.map((listener, index) => (
                <div
                  key={`${listener.name}-${index}`}
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "minmax(8rem, 1fr) minmax(8rem, 1fr) 6rem 7rem auto",
                    gap: "0.5rem",
                    alignItems: "center",
                  }}
                >
                  <input
                    className={styles.fieldInput}
                    value={listener.name}
                    onChange={(event) =>
                      onUpdateListener(index, "name", event.target.value)
                    }
                    placeholder="http-8899"
                  />
                  <input
                    className={styles.fieldInput}
                    value={listener.address}
                    onChange={(event) =>
                      onUpdateListener(index, "address", event.target.value)
                    }
                    placeholder="0.0.0.0"
                  />
                  <input
                    className={styles.fieldInput}
                    type="number"
                    min={1}
                    max={65535}
                    value={listener.port}
                    onChange={(event) =>
                      onUpdateListener(
                        index,
                        "port",
                        parseInt(event.target.value, 10) || 0,
                      )
                    }
                  />
                  <input
                    className={styles.fieldInput}
                    value={listener.timeout ?? "300s"}
                    onChange={(event) =>
                      onUpdateListener(index, "timeout", event.target.value)
                    }
                    placeholder="300s"
                  />
                  <button
                    className={styles.toolbarBtn}
                    style={{
                      padding: "0.2rem 0.5rem",
                      fontSize: "var(--text-xs)",
                    }}
                    onClick={() => onRemoveListener(index)}
                    disabled={listeners.length <= 1}
                    title={
                      listeners.length <= 1
                        ? "At least one listener is required"
                        : "Remove listener"
                    }
                  >
                    Remove
                  </button>
                </div>
              ))}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: "0.75rem",
                }}
              >
                <button
                  className={styles.toolbarBtn}
                  style={{
                    alignSelf: "flex-start",
                    padding: "0.2rem 0.5rem",
                    fontSize: "var(--text-xs)",
                  }}
                  onClick={onAddListener}
                >
                  + Add Listener
                </button>
                <span
                  style={{
                    fontSize: "var(--text-xs)",
                    color: "var(--color-text-muted)",
                  }}
                >
                  These listeners are emitted into `config.yaml` and control
                  the ports OpenClaw and Envoy will target.
                </span>
              </div>
            </div>
          </div>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Default Model</label>
            <input
              className={styles.fieldInput}
              value={getStr(local, "default_model")}
              onChange={(event) =>
                onSetField("default_model", event.target.value)
              }
              placeholder="qwen2.5:3b"
            />
          </div>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Strategy</label>
            <div className={styles.gsRadioGroup}>
              {["priority", "confidence"].map((strategy) => (
                <label key={strategy} className={styles.gsRadio}>
                  <input
                    type="radio"
                    name="gs-strategy"
                    checked={getStr(local, "strategy") === strategy}
                    onChange={() => onSetField("strategy", strategy)}
                  />
                  <span>{strategy}</span>
                </label>
              ))}
            </div>
          </div>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Default Reasoning Effort</label>
            <div className={styles.gsRadioGroup}>
              {["low", "medium", "high"].map((effort) => (
                <label key={effort} className={styles.gsRadio}>
                  <input
                    type="radio"
                    name="gs-effort"
                    checked={
                      getStr(local, "default_reasoning_effort") === effort
                    }
                    onChange={() =>
                      onSetField("default_reasoning_effort", effort)
                    }
                  />
                  <span>{effort}</span>
                </label>
              ))}
            </div>
          </div>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Model Selection</label>
            <div className={styles.gsInlineRow}>
              <label className={styles.gsCheckbox}>
                <input
                  type="checkbox"
                  checked={getBool(modelSelection, "enabled")}
                  onChange={(event) =>
                    onSetNestedField(
                      "model_selection",
                      "enabled",
                      event.target.checked,
                    )
                  }
                />
                <span>Enabled</span>
              </label>
              {getBool(modelSelection, "enabled") && (
                <div className={styles.gsInlineField}>
                  <span className={styles.gsSmallLabel}>Method:</span>
                  <input
                    className={styles.fieldInput}
                    style={{ width: "8rem" }}
                    value={getStr(modelSelection, "method")}
                    onChange={(event) =>
                      onSetNestedField(
                        "model_selection",
                        "method",
                        event.target.value,
                      )
                    }
                    placeholder="knn"
                  />
                </div>
              )}
            </div>
          </div>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Reasoning Families</label>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.25rem",
                width: "100%",
              }}
            >
              {Object.entries(reasoningFamilies).map(([name, value]) => {
                const entry =
                  value && typeof value === "object" && !Array.isArray(value)
                    ? (value as DSLFieldObject)
                    : {};
                return (
                  <div
                    key={name}
                    style={{
                      display: "flex",
                      gap: "0.375rem",
                      alignItems: "center",
                    }}
                  >
                    <input
                      className={styles.fieldInput}
                      style={{ width: "5rem" }}
                      value={name}
                      readOnly
                      title="Family name"
                    />
                    <span className={styles.gsSmallLabel}>type:</span>
                    <input
                      className={styles.fieldInput}
                      style={{ width: "10rem" }}
                      value={getStr(entry, "type")}
                      onChange={(event) => {
                        const families = { ...reasoningFamilies };
                        families[name] = {
                          ...entry,
                          type: event.target.value,
                        };
                        onSetField("reasoning_families", families);
                      }}
                      placeholder="chat_template_kwargs"
                    />
                    <span className={styles.gsSmallLabel}>param:</span>
                    <input
                      className={styles.fieldInput}
                      style={{ width: "7rem" }}
                      value={getStr(entry, "parameter")}
                      onChange={(event) => {
                        const families = { ...reasoningFamilies };
                        families[name] = {
                          ...entry,
                          parameter: event.target.value,
                        };
                        onSetField("reasoning_families", families);
                      }}
                      placeholder="thinking"
                    />
                    <button
                      className={styles.toolbarBtn}
                      style={{
                        padding: "0.2rem 0.4rem",
                        fontSize: "var(--text-xs)",
                      }}
                      onClick={() => {
                        const families = { ...reasoningFamilies };
                        delete families[name];
                        onSetField("reasoning_families", families);
                      }}
                      title="Remove"
                    >
                      &times;
                    </button>
                  </div>
                );
              })}
              <button
                className={styles.toolbarBtn}
                style={{
                  alignSelf: "flex-start",
                  padding: "0.2rem 0.5rem",
                  fontSize: "var(--text-xs)",
                }}
                onClick={() => {
                  const families = { ...reasoningFamilies };
                  const nextName = `family_${Object.keys(families).length + 1}`;
                  families[nextName] = {
                    type: "chat_template_kwargs",
                    parameter: "thinking",
                  };
                  onSetField("reasoning_families", families);
                }}
              >
                + Add Family
              </button>
            </div>
          </div>
          <div className={styles.gsRow}>
            <label className={styles.gsLabel}>Looper Endpoint</label>
            <input
              className={styles.fieldInput}
              value={getStr(looper, "endpoint")}
              onChange={(event) =>
                onSetNestedField("looper", "endpoint", event.target.value)
              }
              placeholder="http://looper:8080"
            />
          </div>
          {getStr(looper, "endpoint") && (
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Looper Timeout (s)</label>
              <input
                className={styles.fieldInput}
                type="number"
                style={{ width: "6rem" }}
                value={getNum(looper, "timeout_seconds", 30)}
                onChange={(event) =>
                  onSetNestedField(
                    "looper",
                    "timeout_seconds",
                    parseInt(event.target.value, 10) || 0,
                  )
                }
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export { GlobalSettingsRoutingSection };
