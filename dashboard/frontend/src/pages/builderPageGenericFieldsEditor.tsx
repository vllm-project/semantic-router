import React, { useCallback, useEffect, useState } from "react";

import type { DSLFieldObject, DSLFieldValue } from "@/types/dsl";

import styles from "./BuilderPage.module.css";
import { tryParseValue } from "./builderPageFieldControls";

export const GenericFieldsEditor: React.FC<{
  fields: DSLFieldObject;
  onUpdate: (fields: DSLFieldObject) => void;
}> = ({ fields, onUpdate }) => {
  const [localFields, setLocalFields] = useState<DSLFieldObject>(
    () => ({ ...fields }),
  );
  const [newKey, setNewKey] = useState("");

  useEffect(() => {
    setLocalFields({ ...fields });
  }, [fields]);

  const handleSave = useCallback(() => {
    onUpdate(localFields);
  }, [localFields, onUpdate]);

  const updateField = useCallback((key: string, rawValue: string) => {
    setLocalFields((prev) => {
      const parsed = tryParseValue(rawValue) as DSLFieldValue;
      return { ...prev, [key]: parsed };
    });
  }, []);

  const deleteField = useCallback((key: string) => {
    setLocalFields((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);

  const addField = useCallback(() => {
    const k = newKey.trim();
    if (!k || k in localFields) return;
    setLocalFields((prev) => ({ ...prev, [k]: "" }));
    setNewKey("");
  }, [newKey, localFields]);

  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>Fields</span>
        <button
          className={styles.toolbarBtnPrimary}
          onClick={handleSave}
          style={{ padding: "0.25rem 0.5rem", fontSize: "var(--text-xs)" }}
        >
          Save
        </button>
      </div>
      <div
        style={{
          padding: "var(--spacing-md)",
          display: "flex",
          flexDirection: "column",
          gap: "var(--spacing-sm)",
        }}
      >
        {Object.entries(localFields).map(([key, value]) => (
          <div
            key={key}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: "var(--spacing-sm)",
            }}
          >
            <span
              style={{
                minWidth: "120px",
                fontSize: "var(--text-xs)",
                color: "var(--color-text-secondary)",
                fontFamily: "var(--font-mono)",
                paddingTop: "0.5rem",
              }}
            >
              {key}
            </span>
            <input
              className={styles.fieldInput}
              style={{ flex: 1, fontSize: "var(--text-xs)" }}
              value={typeof value === "string" ? value : JSON.stringify(value)}
              onChange={(e) => updateField(key, e.target.value)}
            />
            <button
              className={styles.toolbarBtnDanger}
              onClick={() => deleteField(key)}
              style={{
                padding: "0.375rem",
                fontSize: "var(--text-xs)",
                flexShrink: 0,
              }}
              title="Remove field"
            >
              ×
            </button>
          </div>
        ))}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--spacing-sm)",
            marginTop: "var(--spacing-sm)",
          }}
        >
          <input
            className={styles.fieldInput}
            style={{ flex: 1, fontSize: "var(--text-xs)" }}
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            placeholder="New field name..."
            onKeyDown={(e) => e.key === "Enter" && addField()}
          />
          <button
            className={styles.toolbarBtn}
            onClick={addField}
            disabled={!newKey.trim()}
            style={{ padding: "0.375rem 0.5rem", fontSize: "var(--text-xs)" }}
          >
            + Add
          </button>
        </div>
      </div>
    </div>
  );
};
