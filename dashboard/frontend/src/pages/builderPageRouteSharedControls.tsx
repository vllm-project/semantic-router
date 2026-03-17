import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { DSLFieldObject } from "@/types/dsl";
import styles from "./BuilderPage.module.css";

// ===================================================================
// Model Name Input (combo: dropdown + manual text input)
// ===================================================================

const ModelNameInput: React.FC<{
  value: string;
  availableModels: string[];
  onChange: (value: string) => void;
}> = ({ value, availableModels, onChange }) => {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState(value);
  const ref = useRef<HTMLDivElement>(null);

  // Sync external value
  useEffect(() => {
    setInputValue(value);
  }, [value]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node))
        setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const filteredModels = useMemo(() => {
    if (!inputValue.trim()) return availableModels;
    const lower = inputValue.toLowerCase();
    return availableModels.filter((m) => m.toLowerCase().includes(lower));
  }, [inputValue, availableModels]);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setInputValue(e.target.value);
      onChange(e.target.value);
      if (!open && availableModels.length > 0) setOpen(true);
    },
    [onChange, open, availableModels.length],
  );

  const handleSelect = useCallback(
    (model: string) => {
      setInputValue(model);
      onChange(model);
      setOpen(false);
    },
    [onChange],
  );

  return (
    <div ref={ref} style={{ flex: 1, position: "relative" }}>
      <div style={{ display: "flex", gap: "0" }}>
        <input
          className={styles.fieldInput}
          style={{
            flex: 1,
            fontSize: "var(--text-xs)",
            borderTopRightRadius: availableModels.length > 0 ? 0 : undefined,
            borderBottomRightRadius: availableModels.length > 0 ? 0 : undefined,
          }}
          value={inputValue}
          onChange={handleInputChange}
          onFocus={() => availableModels.length > 0 && setOpen(true)}
          placeholder="model name (e.g. qwen3:70b)"
        />
        {availableModels.length > 0 && (
          <button
            className={styles.modelDropdownBtn}
            onClick={() => setOpen(!open)}
            title="Select from known models"
            type="button"
          >
            <svg
              width="10"
              height="10"
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
        )}
      </div>
      {open && filteredModels.length > 0 && (
        <div className={styles.customSelectDropdown}>
          {filteredModels.map((m) => (
            <div
              key={m}
              className={
                m === value
                  ? styles.customSelectOptionActive
                  : styles.customSelectOption
              }
              onClick={() => handleSelect(m)}
            >
              {m === value && (
                <svg
                  className={styles.customSelectCheck}
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
              )}
              {m !== value && (
                <span className={styles.customSelectPlaceholder} />
              )}
              {m}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ===================================================================
// Manual Plugin Adder (for inline plugins not in templates)
// ===================================================================

const ManualPluginAdder: React.FC<{
  existingNames: Set<string>;
  onAdd: (name: string, fields?: DSLFieldObject) => void;
}> = ({ existingNames, onAdd }) => {
  const [name, setName] = useState("");

  const handleAdd = useCallback(() => {
    const n = name.trim();
    if (!n || existingNames.has(n)) return;
    onAdd(n);
    setName("");
  }, [name, existingNames, onAdd]);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--spacing-sm)",
        marginTop: "var(--spacing-xs)",
      }}
    >
      <input
        className={styles.fieldInput}
        style={{ flex: 1, fontSize: "var(--text-xs)" }}
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Add inline plugin by name..."
        onKeyDown={(e) => e.key === "Enter" && handleAdd()}
      />
      <button
        className={styles.toolbarBtn}
        onClick={handleAdd}
        disabled={!name.trim() || existingNames.has(name.trim())}
        style={{ padding: "0.25rem 0.5rem", fontSize: "var(--text-xs)" }}
      >
        + Add
      </button>
    </div>
  );
};

export { ManualPluginAdder, ModelNameInput };
