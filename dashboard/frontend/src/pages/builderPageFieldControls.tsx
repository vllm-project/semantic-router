import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

import type { FieldSchema } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";

export const CustomSelect: React.FC<{
  value: string;
  options: string[];
  onChange: (value: string) => void;
  placeholder?: string;
}> = ({ value, options, onChange, placeholder = "— select —" }) => {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number; width: number }>({
    top: 0,
    left: 0,
    width: 0,
  });

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setPos({ top: rect.bottom + 4, left: rect.left, width: rect.width });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open]);

  return (
    <div className={styles.customSelect} ref={triggerRef}>
      <div
        className={
          open ? styles.customSelectTriggerOpen : styles.customSelectTrigger
        }
        onClick={() => setOpen(!open)}
      >
        <span>{value || placeholder}</span>
        <svg
          className={`${styles.customSelectChevron} ${open ? styles.customSelectChevronOpen : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      {open &&
        createPortal(
          <div
            ref={dropdownRef}
            className={styles.customSelectDropdown}
            style={{
              position: "fixed",
              top: pos.top,
              left: pos.left,
              width: pos.width,
            }}
          >
            {options.map((opt) => (
              <div
                key={opt}
                className={
                  opt === value
                    ? styles.customSelectOptionActive
                    : styles.customSelectOption
                }
                onClick={() => {
                  onChange(opt);
                  setOpen(false);
                }}
              >
                {opt === value ? (
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
                ) : (
                  <span className={styles.customSelectPlaceholder} />
                )}
                {opt || "(none)"}
              </div>
            ))}
          </div>,
          document.body,
        )}
    </div>
  );
};

export const FieldEditor: React.FC<{
  schema: FieldSchema;
  value: unknown;
  onChange: (value: unknown) => void;
}> = ({ schema, value, onChange }) => {
  switch (schema.type) {
    case "string":
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label}{" "}
            {schema.required && (
              <span style={{ color: "var(--color-danger)" }}>*</span>
            )}
          </label>
          <input
            className={styles.fieldInput}
            value={(value as string) ?? ""}
            onChange={(e) => onChange(e.target.value)}
            placeholder={schema.placeholder}
          />
          {schema.description && (
            <span
              style={{ fontSize: "0.625rem", color: "var(--color-text-muted)" }}
            >
              {schema.description}
            </span>
          )}
        </div>
      );
    case "number":
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label}{" "}
            {schema.required && (
              <span style={{ color: "var(--color-danger)" }}>*</span>
            )}
          </label>
          <input
            className={styles.fieldInput}
            type="number"
            step="any"
            value={value !== undefined && value !== null ? String(value) : ""}
            onChange={(e) => {
              const v = e.target.value;
              onChange(v === "" ? undefined : Number(v));
            }}
            placeholder={schema.placeholder}
          />
        </div>
      );
    case "boolean":
      return (
        <div className={styles.fieldGroup}>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={!!value}
              onChange={(e) => onChange(e.target.checked)}
              style={{ accentColor: "var(--color-primary)" }}
            />
            <span
              className={styles.fieldLabel}
              style={{ textTransform: "none" }}
            >
              {schema.label}
            </span>
          </label>
        </div>
      );
    case "select":
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label}{" "}
            {schema.required && (
              <span style={{ color: "var(--color-danger)" }}>*</span>
            )}
          </label>
          <CustomSelect
            value={(value as string) ?? ""}
            options={schema.options ?? []}
            onChange={(v) => onChange(v || undefined)}
            placeholder="— select —"
          />
        </div>
      );
    case "string[]":
      return (
        <StringArrayEditor
          label={schema.label}
          required={schema.required}
          value={(value as string[]) ?? []}
          onChange={onChange}
          placeholder={schema.placeholder}
        />
      );
    case "number[]": {
      let arr: number[] = [];
      if (Array.isArray(value)) {
        arr = value.map(Number).filter((n) => !isNaN(n));
      } else if (typeof value === "string") {
        try {
          const parsed = JSON.parse(value);
          if (Array.isArray(parsed))
            arr = parsed.map(Number).filter((n) => !isNaN(n));
        } catch {
          /* ignore */
        }
      }
      return (
        <NumberArrayEditor
          label={schema.label}
          required={schema.required}
          value={arr}
          onChange={onChange}
          placeholder={schema.placeholder}
          description={schema.description}
        />
      );
    }
    case "json":
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label}{" "}
            {schema.required && (
              <span style={{ color: "var(--color-danger)" }}>*</span>
            )}
          </label>
          <textarea
            className={styles.fieldTextarea}
            value={
              value !== undefined && value !== null
                ? typeof value === "string"
                  ? value
                  : JSON.stringify(value, null, 2)
                : ""
            }
            onChange={(e) => {
              try {
                onChange(JSON.parse(e.target.value));
              } catch {
                onChange(e.target.value);
              }
            }}
            rows={3}
            style={{ fontSize: "var(--text-xs)" }}
          />
          {schema.description && (
            <span
              style={{ fontSize: "0.625rem", color: "var(--color-text-muted)" }}
            >
              {schema.description}
            </span>
          )}
        </div>
      );
    default:
      return null;
  }
};

export const StringArrayEditor: React.FC<{
  label: string;
  required?: boolean;
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
}> = ({ label, required, value, onChange, placeholder }) => {
  const [inputValue, setInputValue] = useState("");

  const addItem = useCallback(() => {
    const v = inputValue.trim();
    if (v && !value.includes(v)) {
      onChange([...value, v]);
      setInputValue("");
    }
  }, [inputValue, value, onChange]);

  const removeItem = useCallback(
    (idx: number) => {
      onChange(value.filter((_, i) => i !== idx));
    },
    [value, onChange],
  );

  return (
    <div className={styles.fieldGroup}>
      <label className={styles.fieldLabel}>
        {label}{" "}
        {required && <span style={{ color: "var(--color-danger)" }}>*</span>}
      </label>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.25rem",
          minHeight: "1.5rem",
        }}
      >
        {value.map((item, idx) => (
          <span
            key={idx}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.25rem",
              padding: "0.125rem 0.5rem",
              fontSize: "var(--text-xs)",
              background: "var(--color-bg-tertiary)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-mono)",
              color: "var(--color-text)",
            }}
          >
            {item}
            <button
              onClick={() => removeItem(idx)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: 0,
                color: "var(--color-text-muted)",
                fontSize: "0.75rem",
                lineHeight: 1,
              }}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div style={{ display: "flex", gap: "var(--spacing-sm)" }}>
        <input
          className={styles.fieldInput}
          style={{ flex: 1 }}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={placeholder}
          onKeyDown={(e) =>
            e.key === "Enter" && (e.preventDefault(), addItem())
          }
        />
        <button
          className={styles.toolbarBtn}
          onClick={addItem}
          disabled={!inputValue.trim()}
          style={{ padding: "0.375rem 0.5rem", fontSize: "var(--text-xs)" }}
        >
          + Add
        </button>
      </div>
    </div>
  );
};

export const NumberArrayEditor: React.FC<{
  label: string;
  required?: boolean;
  value: number[];
  onChange: (value: number[]) => void;
  placeholder?: string;
  description?: string;
}> = ({ label, required, value, onChange, placeholder, description }) => {
  const [inputValue, setInputValue] = useState("");

  const addItem = useCallback(() => {
    const v = inputValue.trim();
    if (v === "") return;
    const num = Number(v);
    if (isNaN(num)) return;
    onChange([...value, num]);
    setInputValue("");
  }, [inputValue, value, onChange]);

  const removeItem = useCallback(
    (idx: number) => {
      onChange(value.filter((_, i) => i !== idx));
    },
    [value, onChange],
  );

  return (
    <div className={styles.fieldGroup}>
      <label className={styles.fieldLabel}>
        {label}{" "}
        {required && <span style={{ color: "var(--color-danger)" }}>*</span>}
      </label>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.25rem",
          minHeight: "1.75rem",
        }}
      >
        <span
          style={{
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            fontFamily: "var(--font-mono)",
          }}
        >
          [
        </span>
        {value.map((item, idx) => (
          <span
            key={idx}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.25rem",
              padding: "0.125rem 0.5rem",
              fontSize: "var(--text-xs)",
              background: "var(--color-bg-tertiary)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-mono)",
              color: "var(--color-text)",
            }}
          >
            {item}
            <button
              onClick={() => removeItem(idx)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: 0,
                color: "var(--color-text-muted)",
                fontSize: "0.75rem",
                lineHeight: 1,
              }}
            >
              ×
            </button>
          </span>
        ))}
        <span
          style={{
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            fontFamily: "var(--font-mono)",
          }}
        >
          ]
        </span>
      </div>
      <div style={{ display: "flex", gap: "var(--spacing-sm)" }}>
        <input
          className={styles.fieldInput}
          style={{ flex: 1 }}
          type="number"
          step="any"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={placeholder}
          onKeyDown={(e) =>
            e.key === "Enter" && (e.preventDefault(), addItem())
          }
        />
        <button
          className={styles.toolbarBtn}
          onClick={addItem}
          disabled={!inputValue.trim() || isNaN(Number(inputValue))}
          style={{ padding: "0.375rem 0.5rem", fontSize: "var(--text-xs)" }}
        >
          + Add
        </button>
      </div>
      {description && (
        <span
          style={{ fontSize: "0.625rem", color: "var(--color-text-muted)" }}
        >
          {description}
        </span>
      )}
    </div>
  );
};

export function tryParseValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (trimmed === "") return "";
  if (/^-?\d+$/.test(trimmed)) return parseInt(trimmed, 10);
  if (/^-?\d+\.\d+$/.test(trimmed)) return parseFloat(trimmed);
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === "object") return parsed;
  } catch {
    /* not JSON */
  }
  return raw;
}
