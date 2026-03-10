import React, { useCallback, useEffect, useMemo, useState } from "react";

import type { ASTSignalDecl } from "@/types/dsl";
import {
  getAlgorithmFieldSchema,
  getPluginFieldSchema,
  getSignalFieldSchema,
  PLUGIN_DESCRIPTIONS,
  serializeFields,
} from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import { FieldEditor, tryParseValue } from "./builderPageFormPrimitives";

function generateSignalDslPreview(
  signalType: string,
  signalName: string,
  fields: Record<string, unknown>,
): string {
  const body = serializeFields(fields);
  if (!body.trim()) {
    return `SIGNAL ${signalType} ${signalName} {}`;
  }
  return `SIGNAL ${signalType} ${signalName} {\n${body}\n}`;
}

function generateGlobalDslPreview(fields: Record<string, unknown>): string {
  const body = serializeFields(fields, "  ", { blankLineBefore: true });
  if (!body.trim()) return "GLOBAL {}";
  return `GLOBAL {\n${body}\n}`;
}

const DslPreviewPanel: React.FC<{
  title?: string;
  dslText: string;
}> = ({ title = "DSL Preview", dslText }) => {
  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>{title}</span>
      </div>
      <pre className={styles.dslPreviewCode}>{dslText}</pre>
    </div>
  );
};

const ExtraFieldsEditor: React.FC<{
  fields: Record<string, unknown>;
  schemaKeys: string[];
  onUpdate: (fields: Record<string, unknown>) => void;
}> = ({ fields, schemaKeys, onUpdate }) => {
  const extraEntries = useMemo(() => {
    const known = new Set(schemaKeys);
    return Object.entries(fields).filter(([key]) => !known.has(key));
  }, [fields, schemaKeys]);

  const [newKey, setNewKey] = useState("");

  const updateField = useCallback(
    (key: string, rawValue: string) => {
      const parsed = tryParseValue(rawValue);
      onUpdate({ ...fields, [key]: parsed });
    },
    [fields, onUpdate],
  );

  const deleteField = useCallback(
    (key: string) => {
      const next = { ...fields };
      delete next[key];
      onUpdate(next);
    },
    [fields, onUpdate],
  );

  const addField = useCallback(() => {
    const key = newKey.trim();
    if (!key || key in fields) return;
    onUpdate({ ...fields, [key]: "" });
    setNewKey("");
  }, [newKey, fields, onUpdate]);

  return (
    <>
      {extraEntries.map(([key, value]) => (
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
              minWidth: "100px",
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
            onChange={(event) => updateField(key, event.target.value)}
          />
          <button
            className={styles.toolbarBtnDanger}
            onClick={() => deleteField(key)}
            style={{
              padding: "0.375rem",
              fontSize: "var(--text-xs)",
              flexShrink: 0,
            }}
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
        }}
      >
        <input
          className={styles.fieldInput}
          style={{ flex: 1, fontSize: "var(--text-xs)" }}
          value={newKey}
          onChange={(event) => setNewKey(event.target.value)}
          placeholder="Add custom field..."
          onKeyDown={(event) => event.key === "Enter" && addField()}
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
    </>
  );
};

const AlgorithmSchemaEditor: React.FC<{
  algoType: string;
  fields: Record<string, unknown>;
  onUpdate: (fields: Record<string, unknown>) => void;
}> = ({ algoType, fields, onUpdate }) => {
  const schema = useMemo(() => getAlgorithmFieldSchema(algoType), [algoType]);

  const updateField = useCallback(
    (key: string, value: unknown) => {
      onUpdate({ ...fields, [key]: value });
    },
    [fields, onUpdate],
  );

  if (schema.length === 0) {
    return (
      <div
        style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}
      >
        No configurable fields for this algorithm type.
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--spacing-md)",
      }}
    >
      {schema.map((field) => (
        <FieldEditor
          key={field.key}
          schema={field}
          value={fields[field.key]}
          onChange={(value) => updateField(field.key, value)}
        />
      ))}
      <ExtraFieldsEditor
        fields={fields}
        schemaKeys={schema.map((field) => field.key)}
        onUpdate={onUpdate}
      />
    </div>
  );
};

const SignalEditorForm: React.FC<{
  signal: ASTSignalDecl;
  onUpdate: (fields: Record<string, unknown>) => void;
}> = ({ signal, onUpdate }) => {
  const schema = useMemo(
    () => getSignalFieldSchema(signal.signalType),
    [signal.signalType],
  );
  const [localFields, setLocalFields] = useState<Record<string, unknown>>(
    () => ({ ...signal.fields }),
  );

  useEffect(() => {
    setLocalFields({ ...signal.fields });
  }, [signal.name, signal.signalType, signal.fields]);

  const updateField = useCallback((key: string, value: unknown) => {
    setLocalFields((previous) => ({ ...previous, [key]: value }));
  }, []);

  const handleSave = useCallback(() => {
    onUpdate(localFields);
  }, [localFields, onUpdate]);

  const dslPreview = useMemo(
    () => generateSignalDslPreview(signal.signalType, signal.name, localFields),
    [signal.signalType, signal.name, localFields],
  );

  return (
    <>
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
            gap: "var(--spacing-md)",
          }}
        >
          {schema.map((field) => (
            <FieldEditor
              key={field.key}
              schema={field}
              value={localFields[field.key]}
              onChange={(value) => updateField(field.key, value)}
            />
          ))}
        </div>
      </div>
      <DslPreviewPanel dslText={dslPreview} />
    </>
  );
};

const PluginSchemaEditor: React.FC<{
  pluginType: string;
  pluginName?: string;
  fields: Record<string, unknown>;
  onUpdate: (fields: Record<string, unknown>) => void;
  buffered?: boolean;
  compact?: boolean;
}> = ({
  pluginType,
  pluginName,
  fields,
  onUpdate,
  buffered = false,
  compact = false,
}) => {
  const schema = useMemo(() => getPluginFieldSchema(pluginType), [pluginType]);
  const [localFields, setLocalFields] = useState<Record<string, unknown>>(
    () => ({ ...fields }),
  );

  useEffect(() => {
    setLocalFields({ ...fields });
  }, [fields]);

  const currentFields = buffered ? localFields : fields;
  const doUpdate = buffered ? setLocalFields : onUpdate;

  const updateField = useCallback(
    (key: string, value: unknown) => {
      doUpdate({ ...currentFields, [key]: value });
    },
    [currentFields, doUpdate],
  );

  const handleSave = useCallback(() => {
    if (buffered) onUpdate(localFields);
  }, [buffered, localFields, onUpdate]);

  if (compact) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "var(--spacing-sm)",
          padding: "var(--spacing-sm) 0",
        }}
      >
        {pluginName && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-sm)",
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "var(--text-xs)",
                color: "var(--color-primary)",
                fontWeight: 600,
              }}
            >
              {pluginName}
            </span>
            <span
              style={{ fontSize: "0.625rem", color: "var(--color-text-muted)" }}
            >
              {pluginType}
            </span>
          </div>
        )}
        {schema.map((field) => (
          <FieldEditor
            key={field.key}
            schema={field}
            value={currentFields[field.key]}
            onChange={(value) => updateField(field.key, value)}
          />
        ))}
        <ExtraFieldsEditor
          fields={currentFields}
          schemaKeys={schema.map((field) => field.key)}
          onUpdate={doUpdate}
        />
      </div>
    );
  }

  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>
          {pluginName ?? "Configuration"}
          {PLUGIN_DESCRIPTIONS[pluginType] && (
            <span
              style={{
                fontWeight: 400,
                fontSize: "0.625rem",
                color: "var(--color-text-muted)",
                marginLeft: "0.5rem",
              }}
            >
              — {PLUGIN_DESCRIPTIONS[pluginType]}
            </span>
          )}
        </span>
        {buffered && (
          <button
            className={styles.toolbarBtnPrimary}
            onClick={handleSave}
            style={{ padding: "0.25rem 0.5rem", fontSize: "var(--text-xs)" }}
          >
            Save
          </button>
        )}
      </div>
      <div
        style={{
          padding: "var(--spacing-md)",
          display: "flex",
          flexDirection: "column",
          gap: "var(--spacing-md)",
        }}
      >
        {schema.map((field) => (
          <FieldEditor
            key={field.key}
            schema={field}
            value={currentFields[field.key]}
            onChange={(value) => updateField(field.key, value)}
          />
        ))}
        <ExtraFieldsEditor
          fields={currentFields}
          schemaKeys={schema.map((field) => field.key)}
          onUpdate={doUpdate}
        />
      </div>
    </div>
  );
};

export {
  AlgorithmSchemaEditor,
  DslPreviewPanel,
  ExtraFieldsEditor,
  generateGlobalDslPreview,
  generateSignalDslPreview,
  PluginSchemaEditor,
  SignalEditorForm,
};
