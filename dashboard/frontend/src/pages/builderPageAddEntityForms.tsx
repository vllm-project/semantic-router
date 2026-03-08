import React, { useCallback, useEffect, useMemo, useState } from "react";

import {
  BACKEND_TYPES,
  getSignalFieldSchema,
  PLUGIN_DESCRIPTIONS,
  PLUGIN_TYPES,
  SIGNAL_TYPES,
} from "@/lib/dslMutations";
import type { SignalType } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import {
  BackendIcon,
  CustomSelect,
  FieldEditor,
  GenericFieldsEditor,
  PluginIcon,
  SignalIcon,
} from "./builderPageFormPrimitives";
import { PluginSchemaEditor } from "./builderPageSharedDslEditors";

const AddSignalForm: React.FC<{
  onAdd: (
    signalType: string,
    name: string,
    fields: Record<string, unknown>,
  ) => void;
  onCancel: () => void;
}> = ({ onAdd, onCancel }) => {
  const [signalType, setSignalType] = useState<SignalType>("domain");
  const [name, setName] = useState("");
  const schema = useMemo(() => getSignalFieldSchema(signalType), [signalType]);
  const [fields, setFields] = useState<Record<string, unknown>>({});

  useEffect(() => {
    setFields({});
  }, [signalType]);

  const updateField = useCallback((key: string, value: unknown) => {
    setFields((previous) => ({ ...previous, [key]: value }));
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmed = name.trim().replace(/\s+/g, "_");
    if (!trimmed) return;
    onAdd(signalType, trimmed, fields);
  }, [signalType, name, fields, onAdd]);

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <SignalIcon className={styles.statIcon} />
          New Signal
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>
            Cancel
          </button>
          <button
            className={styles.toolbarBtnPrimary}
            onClick={handleSubmit}
            disabled={!name.trim()}
          >
            Create
          </button>
        </div>
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>
          Signal Type <span style={{ color: "var(--color-danger)" }}>*</span>
        </label>
        <CustomSelect
          value={signalType}
          options={[...SIGNAL_TYPES]}
          onChange={(value) => setSignalType(value as SignalType)}
        />
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>
          Name <span style={{ color: "var(--color-danger)" }}>*</span>
        </label>
        <input
          className={styles.fieldInput}
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="my_signal_name"
          autoFocus
        />
      </div>

      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Fields</span>
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
              value={fields[field.key]}
              onChange={(value) => updateField(field.key, value)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

const AddPluginForm: React.FC<{
  onAdd: (
    name: string,
    pluginType: string,
    fields: Record<string, unknown>,
  ) => void;
  onCancel: () => void;
}> = ({ onAdd, onCancel }) => {
  const [pluginType, setPluginType] = useState<string>(PLUGIN_TYPES[0]);
  const [name, setName] = useState("");
  const [fields, setFields] = useState<Record<string, unknown>>({
    enabled: true,
  });

  const handlePluginTypeChange = useCallback((value: string) => {
    setPluginType(value);
    setFields({ enabled: true });
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmed = name.trim().replace(/\s+/g, "_");
    if (!trimmed) return;
    onAdd(trimmed, pluginType, fields);
  }, [name, pluginType, fields, onAdd]);

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <PluginIcon className={styles.statIcon} />
          New Plugin
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>
            Cancel
          </button>
          <button
            className={styles.toolbarBtnPrimary}
            onClick={handleSubmit}
            disabled={!name.trim()}
          >
            Create
          </button>
        </div>
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>
          Plugin Type <span style={{ color: "var(--color-danger)" }}>*</span>
        </label>
        <CustomSelect
          value={pluginType}
          options={[...PLUGIN_TYPES]}
          onChange={handlePluginTypeChange}
        />
        {PLUGIN_DESCRIPTIONS[pluginType] && (
          <span
            style={{
              fontSize: "0.625rem",
              color: "var(--color-text-muted)",
              marginTop: "0.25rem",
            }}
          >
            {PLUGIN_DESCRIPTIONS[pluginType]}
          </span>
        )}
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>
          Name <span style={{ color: "var(--color-danger)" }}>*</span>
        </label>
        <input
          className={styles.fieldInput}
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="my_plugin"
          autoFocus
        />
      </div>

      <PluginSchemaEditor
        pluginType={pluginType}
        fields={fields}
        onUpdate={setFields}
      />
    </div>
  );
};

const AddBackendForm: React.FC<{
  onAdd: (
    backendType: string,
    name: string,
    fields: Record<string, unknown>,
  ) => void;
  onCancel: () => void;
}> = ({ onAdd, onCancel }) => {
  const [backendType, setBackendType] = useState<string>(BACKEND_TYPES[0]);
  const [name, setName] = useState("");
  const [fields, setFields] = useState<Record<string, unknown>>({});

  const handleSubmit = useCallback(() => {
    const trimmed = name.trim().replace(/\s+/g, "_");
    if (!trimmed) return;
    onAdd(backendType, trimmed, fields);
  }, [name, backendType, fields, onAdd]);

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <BackendIcon className={styles.statIcon} />
          New Backend
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>
            Cancel
          </button>
          <button
            className={styles.toolbarBtnPrimary}
            onClick={handleSubmit}
            disabled={!name.trim()}
          >
            Create
          </button>
        </div>
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>
          Backend Type <span style={{ color: "var(--color-danger)" }}>*</span>
        </label>
        <CustomSelect
          value={backendType}
          options={[...BACKEND_TYPES]}
          onChange={(value) => setBackendType(value)}
        />
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>
          Name <span style={{ color: "var(--color-danger)" }}>*</span>
        </label>
        <input
          className={styles.fieldInput}
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="my_backend"
          autoFocus
        />
      </div>

      <GenericFieldsEditor fields={fields} onUpdate={setFields} />
    </div>
  );
};

export { AddBackendForm, AddPluginForm, AddSignalForm };
