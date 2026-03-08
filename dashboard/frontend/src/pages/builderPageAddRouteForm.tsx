import React, { useCallback, useMemo, useState } from "react";

import ExpressionBuilder from "@/components/ExpressionBuilder";
import {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
} from "@/lib/dslMutations";
import type {
  RouteAlgoInput,
  RouteInput,
  RouteModelInput,
  RoutePluginInput,
} from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import { CustomSelect, RouteIcon } from "./builderPageFormPrimitives";
import { AlgorithmSchemaEditor, PluginSchemaEditor } from "./builderPageEntityForms";
import {
  RouteDslPreviewPanel,
  generateRouteDslPreview,
  validateRouteInput,
} from "./builderPageRoutePreview";
import { ModelNameInput, ManualPluginAdder } from "./builderPageRouteSharedControls";
import type { AvailablePlugin, AvailableSignal } from "./builderPageTypes";

// ===================================================================
// Add Route Form
// ===================================================================

const AddRouteForm: React.FC<{
  onAdd: (name: string, input: RouteInput) => void;
  onCancel: () => void;
  availableSignals: AvailableSignal[];
  availablePlugins: AvailablePlugin[];
  availableModels: string[];
}> = ({
  onAdd,
  onCancel,
  availableSignals,
  availablePlugins,
  availableModels,
}) => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [priority, setPriority] = useState(100);
  const [whenExpr, setWhenExpr] = useState("");
  const [models, setModels] = useState<RouteModelInput[]>([{ model: "" }]);
  const [algorithm, setAlgorithm] = useState<RouteAlgoInput | undefined>(
    undefined,
  );
  const [plugins, setPlugins] = useState<RoutePluginInput[]>([]);

  const handleSubmit = useCallback(() => {
    const n = name.trim().replace(/\s+/g, "_");
    if (!n) return;
    onAdd(n, {
      description: description.trim() || undefined,
      priority,
      when: whenExpr.trim() || undefined,
      models: models.filter((m) => m.model.trim()),
      algorithm: algorithm?.algoType ? algorithm : undefined,
      plugins,
    });
  }, [
    name,
    description,
    priority,
    whenExpr,
    models,
    algorithm,
    plugins,
    onAdd,
  ]);

  const addModel = useCallback(() => {
    setModels((prev) => [...prev, { model: "" }]);
  }, []);

  const removeModel = useCallback((idx: number) => {
    setModels((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const updateModel = useCallback(
    (idx: number, patch: Partial<RouteModelInput>) => {
      setModels((prev) =>
        prev.map((m, i) => (i === idx ? { ...m, ...patch } : m)),
      );
    },
    [],
  );

  // Expression builder: tree-based, managed by ExpressionBuilder component

  // Generate DSL preview & validation for AddRouteForm
  const routeName = useMemo(
    () => name.trim().replace(/\s+/g, "_") || "new_route",
    [name],
  );
  const dslPreview = useMemo(
    () =>
      generateRouteDslPreview(
        routeName,
        description,
        priority,
        whenExpr,
        models,
        algorithm,
        plugins,
      ),
    [routeName, description, priority, whenExpr, models, algorithm, plugins],
  );
  const validationIssues = useMemo(
    () => validateRouteInput(name.trim(), models, algorithm, plugins),
    [name, models, algorithm, plugins],
  );

  const activePluginNames = useMemo(
    () => new Set(plugins.map((p) => p.name)),
    [plugins],
  );

  const togglePlugin = useCallback((pluginName: string) => {
    setPlugins((prev) => {
      const exists = prev.find((p) => p.name === pluginName);
      if (exists) return prev.filter((p) => p.name !== pluginName);
      return [...prev, { name: pluginName }];
    });
  }, []);

  const updatePluginFields = useCallback(
    (pluginName: string, fields: Record<string, unknown>) => {
      setPlugins((prev) =>
        prev.map((p) => (p.name === pluginName ? { ...p, fields } : p)),
      );
    },
    [],
  );

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <RouteIcon className={styles.statIcon} />
          New Route
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

      {/* Basic fields */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Route Configuration</span>
        </div>
        <div
          style={{
            padding: "var(--spacing-md)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--spacing-md)",
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: "var(--spacing-md)",
            }}
          >
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>
                Name <span style={{ color: "var(--color-danger)" }}>*</span>
              </label>
              <input
                className={styles.fieldInput}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="my_route"
                autoFocus
              />
            </div>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>
                Priority <span style={{ color: "var(--color-danger)" }}>*</span>
              </label>
              <input
                className={styles.fieldInput}
                type="number"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
                style={{ width: "100px" }}
              />
            </div>
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.fieldLabel}>Description</label>
            <input
              className={styles.fieldInput}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Route description..."
            />
          </div>
        </div>
      </div>

      {/* WHEN Expression Builder */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>
            WHEN (Expression Builder)
          </span>
        </div>
        <div
          style={{
            padding: "var(--spacing-md)",
            minHeight: "350px",
            maxHeight: "50vh",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <ExpressionBuilder
            value={whenExpr}
            onChange={setWhenExpr}
            availableSignals={availableSignals}
          />
        </div>
      </div>

      {/* Models */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>
            Models ({models.length})
          </span>
          <button
            className={styles.toolbarBtn}
            onClick={addModel}
            style={{ padding: "0.25rem 0.5rem", fontSize: "var(--text-xs)" }}
          >
            + Add
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
          {models.map((m, idx) => (
            <div key={idx} className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelIndex}>{idx + 1}</span>
                <ModelNameInput
                  value={m.model}
                  availableModels={availableModels}
                  onChange={(v) => updateModel(idx, { model: v })}
                />
                {models.length > 1 && (
                  <button
                    className={styles.toolbarBtnDanger}
                    onClick={() => removeModel(idx)}
                    style={{
                      padding: "0.25rem 0.5rem",
                      fontSize: "var(--text-xs)",
                      flexShrink: 0,
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
              <div className={styles.modelAttrs}>
                <label className={styles.modelAttrCheck}>
                  <input
                    type="checkbox"
                    checked={m.reasoning ?? false}
                    onChange={(e) =>
                      updateModel(idx, {
                        reasoning: e.target.checked || undefined,
                      })
                    }
                    style={{ accentColor: "var(--color-primary)" }}
                  />
                  reasoning
                </label>
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>effort:</span>
                  <div style={{ minWidth: "90px" }}>
                    <CustomSelect
                      value={m.effort ?? ""}
                      options={["", "low", "medium", "high"]}
                      onChange={(v) =>
                        updateModel(idx, { effort: v || undefined })
                      }
                      placeholder="—"
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Algorithm</span>
          {!algorithm && (
            <button
              className={styles.toolbarBtn}
              onClick={() =>
                setAlgorithm({ algoType: "confidence", fields: {} })
              }
              style={{ padding: "0.25rem 0.5rem", fontSize: "var(--text-xs)" }}
            >
              + Add
            </button>
          )}
          {algorithm && (
            <button
              className={styles.toolbarBtnDanger}
              onClick={() => setAlgorithm(undefined)}
              style={{ padding: "0.25rem 0.5rem", fontSize: "var(--text-xs)" }}
            >
              Remove
            </button>
          )}
        </div>
        {algorithm && (
          <div
            style={{
              padding: "var(--spacing-md)",
              display: "flex",
              flexDirection: "column",
              gap: "var(--spacing-md)",
            }}
          >
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Algorithm Type</label>
              <CustomSelect
                value={algorithm.algoType}
                options={[...ALGORITHM_TYPES]}
                onChange={(v) => setAlgorithm({ algoType: v, fields: {} })}
              />
              {ALGORITHM_DESCRIPTIONS[algorithm.algoType] && (
                <span
                  style={{
                    fontSize: "0.625rem",
                    color: "var(--color-text-muted)",
                    marginTop: "0.25rem",
                  }}
                >
                  {ALGORITHM_DESCRIPTIONS[algorithm.algoType]}
                </span>
              )}
            </div>
            <AlgorithmSchemaEditor
              algoType={algorithm.algoType}
              fields={algorithm.fields}
              onUpdate={(f) => setAlgorithm({ ...algorithm, fields: f })}
            />
          </div>
        )}
      </div>

      {/* Plugins Toggle Panel */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>
            Plugins ({plugins.length})
          </span>
        </div>
        <div
          style={{
            padding: "var(--spacing-md)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--spacing-sm)",
          }}
        >
          {availablePlugins.length > 0 ? (
            <div className={styles.pluginToggleGrid}>
              {availablePlugins.map((p) => {
                const active = activePluginNames.has(p.name);
                return (
                  <button
                    key={p.name}
                    className={
                      active ? styles.pluginToggleActive : styles.pluginToggle
                    }
                    onClick={() => togglePlugin(p.name)}
                  >
                    <span className={styles.pluginToggleCheck}>
                      {active ? "✓" : "○"}
                    </span>
                    <span className={styles.pluginToggleName}>{p.name}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <span
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
              }}
            >
              No plugins defined yet.
            </span>
          )}

          {/* Active plugin configuration editors */}
          {plugins.length > 0 && (
            <div
              style={{
                marginTop: "var(--spacing-sm)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--spacing-sm)",
              }}
            >
              <span className={styles.fieldLabel} style={{ display: "block" }}>
                Plugin Configuration
              </span>
              {plugins.map((p) => {
                const tmpl = availablePlugins.find((ap) => ap.name === p.name);
                const pType = tmpl?.pluginType ?? p.name;
                return (
                  <div key={p.name} className={styles.pluginOverride}>
                    <PluginSchemaEditor
                      pluginType={pType}
                      pluginName={p.name}
                      fields={p.fields ?? {}}
                      onUpdate={(f) => updatePluginFields(p.name, f)}
                      compact
                    />
                  </div>
                );
              })}
            </div>
          )}

          {/* Manual plugin add (for inline plugins not in templates) */}
          <ManualPluginAdder
            existingNames={activePluginNames}
            onAdd={(name) => setPlugins((prev) => [...prev, { name }])}
          />
        </div>
      </div>

      {/* DSL Preview with validation */}
      <RouteDslPreviewPanel dslText={dslPreview} issues={validationIssues} />
    </div>
  );
};

export { AddRouteForm };
