import React, { useCallback, useEffect, useMemo, useState } from "react";

import ExpressionBuilder from "@/components/ExpressionBuilder";
import { useDSLStore } from "@/stores/dslStore";
import type { ASTRouteDecl } from "@/types/dsl";
import {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  serializeBoolExpr,
} from "@/lib/dslMutations";
import type {
  RouteAlgoInput,
  RouteInput,
  RouteModelInput,
  RoutePluginInput,
} from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import { CustomSelect } from "./builderPageFormPrimitives";
import { AlgorithmSchemaEditor, PluginSchemaEditor } from "./builderPageEntityForms";
import {
  RouteDslPreviewPanel,
  astAlgoToInput,
  astModelToInput,
  astPluginRefToInput,
  generateRouteDslPreview,
  validateRouteInput,
} from "./builderPageRoutePreview";
import { ModelNameInput, ManualPluginAdder } from "./builderPageRouteSharedControls";
import type { AvailablePlugin, AvailableSignal } from "./builderPageTypes";

const RouteEditorForm: React.FC<{
  route: ASTRouteDecl;
  onUpdate: (input: RouteInput) => void;
  availableSignals: AvailableSignal[];
  availablePlugins: AvailablePlugin[];
  availableModels: string[];
}> = ({
  route,
  onUpdate,
  availableSignals,
  availablePlugins,
  availableModels,
}) => {
  const [description, setDescription] = useState(route.description ?? "");
  const [priority, setPriority] = useState(route.priority);
  const [whenExpr, setWhenExpr] = useState(() =>
    route.when
      ? serializeBoolExpr(route.when as unknown as Record<string, unknown>)
      : "",
  );
  const [models, setModels] = useState<RouteModelInput[]>(() =>
    route.models.map(astModelToInput),
  );
  const [algorithm, setAlgorithm] = useState<RouteAlgoInput | undefined>(() =>
    astAlgoToInput(route.algorithm),
  );
  const [plugins, setPlugins] = useState<RoutePluginInput[]>(() =>
    route.plugins.map(astPluginRefToInput),
  );

  // Sync from parent when route changes
  useEffect(() => {
    setDescription(route.description ?? "");
    setPriority(route.priority);
    setWhenExpr(
      route.when
        ? serializeBoolExpr(route.when as unknown as Record<string, unknown>)
        : "",
    );
    setModels(route.models.map(astModelToInput));
    setAlgorithm(astAlgoToInput(route.algorithm));
    setPlugins(route.plugins.map(astPluginRefToInput));
  }, [
    route.name,
    route.priority,
    route.description,
    route.when,
    route.models,
    route.algorithm,
    route.plugins,
  ]);

  const handleSave = useCallback(() => {
    onUpdate({
      description: description.trim() || undefined,
      priority,
      when: whenExpr.trim() || undefined,
      models,
      algorithm: algorithm?.algoType ? algorithm : undefined,
      plugins,
    });
  }, [description, priority, whenExpr, models, algorithm, plugins, onUpdate]);

  // Model helpers
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

  // Plugin toggle helpers
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

  // Expression builder: tree-based, managed by ExpressionBuilder component

  // Generate DSL preview & validation
  const dslPreview = useMemo(
    () =>
      generateRouteDslPreview(
        route.name,
        description,
        priority,
        whenExpr,
        models,
        algorithm,
        plugins,
      ),
    [route.name, description, priority, whenExpr, models, algorithm, plugins],
  );

  const validationIssues = useMemo(
    () => validateRouteInput(route.name, models, algorithm, plugins),
    [route.name, models, algorithm, plugins],
  );

  // Get WASM diagnostics scoped to this route
  const diagnostics = useDSLStore((s) => s.diagnostics);
  const routeDiagnostics = useMemo(() => {
    if (!route.pos?.Line) return [];
    const startLine = route.pos.Line;
    return diagnostics
      .filter((d) => d.line >= startLine && d.line <= startLine + 50)
      .map((d) => ({ level: d.level, message: d.message }));
  }, [diagnostics, route.pos]);

  return (
    <>
      {/* Header with Save */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Route Configuration</span>
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
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: "var(--spacing-md)",
            }}
          >
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Description</label>
              <input
                className={styles.fieldInput}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Route description..."
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
            initialAstExpr={
              route.when as unknown as Record<string, unknown> | null
            }
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
            + Add Model
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
          {models.length === 0 && (
            <span
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
              }}
            >
              No models configured. Add at least one model.
            </span>
          )}
          {models.map((m, idx) => (
            <div key={idx} className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelIndex}>{idx + 1}</span>
                <ModelNameInput
                  value={m.model}
                  availableModels={availableModels}
                  onChange={(v) => updateModel(idx, { model: v })}
                />
                <button
                  className={styles.toolbarBtnDanger}
                  onClick={() => removeModel(idx)}
                  style={{
                    padding: "0.25rem 0.5rem",
                    fontSize: "var(--text-xs)",
                    flexShrink: 0,
                  }}
                  title="Remove model"
                >
                  ×
                </button>
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
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>weight:</span>
                  <input
                    className={styles.fieldInput}
                    style={{
                      width: "60px",
                      fontSize: "var(--text-xs)",
                      padding: "0.25rem 0.5rem",
                    }}
                    type="number"
                    step="any"
                    value={m.weight !== undefined ? m.weight : ""}
                    onChange={(e) =>
                      updateModel(idx, {
                        weight: e.target.value
                          ? Number(e.target.value)
                          : undefined,
                      })
                    }
                    placeholder="—"
                  />
                </div>
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>param_size:</span>
                  <input
                    className={styles.fieldInput}
                    style={{
                      width: "70px",
                      fontSize: "var(--text-xs)",
                      padding: "0.25rem 0.5rem",
                    }}
                    value={m.paramSize ?? ""}
                    onChange={(e) =>
                      updateModel(idx, {
                        paramSize: e.target.value || undefined,
                      })
                    }
                    placeholder="—"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>
            Algorithm {models.length >= 2 ? "" : "(optional — for multi-model)"}
          </span>
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
        {!algorithm && (
          <div
            style={{
              padding: "var(--spacing-md)",
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
            }}
          >
            No algorithm configured.{" "}
            {models.length >= 2
              ? "Recommended when using multiple models."
              : ""}
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
          {/* Toggle chips for available plugins */}
          {availablePlugins.length > 0 && (
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
                    title={`${active ? "Remove" : "Add"} plugin ${p.name}`}
                  >
                    <span className={styles.pluginToggleCheck}>
                      {active ? "✓" : "○"}
                    </span>
                    <span className={styles.pluginToggleName}>{p.name}</span>
                    <span className={styles.pluginToggleType}>
                      {p.pluginType}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
          {availablePlugins.length === 0 && (
            <span
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
              }}
            >
              No plugins defined. Create plugins first.
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
                // Resolve pluginType: from top-level template, or treat name as type for inline plugins
                const tmpl = availablePlugins.find((ap) => ap.name === p.name);
                const pluginType = tmpl?.pluginType ?? p.name;
                return (
                  <div key={p.name} className={styles.pluginOverride}>
                    <PluginSchemaEditor
                      pluginType={pluginType}
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
            onAdd={(name, fields) =>
              setPlugins((prev) => [...prev, { name, fields }])
            }
          />
        </div>
      </div>

      {/* DSL Preview with validation */}
      <RouteDslPreviewPanel
        dslText={dslPreview}
        issues={validationIssues}
        wasmDiagnostics={routeDiagnostics}
      />
    </>
  );
};

export { RouteEditorForm };
