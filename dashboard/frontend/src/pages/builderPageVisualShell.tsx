import React, { useCallback, useMemo, useState } from "react";

import type {
  Diagnostic,
  EditorMode,
  DSLFieldObject,
} from "@/types/dsl";
import { useDSLStore } from "@/stores/dslStore";
import type { RouteInput } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import {
  ModelIcon,
  PluginIcon,
  RouteIcon,
  SignalIcon,
} from "./builderPageFormPrimitives";
import {
  AddModelForm,
  AddPluginForm,
  AddSignalForm,
} from "./builderPageEntityForms";
import {
  AddProjectionMappingForm,
  AddProjectionPartitionForm,
  AddProjectionScoreForm,
} from "./builderPageProjectionEditors";
import {
  DashboardView,
  EntityListView,
  SidebarSection,
} from "./builderPageDashboardViews";
import { EntityDetailView } from "./builderPageEntityDetailView";
import { AddRouteForm } from "./builderPageRouteForms";
import { BuilderValidationPanel } from "./builderPageValidationPanel";
import type {
  BuilderSelectedEntity,
  EntityKind,
  SectionState,
  Selection,
} from "./builderPageTypes";

interface VisualModeProps {
  ast: ReturnType<typeof useDSLStore.getState>["ast"];
  dslSource: string;
  diagnostics: Diagnostic[];
  selection: Selection | null;
  onSelect: (sel: Selection | null) => void;
  sections: SectionState;
  onToggleSection: (key: keyof SectionState) => void;
  selectedEntity: BuilderSelectedEntity;
  modelCount: number;
  signalCount: number;
  projectionPartitionCount: number;
  projectionScoreCount: number;
  projectionMappingCount: number;
  routeCount: number;
  pluginCount: number;
  wasmReady: boolean;
  wasmError: string | null;
  addingEntity: EntityKind | null;
  onSetAddingEntity: (kind: EntityKind | null) => void;
  onDeleteEntity: (kind: EntityKind, name: string, subType?: string) => void;
  onUpdateModelFields: (name: string, fields: DSLFieldObject) => void;
  onUpdateSignalFields: (
    signalType: string,
    name: string,
    fields: DSLFieldObject,
  ) => void;
  onUpdateProjectionPartitionFields: (name: string, fields: DSLFieldObject) => void;
  onUpdateProjectionScoreFields: (
    name: string,
    fields: DSLFieldObject,
  ) => void;
  onUpdateProjectionMappingFields: (
    name: string,
    fields: DSLFieldObject,
  ) => void;
  onUpdatePluginFields: (
    name: string,
    pluginType: string,
    fields: DSLFieldObject,
  ) => void;
  onAddModel: (name: string, fields: DSLFieldObject) => void;
  onAddSignal: (
    signalType: string,
    name: string,
    fields: DSLFieldObject,
  ) => void;
  onAddProjectionPartition: (name: string, fields: DSLFieldObject) => void;
  onAddProjectionScore: (name: string, fields: DSLFieldObject) => void;
  onAddProjectionMapping: (name: string, fields: DSLFieldObject) => void;
  onAddPlugin: (
    name: string,
    pluginType: string,
    fields: DSLFieldObject,
  ) => void;
  onUpdateRoute: (name: string, input: RouteInput) => void;
  onAddRoute: (name: string, input: RouteInput) => void;
  errorCount: number;
  isValid: boolean;
  onModeSwitch: (mode: EditorMode) => void;
}

const VisualMode: React.FC<VisualModeProps> = ({
  ast,
  diagnostics,
  selection,
  onSelect,
  sections,
  onToggleSection,
  selectedEntity,
  modelCount,
  signalCount,
  projectionPartitionCount,
  projectionScoreCount,
  projectionMappingCount,
  routeCount,
  pluginCount,
  wasmReady,
  wasmError,
  addingEntity,
  onSetAddingEntity,
  onDeleteEntity,
  onUpdateModelFields,
  onUpdateSignalFields,
  onUpdateProjectionPartitionFields,
  onUpdateProjectionScoreFields,
  onUpdateProjectionMappingFields,
  onUpdatePluginFields,
  onAddModel,
  onAddSignal,
  onAddProjectionPartition,
  onAddProjectionScore,
  onAddProjectionMapping,
  onAddPlugin,
  onUpdateRoute,
  onAddRoute,
  errorCount,
  isValid,
  onModeSwitch,
}) => {
  // Collect available signal names for expression builder
  // Complexity signals are referenced as "<name>:easy", "<name>:medium", "<name>:hard" in route conditions
  const availableSignals = useMemo(() => {
    const result: { signalType: string; name: string }[] = [];
    for (const s of ast?.signals ?? []) {
      if (s.signalType === "complexity") {
        result.push({ signalType: s.signalType, name: `${s.name}:easy` });
        result.push({ signalType: s.signalType, name: `${s.name}:medium` });
        result.push({ signalType: s.signalType, name: `${s.name}:hard` });
      } else {
        result.push({ signalType: s.signalType, name: s.name });
      }
    }
    for (const mapping of ast?.projectionMappings ?? []) {
      for (const output of mapping.outputs ?? []) {
        result.push({ signalType: "projection", name: output.name });
      }
    }
    return result;
  }, [ast?.signals, ast?.projectionMappings]);
  // Collect available plugin names for toggle panel
  const availablePlugins = useMemo(
    () =>
      ast?.plugins?.map((p) => ({ name: p.name, pluginType: p.pluginType })) ??
      [],
    [ast?.plugins],
  );
  const semanticModels = useMemo(
    () => ast?.models?.map((model) => model.name) ?? [],
    [ast?.models],
  );
  // Collect available model names for route selection.
  const availableModels = useMemo(() => {
    if (semanticModels.length > 0) {
      return [...semanticModels].sort()
    }
    const modelSet = new Set<string>();
    ast?.routes?.forEach((r) =>
      r.models.forEach((m) => {
        if (m.model) modelSet.add(m.model);
      }),
    );
    return Array.from(modelSet).sort();
  }, [ast?.routes, semanticModels]);

  // Validation panel state
  const [validationOpen, setValidationOpen] = useState(true);
  const errorDiags = useMemo(
    () => diagnostics.filter((d) => d.level === "error"),
    [diagnostics],
  );
  const warnDiags = useMemo(
    () => diagnostics.filter((d) => d.level === "warning"),
    [diagnostics],
  );
  const constraintDiags = useMemo(
    () => diagnostics.filter((d) => d.level === "constraint"),
    [diagnostics],
  );

  const handleApplyFix = useCallback((diag: Diagnostic, newText: string) => {
    const store = useDSLStore.getState();
    const src = store.dslSource;
    const lines = src.split("\n");
    if (diag.line < 1 || diag.line > lines.length) return;

    const lineContent = lines[diag.line - 1];
    let startCol = diag.column;
    let endCol = diag.column;
    while (startCol > 1 && /[\w\-.]/.test(lineContent[startCol - 2]))
      startCol--;
    while (
      endCol <= lineContent.length &&
      /[\w\-.]/.test(lineContent[endCol - 1])
    )
      endCol++;

    const before = lineContent.slice(0, startCol - 1);
    const after = lineContent.slice(endCol - 1);
    lines[diag.line - 1] = before + newText + after;

    const newSrc = lines.join("\n");
    useDSLStore.getState().setDslSource(newSrc);
    // Re-parse AST for visual mode
    if (useDSLStore.getState().wasmReady) useDSLStore.getState().parseAST();
  }, []);

  return (
    <div className={styles.visualContainer}>
      <div className={styles.visualRow}>
        {/* Sidebar */}
        <div className={styles.sidebar}>
          {/* Dashboard home link */}
          <div
            className={
              selection === null && !addingEntity
                ? styles.sidebarHomeActive
                : styles.sidebarHome
            }
            onClick={() => {
              onSetAddingEntity(null);
              onSelect(null);
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
            </svg>
            Dashboard
          </div>

          <SidebarSection
            title="Models"
            count={modelCount}
            open={sections.models}
            onToggle={() => onToggleSection("models")}
            onAdd={() => {
              onSetAddingEntity("model");
              onSelect(null);
            }}
          >
            {ast?.models?.map((model) => (
              <li
                key={model.name}
                className={
                  selection?.kind === "model" && selection.name === model.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "model", name: model.name });
                }}
              >
                <ModelIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{model.name}</span>
                <span className={styles.sidebarItemType}>catalog</span>
              </li>
            ))}
          </SidebarSection>

          {/* Signals */}
          <SidebarSection
            title="Signals"
            count={signalCount}
            open={sections.signals}
            onToggle={() => onToggleSection("signals")}
            onAdd={() => {
              onSetAddingEntity("signal");
              onSelect(null);
            }}
          >
            {ast?.signals?.map((s) => (
              <li
                key={s.name}
                className={
                  selection?.kind === "signal" && selection.name === s.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "signal", name: s.name });
                }}
              >
                <SignalIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{s.name}</span>
                <span className={styles.sidebarItemType}>{s.signalType}</span>
              </li>
            ))}
          </SidebarSection>

          <SidebarSection
            title="Projection Partitions"
            count={projectionPartitionCount}
            open={sections.projectionPartitions}
            onToggle={() => onToggleSection("projectionPartitions")}
            onAdd={() => {
              onSetAddingEntity("projection-partition");
              onSelect(null);
            }}
          >
            {ast?.projectionPartitions?.map((partition) => (
              <li
                key={partition.name}
                className={
                  selection?.kind === "projection-partition" &&
                  selection.name === partition.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "projection-partition", name: partition.name });
                }}
              >
                <SignalIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{partition.name}</span>
                <span className={styles.sidebarItemType}>partition</span>
              </li>
            ))}
          </SidebarSection>

          <SidebarSection
            title="Projection Scores"
            count={projectionScoreCount}
            open={sections.projectionScores}
            onToggle={() => onToggleSection("projectionScores")}
            onAdd={() => {
              onSetAddingEntity("projection-score");
              onSelect(null);
            }}
          >
            {ast?.projectionScores?.map((score) => (
              <li
                key={score.name}
                className={
                  selection?.kind === "projection-score" &&
                  selection.name === score.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "projection-score", name: score.name });
                }}
              >
                <RouteIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{score.name}</span>
                <span className={styles.sidebarItemType}>score</span>
              </li>
            ))}
          </SidebarSection>

          <SidebarSection
            title="Projection Mappings"
            count={projectionMappingCount}
            open={sections.projectionMappings}
            onToggle={() => onToggleSection("projectionMappings")}
            onAdd={() => {
              onSetAddingEntity("projection-mapping");
              onSelect(null);
            }}
          >
            {ast?.projectionMappings?.map((mapping) => (
              <li
                key={mapping.name}
                className={
                  selection?.kind === "projection-mapping" &&
                  selection.name === mapping.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "projection-mapping", name: mapping.name });
                }}
              >
                <RouteIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{mapping.name}</span>
                <span className={styles.sidebarItemType}>mapping</span>
              </li>
            ))}
          </SidebarSection>

          {/* Routes */}
          <SidebarSection
            title="Routes"
            count={routeCount}
            open={sections.routes}
            onToggle={() => onToggleSection("routes")}
            onAdd={() => {
              onSetAddingEntity("route");
              onSelect(null);
            }}
          >
            {ast?.routes?.map((r) => (
              <li
                key={r.name}
                className={
                  selection?.kind === "route" && selection.name === r.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "route", name: r.name });
                }}
              >
                <RouteIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{r.name}</span>
                <span className={styles.sidebarItemType}>P{r.priority}</span>
              </li>
            ))}
          </SidebarSection>

          {/* Plugins */}
          <SidebarSection
            title="Plugins"
            count={pluginCount}
            open={sections.plugins}
            onToggle={() => onToggleSection("plugins")}
            onAdd={() => {
              onSetAddingEntity("plugin");
              onSelect(null);
            }}
          >
            {ast?.plugins?.map((p) => (
              <li
                key={p.name}
                className={
                  selection?.kind === "plugin" && selection.name === p.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "plugin", name: p.name });
                }}
              >
                <PluginIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{p.name}</span>
                <span className={styles.sidebarItemType}>{p.pluginType}</span>
              </li>
            ))}
          </SidebarSection>

        </div>

        {/* Main panel */}
        <div className={styles.mainPanel}>
          {!wasmReady && !wasmError && (
            <div className={styles.wasmOverlay}>
              <div className={styles.spinner} />
              Loading Signal Compiler…
            </div>
          )}

          <div className={styles.mainPanelContent}>
            {addingEntity === "model" ? (
              <AddModelForm
                onAdd={onAddModel}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "signal" ? (
              <AddSignalForm
                onAdd={onAddSignal}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "projection-partition" ? (
              <AddProjectionPartitionForm
                onAdd={onAddProjectionPartition}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "projection-score" ? (
              <AddProjectionScoreForm
                onAdd={onAddProjectionScore}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "projection-mapping" ? (
              <AddProjectionMappingForm
                onAdd={onAddProjectionMapping}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "plugin" ? (
              <AddPluginForm
                onAdd={onAddPlugin}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "route" ? (
              <AddRouteForm
                onAdd={onAddRoute}
                onCancel={() => onSetAddingEntity(null)}
                availableSignals={availableSignals}
                availablePlugins={availablePlugins}
                availableModels={availableModels}
              />
            ) : !selection ? (
              <DashboardView
                ast={ast}
                modelCount={modelCount}
                signalCount={signalCount}
                routeCount={routeCount}
                pluginCount={pluginCount}
                isValid={isValid}
                errorCount={errorCount}
                onSelect={onSelect}
                onAddEntity={onSetAddingEntity}
                onModeSwitch={onModeSwitch}
              />
            ) : selection.name === "__list__" ? (
              <EntityListView
                kind={selection.kind}
                ast={ast}
                onSelect={onSelect}
                onBack={() => onSelect(null)}
                onAddEntity={onSetAddingEntity}
              />
            ) : (
              <EntityDetailView
                selection={selection}
                entity={selectedEntity}
                onDeleteEntity={onDeleteEntity}
                onUpdateModelFields={onUpdateModelFields}
                onUpdateSignalFields={onUpdateSignalFields}
                onUpdateProjectionPartitionFields={onUpdateProjectionPartitionFields}
                onUpdateProjectionScoreFields={onUpdateProjectionScoreFields}
                onUpdateProjectionMappingFields={onUpdateProjectionMappingFields}
                onUpdatePluginFields={onUpdatePluginFields}
                onUpdateRoute={onUpdateRoute}
                availableSignals={availableSignals}
                availablePlugins={availablePlugins}
                availableModels={availableModels}
                onBack={() => onSelect(null)}
              />
            )}
          </div>
        </div>
      </div>
      {/* end visualRow */}

      <BuilderValidationPanel
        diagnostics={diagnostics}
        validationOpen={validationOpen}
        errorDiags={errorDiags}
        warnDiags={warnDiags}
        constraintDiags={constraintDiags}
        onToggle={() => setValidationOpen(!validationOpen)}
        onApplyFix={handleApplyFix}
      />
    </div>
  );
};

export { VisualMode };
