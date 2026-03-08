import React, { useCallback, useMemo, useState } from "react";

import type {
  Diagnostic,
  EditorMode,
} from "@/types/dsl";
import { useDSLStore } from "@/stores/dslStore";
import type { RouteInput } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import {
  BackendIcon,
  GlobalIcon,
  PluginIcon,
  RouteIcon,
  SignalIcon,
} from "./builderPageFormPrimitives";
import {
  AddBackendForm,
  AddPluginForm,
  AddSignalForm,
} from "./builderPageEntityForms";
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
  signalCount: number;
  routeCount: number;
  pluginCount: number;
  backendCount: number;
  hasGlobal: boolean;
  wasmReady: boolean;
  wasmError: string | null;
  addingEntity: EntityKind | null;
  onSetAddingEntity: (kind: EntityKind | null) => void;
  onDeleteEntity: (kind: EntityKind, name: string, subType?: string) => void;
  onUpdateSignalFields: (
    signalType: string,
    name: string,
    fields: Record<string, unknown>,
  ) => void;
  onUpdatePluginFields: (
    name: string,
    pluginType: string,
    fields: Record<string, unknown>,
  ) => void;
  onUpdateBackendFields: (
    backendType: string,
    name: string,
    fields: Record<string, unknown>,
  ) => void;
  onAddSignal: (
    signalType: string,
    name: string,
    fields: Record<string, unknown>,
  ) => void;
  onAddPlugin: (
    name: string,
    pluginType: string,
    fields: Record<string, unknown>,
  ) => void;
  onAddBackend: (
    backendType: string,
    name: string,
    fields: Record<string, unknown>,
  ) => void;
  onUpdateRoute: (name: string, input: RouteInput) => void;
  onUpdateGlobalFields: (fields: Record<string, unknown>) => void;
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
  signalCount,
  routeCount,
  pluginCount,
  backendCount,
  hasGlobal,
  wasmReady,
  wasmError,
  addingEntity,
  onSetAddingEntity,
  onDeleteEntity,
  onUpdateSignalFields,
  onUpdatePluginFields,
  onUpdateBackendFields,
  onAddSignal,
  onAddPlugin,
  onAddBackend,
  onUpdateRoute,
  onUpdateGlobalFields,
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
    return result;
  }, [ast?.signals]);
  // Collect available plugin names for toggle panel
  const availablePlugins = useMemo(
    () =>
      ast?.plugins?.map((p) => ({ name: p.name, pluginType: p.pluginType })) ??
      [],
    [ast?.plugins],
  );
  // Collect available model names from all routes for selection
  const availableModels = useMemo(() => {
    const modelSet = new Set<string>();
    ast?.routes?.forEach((r) =>
      r.models.forEach((m) => {
        if (m.model) modelSet.add(m.model);
      }),
    );
    return Array.from(modelSet).sort();
  }, [ast?.routes]);

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

          {/* Backends */}
          <SidebarSection
            title="Backends"
            count={backendCount}
            open={sections.backends}
            onToggle={() => onToggleSection("backends")}
            onAdd={() => {
              onSetAddingEntity("backend");
              onSelect(null);
            }}
          >
            {ast?.backends?.map((b) => (
              <li
                key={b.name}
                className={
                  selection?.kind === "backend" && selection.name === b.name
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "backend", name: b.name });
                }}
              >
                <BackendIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>{b.name}</span>
                <span className={styles.sidebarItemType}>{b.backendType}</span>
              </li>
            ))}
          </SidebarSection>

          {/* Global */}
          <SidebarSection
            title="Global"
            count={hasGlobal ? 1 : 0}
            open={sections.global}
            onToggle={() => onToggleSection("global")}
          >
            {hasGlobal && (
              <li
                className={
                  selection?.kind === "global"
                    ? styles.sidebarItemActive
                    : styles.sidebarItem
                }
                onClick={() => {
                  onSetAddingEntity(null);
                  onSelect({ kind: "global", name: "global" });
                }}
              >
                <GlobalIcon className={styles.sidebarItemIcon} />
                <span className={styles.sidebarItemName}>Global Settings</span>
              </li>
            )}
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
            {addingEntity === "signal" ? (
              <AddSignalForm
                onAdd={onAddSignal}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "plugin" ? (
              <AddPluginForm
                onAdd={onAddPlugin}
                onCancel={() => onSetAddingEntity(null)}
              />
            ) : addingEntity === "backend" ? (
              <AddBackendForm
                onAdd={onAddBackend}
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
                signalCount={signalCount}
                routeCount={routeCount}
                pluginCount={pluginCount}
                backendCount={backendCount}
                hasGlobal={hasGlobal}
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
                ast={ast}
                onDeleteEntity={onDeleteEntity}
                onUpdateSignalFields={onUpdateSignalFields}
                onUpdatePluginFields={onUpdatePluginFields}
                onUpdateBackendFields={onUpdateBackendFields}
                onUpdateRoute={onUpdateRoute}
                onUpdateGlobalFields={onUpdateGlobalFields}
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
