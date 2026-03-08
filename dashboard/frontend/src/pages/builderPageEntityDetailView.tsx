import React from "react";

import type {
  ASTBackendDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
} from "@/types/dsl";
import type { RouteInput } from "@/lib/dslMutations";
import { useDSLStore } from "@/stores/dslStore";

import styles from "./BuilderPage.module.css";
import {
  BackendIcon,
  GenericFieldsEditor,
  GlobalIcon,
  PluginIcon,
  RouteIcon,
  SignalIcon,
} from "./builderPageFormPrimitives";
import {
  GlobalSettingsEditor,
  PluginSchemaEditor,
  SignalEditorForm,
} from "./builderPageEntityForms";
import { RouteEditorForm } from "./builderPageRouteForms";
import type {
  AvailablePlugin,
  AvailableSignal,
  BuilderSelectedEntity,
  EntityKind,
  Selection,
} from "./builderPageTypes";

interface EntityDetailViewProps {
  selection: Selection;
  entity: BuilderSelectedEntity;
  ast: ReturnType<typeof useDSLStore.getState>["ast"];
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
  onUpdateRoute: (name: string, input: RouteInput) => void;
  onUpdateGlobalFields: (fields: Record<string, unknown>) => void;
  availableSignals: AvailableSignal[];
  availablePlugins: AvailablePlugin[];
  availableModels: string[];
  onBack: () => void;
}

const EntityDetailView: React.FC<EntityDetailViewProps> = ({
  selection,
  entity,
  ast,
  onDeleteEntity,
  onUpdateSignalFields,
  onUpdatePluginFields,
  onUpdateBackendFields,
  onUpdateRoute,
  onUpdateGlobalFields,
  availableSignals,
  availablePlugins,
  availableModels,
  onBack,
}) => {
  if (!entity) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>🔍</div>
        <div>Entity &quot;{selection.name}&quot; not found in current AST</div>
        <div
          style={{
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
          }}
        >
          Try compiling or validating your DSL first
        </div>
      </div>
    );
  }

  const subType =
    "signalType" in entity
      ? entity.signalType
      : "pluginType" in entity
        ? entity.pluginType
        : "backendType" in entity
          ? entity.backendType
          : undefined;

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <button
          className={styles.backBtn}
          onClick={onBack}
          title="Back to Dashboard"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <div className={styles.editorTitle}>
          {selection.kind === "signal" && (
            <SignalIcon className={styles.statIcon} />
          )}
          {selection.kind === "route" && (
            <RouteIcon className={styles.statIcon} />
          )}
          {selection.kind === "plugin" && (
            <PluginIcon className={styles.statIcon} />
          )}
          {selection.kind === "backend" && (
            <BackendIcon className={styles.statIcon} />
          )}
          {selection.kind === "global" && (
            <GlobalIcon className={styles.statIcon} />
          )}
          {"name" in entity ? entity.name : "Global Settings"}
          {"signalType" in entity && (
            <span className={styles.editorBadge}>{entity.signalType}</span>
          )}
          {"pluginType" in entity && (
            <span className={styles.editorBadge}>{entity.pluginType}</span>
          )}
          {"backendType" in entity && (
            <span className={styles.editorBadge}>{entity.backendType}</span>
          )}
        </div>
        {selection.kind !== "global" && (
          <div className={styles.editorActions}>
            <button
              className={styles.toolbarBtnDanger}
              onClick={() =>
                onDeleteEntity(selection.kind, selection.name, subType)
              }
              title="Delete this entity"
            >
              Delete
            </button>
          </div>
        )}
      </div>

      {/* Editable Signal form */}
      {selection.kind === "signal" && "signalType" in entity && (
        <SignalEditorForm
          signal={entity as ASTSignalDecl}
          onUpdate={(fields) =>
            onUpdateSignalFields(
              (entity as ASTSignalDecl).signalType,
              (entity as ASTSignalDecl).name,
              fields,
            )
          }
        />
      )}

      {/* Editable Plugin form */}
      {selection.kind === "plugin" && "pluginType" in entity && (
        <PluginSchemaEditor
          pluginType={(entity as ASTPluginDecl).pluginType}
          fields={
            "fields" in entity ? (entity.fields as Record<string, unknown>) : {}
          }
          onUpdate={(fields) =>
            onUpdatePluginFields(
              (entity as ASTPluginDecl).name,
              (entity as ASTPluginDecl).pluginType,
              fields,
            )
          }
          buffered
        />
      )}

      {/* Editable Backend form */}
      {selection.kind === "backend" && "backendType" in entity && (
        <GenericFieldsEditor
          fields={
            "fields" in entity ? (entity.fields as Record<string, unknown>) : {}
          }
          onUpdate={(fields) =>
            onUpdateBackendFields(
              (entity as ASTBackendDecl).backendType,
              (entity as ASTBackendDecl).name,
              fields,
            )
          }
        />
      )}

      {/* Editable Route form */}
      {selection.kind === "route" && "priority" in entity && (
        <RouteEditorForm
          route={entity as ASTRouteDecl}
          onUpdate={(input) =>
            onUpdateRoute((entity as ASTRouteDecl).name, input)
          }
          availableSignals={availableSignals}
          availablePlugins={availablePlugins}
          availableModels={availableModels}
        />
      )}

      {/* Editable Global form — structured sections */}
      {selection.kind === "global" && (
        <GlobalSettingsEditor
          fields={
            "fields" in entity ? (entity.fields as Record<string, unknown>) : {}
          }
          onUpdate={onUpdateGlobalFields}
          endpoints={
            ast?.backends?.filter(
              (b) =>
                b.backendType === "vllm_endpoint" ||
                b.backendType === "provider_profile",
            ) ?? []
          }
          onSelectEndpoint={() => {
            onBack();
            setTimeout(() => onBack(), 0);
          }}
        />
      )}
    </div>
  );
};

export { EntityDetailView };
