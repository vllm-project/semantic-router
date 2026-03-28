import React from "react";

import type {
  ASTModelDecl,
  ASTProjectionPartitionDecl,
  ASTProjectionMappingDecl,
  ASTProjectionScoreDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
  DSLFieldObject,
} from "@/types/dsl";
import type { RouteInput } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import {
  GenericFieldsEditor,
  ModelIcon,
  PluginIcon,
  RouteIcon,
  SignalIcon,
} from "./builderPageFormPrimitives";
import {
  PluginSchemaEditor,
  SignalEditorForm,
} from "./builderPageEntityForms";
import {
  ProjectionMappingEditorForm,
  ProjectionPartitionEditorForm,
  ProjectionScoreEditorForm,
} from "./builderPageProjectionEditors";
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
  onDeleteEntity: (kind: EntityKind, name: string, subType?: string) => void;
  onUpdateSignalFields: (
    signalType: string,
    name: string,
    fields: DSLFieldObject,
  ) => void;
  onUpdateProjectionPartitionFields: (
    name: string,
    fields: DSLFieldObject,
  ) => void;
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
  onUpdateModelFields: (name: string, fields: DSLFieldObject) => void;
  onUpdateRoute: (name: string, input: RouteInput) => void;
  availableSignals: AvailableSignal[];
  availablePlugins: AvailablePlugin[];
  availableModels: string[];
  onBack: () => void;
}

const EntityDetailView: React.FC<EntityDetailViewProps> = ({
  selection,
  entity,
  onDeleteEntity,
  onUpdateModelFields,
  onUpdateSignalFields,
  onUpdateProjectionPartitionFields,
  onUpdateProjectionScoreFields,
  onUpdateProjectionMappingFields,
  onUpdatePluginFields,
  onUpdateRoute,
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
          {selection.kind === "model" && (
            <ModelIcon className={styles.statIcon} />
          )}
          {selection.kind === "signal" && (
            <SignalIcon className={styles.statIcon} />
          )}
          {selection.kind === "route" && (
            <RouteIcon className={styles.statIcon} />
          )}
          {selection.kind === "plugin" && (
            <PluginIcon className={styles.statIcon} />
          )}
          {"name" in entity ? entity.name : "Entity"}
          {"signalType" in entity && (
            <span className={styles.editorBadge}>{entity.signalType}</span>
          )}
          {"semantics" in entity && (
            <span className={styles.editorBadge}>projection partition</span>
          )}
          {"source" in entity && !("signalType" in entity) && !("pluginType" in entity) && (
            <span className={styles.editorBadge}>projection mapping</span>
          )}
          {"inputs" in entity && !("signalType" in entity) && (
            <span className={styles.editorBadge}>projection score</span>
          )}
          {"pluginType" in entity && (
            <span className={styles.editorBadge}>{entity.pluginType}</span>
          )}
        </div>
        <div className={styles.editorActions}>
          <button
            className={styles.toolbarBtnDanger}
            onClick={() => onDeleteEntity(selection.kind, selection.name, subType)}
            title="Delete this entity"
          >
            Delete
          </button>
        </div>
      </div>

      {/* Editable Model form */}
      {selection.kind === "model" && "fields" in entity && (
        <GenericFieldsEditor
          fields={(entity as ASTModelDecl).fields}
          onUpdate={(fields) =>
            onUpdateModelFields((entity as ASTModelDecl).name, fields)
          }
        />
      )}

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

      {selection.kind === "projection-partition" && "members" in entity && (
        <ProjectionPartitionEditorForm
          partition={entity as ASTProjectionPartitionDecl}
          onUpdate={(fields) =>
            onUpdateProjectionPartitionFields(
              (entity as ASTProjectionPartitionDecl).name,
              fields,
            )
          }
        />
      )}

      {selection.kind === "projection-score" && (
        <ProjectionScoreEditorForm
          score={entity as ASTProjectionScoreDecl}
          onUpdate={(fields) =>
            onUpdateProjectionScoreFields(
              (entity as ASTProjectionScoreDecl).name,
              fields,
            )
          }
        />
      )}

      {selection.kind === "projection-mapping" && (
        <ProjectionMappingEditorForm
          mapping={entity as ASTProjectionMappingDecl}
          onUpdate={(fields) =>
            onUpdateProjectionMappingFields(
              (entity as ASTProjectionMappingDecl).name,
              fields,
            )
          }
        />
      )}

      {/* Editable Plugin form */}
      {selection.kind === "plugin" && "pluginType" in entity && (
        <PluginSchemaEditor
          pluginType={(entity as ASTPluginDecl).pluginType}
          fields={"fields" in entity ? entity.fields : {}}
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
    </div>
  );
};

export { EntityDetailView };
