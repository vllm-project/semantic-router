import { useCallback, useMemo, useState } from "react";

import type {
  ASTProjectionMappingDecl,
  ASTProjectionScoreDecl,
  ASTSignalGroupDecl,
  DSLFieldObject,
} from "@/types/dsl";
import { serializeFields } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import { GenericFieldsEditor } from "./builderPageFormPrimitives";
import { DslPreviewPanel } from "./builderPageSharedDslEditors";

function buildDslPreview(header: string, fields: DSLFieldObject): string {
  const body = serializeFields(fields);
  if (!body.trim()) {
    return `${header} {}`;
  }
  return `${header} {\n${body}\n}`;
}

export function signalGroupToFields(group: ASTSignalGroupDecl): DSLFieldObject {
  return {
    ...(group.semantics ? { semantics: group.semantics } : {}),
    ...(typeof group.temperature === "number"
      ? { temperature: group.temperature }
      : {}),
    members: group.members ?? [],
    ...(group.default ? { default: group.default } : {}),
  };
}

export function projectionScoreToFields(
  score: ASTProjectionScoreDecl,
): DSLFieldObject {
  return {
    ...(score.method ? { method: score.method } : {}),
    inputs: (score.inputs ?? []).map((input) => ({
      type: input.signalType,
      name: input.signalName,
      weight: input.weight,
      ...(input.valueSource ? { value_source: input.valueSource } : {}),
      ...(typeof input.match === "number" ? { match: input.match } : {}),
      ...(typeof input.miss === "number" ? { miss: input.miss } : {}),
    })),
  };
}

export function projectionMappingToFields(
  mapping: ASTProjectionMappingDecl,
): DSLFieldObject {
  return {
    ...(mapping.source ? { source: mapping.source } : {}),
    ...(mapping.method ? { method: mapping.method } : {}),
    ...(mapping.calibration
      ? {
          calibration: {
            ...(mapping.calibration.method
              ? { method: mapping.calibration.method }
              : {}),
            ...(typeof mapping.calibration.slope === "number"
              ? { slope: mapping.calibration.slope }
              : {}),
          },
        }
      : {}),
    outputs: (mapping.outputs ?? []).map((output) => ({
      name: output.name,
      ...(typeof output.lt === "number" ? { lt: output.lt } : {}),
      ...(typeof output.lte === "number" ? { lte: output.lte } : {}),
      ...(typeof output.gt === "number" ? { gt: output.gt } : {}),
      ...(typeof output.gte === "number" ? { gte: output.gte } : {}),
    })),
  };
}

const NamedBlockEditor: React.FC<{
  title: string;
  previewTitle: string;
  previewHeader: string;
  fields: DSLFieldObject;
  onUpdate: (fields: DSLFieldObject) => void;
}> = ({ title, previewTitle, previewHeader, fields, onUpdate }) => {
  const dslPreview = useMemo(
    () => buildDslPreview(previewHeader, fields),
    [previewHeader, fields],
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--spacing-md)",
      }}
    >
      <GenericFieldsEditor fields={fields} onUpdate={onUpdate} />
      <DslPreviewPanel title={previewTitle || title} dslText={dslPreview} />
    </div>
  );
};

const NamedBlockAddForm: React.FC<{
  title: string;
  namePlaceholder: string;
  initialFields: DSLFieldObject;
  previewTitle: string;
  buildHeader: (name: string) => string;
  onAdd: (name: string, fields: DSLFieldObject) => void;
  onCancel: () => void;
}> = ({
  title,
  namePlaceholder,
  initialFields,
  previewTitle,
  buildHeader,
  onAdd,
  onCancel,
}) => {
  const [name, setName] = useState("");
  const [fields, setFields] = useState<DSLFieldObject>(initialFields);

  const handleSubmit = useCallback(() => {
    const trimmed = name.trim();
    if (!trimmed) return;
    onAdd(trimmed, fields);
  }, [name, fields, onAdd]);

  const previewHeader = buildHeader(name.trim() || "name_here");

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>{title}</div>
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
        <label className={styles.fieldLabel}>Name</label>
        <input
          className={styles.fieldInput}
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder={namePlaceholder}
          autoFocus
        />
      </div>

      <NamedBlockEditor
        title={title}
        previewTitle={previewTitle}
        previewHeader={previewHeader}
        fields={fields}
        onUpdate={setFields}
      />
    </div>
  );
};

export const SignalGroupEditorForm: React.FC<{
  group: ASTSignalGroupDecl;
  onUpdate: (fields: DSLFieldObject) => void;
}> = ({ group, onUpdate }) => (
  <NamedBlockEditor
    title="Signal Group"
    previewTitle="Signal Group Preview"
    previewHeader={`SIGNAL_GROUP ${group.name}`}
    fields={signalGroupToFields(group)}
    onUpdate={onUpdate}
  />
);

export const ProjectionScoreEditorForm: React.FC<{
  score: ASTProjectionScoreDecl;
  onUpdate: (fields: DSLFieldObject) => void;
}> = ({ score, onUpdate }) => (
  <NamedBlockEditor
    title="Projection Score"
    previewTitle="Projection Score Preview"
    previewHeader={`PROJECTION score ${score.name}`}
    fields={projectionScoreToFields(score)}
    onUpdate={onUpdate}
  />
);

export const ProjectionMappingEditorForm: React.FC<{
  mapping: ASTProjectionMappingDecl;
  onUpdate: (fields: DSLFieldObject) => void;
}> = ({ mapping, onUpdate }) => (
  <NamedBlockEditor
    title="Projection Mapping"
    previewTitle="Projection Mapping Preview"
    previewHeader={`PROJECTION mapping ${mapping.name}`}
    fields={projectionMappingToFields(mapping)}
    onUpdate={onUpdate}
  />
);

export const AddSignalGroupForm: React.FC<{
  onAdd: (name: string, fields: DSLFieldObject) => void;
  onCancel: () => void;
}> = ({ onAdd, onCancel }) => (
  <NamedBlockAddForm
    title="New Signal Group"
    namePlaceholder="balance_domain_partition"
    initialFields={{ semantics: "exclusive", members: [] }}
    previewTitle="Signal Group Preview"
    buildHeader={(name) => `SIGNAL_GROUP ${name}`}
    onAdd={onAdd}
    onCancel={onCancel}
  />
);

export const AddProjectionScoreForm: React.FC<{
  onAdd: (name: string, fields: DSLFieldObject) => void;
  onCancel: () => void;
}> = ({ onAdd, onCancel }) => (
  <NamedBlockAddForm
    title="New Projection Score"
    namePlaceholder="request_difficulty"
    initialFields={{ method: "weighted_sum", inputs: [] }}
    previewTitle="Projection Score Preview"
    buildHeader={(name) => `PROJECTION score ${name}`}
    onAdd={onAdd}
    onCancel={onCancel}
  />
);

export const AddProjectionMappingForm: React.FC<{
  onAdd: (name: string, fields: DSLFieldObject) => void;
  onCancel: () => void;
}> = ({ onAdd, onCancel }) => (
  <NamedBlockAddForm
    title="New Projection Mapping"
    namePlaceholder="request_band"
    initialFields={{ method: "threshold_bands", outputs: [] }}
    previewTitle="Projection Mapping Preview"
    buildHeader={(name) => `PROJECTION mapping ${name}`}
    onAdd={onAdd}
    onCancel={onCancel}
  />
);
