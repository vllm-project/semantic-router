import React, { useCallback, useEffect, useMemo, useState } from "react";

import type { ASTBackendDecl } from "@/types/dsl";

import styles from "./BuilderPage.module.css";
import {
  GlobalSettingsEndpointsSection,
  GlobalSettingsObservabilitySection,
  GlobalSettingsSafetySection,
} from "./builderPageGlobalSettingsAdditionalSections";
import {
  DEFAULT_LISTENER_PORT,
  getListeners,
  getObj,
  type EditableListener,
} from "./builderPageGlobalSettingsSupport";
import { GlobalSettingsRoutingSection } from "./builderPageGlobalSettingsRoutingSection";
import {
  DslPreviewPanel,
  generateGlobalDslPreview,
} from "./builderPageSharedDslEditors";

const GlobalSettingsEditor: React.FC<{
  fields: Record<string, unknown>;
  onUpdate: (fields: Record<string, unknown>) => void;
  endpoints: ASTBackendDecl[];
  onSelectEndpoint: () => void;
}> = ({ fields, onUpdate, endpoints: allEndpoints }) => {
  const [local, setLocal] = useState<Record<string, unknown>>(() =>
    structuredClone(fields),
  );
  const [collapsedSections, setCollapsedSections] = useState<
    Record<string, boolean>
  >({});

  useEffect(() => {
    setLocal(structuredClone(fields));
  }, [fields]);

  const toggleCollapse = useCallback((key: string) => {
    setCollapsedSections((previous) => ({ ...previous, [key]: !previous[key] }));
  }, []);

  const setField = useCallback((key: string, value: unknown) => {
    setLocal((previous) => ({ ...previous, [key]: value }));
  }, []);

  const setNestedField = useCallback(
    (parentKey: string, childKey: string, value: unknown) => {
      setLocal((previous) => {
        const parent = getObj(previous, parentKey);
        return {
          ...previous,
          [parentKey]: { ...parent, [childKey]: value },
        };
      });
    },
    [],
  );

  const setDeepField = useCallback(
    (p1: string, p2: string, p3: string, value: unknown) => {
      setLocal((previous) => {
        const parent = getObj(previous, p1);
        const child = getObj(parent, p2);
        return {
          ...previous,
          [p1]: { ...parent, [p2]: { ...child, [p3]: value } },
        };
      });
    },
    [],
  );

  const handleSave = useCallback(() => {
    onUpdate(local);
  }, [local, onUpdate]);

  const dslPreview = useMemo(() => generateGlobalDslPreview(local), [local]);

  const promptGuard = getObj(local, "prompt_guard");
  const hallucination = getObj(local, "hallucination_mitigation");
  const observability = getObj(local, "observability");
  const tracing = getObj(observability, "tracing");
  const metrics = getObj(observability, "metrics");
  const authz = getObj(local, "authz");
  const ratelimit = getObj(local, "ratelimit");
  const modelSelection = getObj(local, "model_selection");
  const reasoningFamilies = getObj(local, "reasoning_families") as Record<
    string,
    unknown
  >;
  const looper = getObj(local, "looper");
  const listeners = getListeners(local, "listeners");

  const vllmEndpoints = allEndpoints.filter(
    (endpoint) => endpoint.backendType === "vllm_endpoint",
  );
  const providerProfiles = allEndpoints.filter(
    (endpoint) => endpoint.backendType === "provider_profile",
  );

  const updateListener = useCallback(
    (index: number, field: keyof EditableListener, value: string | number) => {
      setLocal((previous) => {
        const current = getListeners(previous, "listeners");
        const next = current.map((listener, listenerIndex) =>
          listenerIndex === index ? { ...listener, [field]: value } : listener,
        );
        return { ...previous, listeners: next };
      });
    },
    [],
  );

  const addListener = useCallback(() => {
    setLocal((previous) => {
      const current = getListeners(previous, "listeners");
      const nextPort =
        current.reduce(
          (maxPort, listener) => Math.max(maxPort, listener.port),
          DEFAULT_LISTENER_PORT - 1,
        ) + 1;
      return {
        ...previous,
        listeners: [
          ...current,
          {
            name: `http-${nextPort}`,
            address: "0.0.0.0",
            port: nextPort,
            timeout: "300s",
          },
        ],
      };
    });
  }, []);

  const removeListener = useCallback((index: number) => {
    setLocal((previous) => {
      const current = getListeners(previous, "listeners");
      if (current.length <= 1) return previous;
      return {
        ...previous,
        listeners: current.filter(
          (_, listenerIndex) => listenerIndex !== index,
        ),
      };
    });
  }, []);

  return (
    <div className={styles.globalEditor}>
      <div className={styles.globalSaveBar}>
        <span className={styles.globalSaveHint}>
          Edit global defaults and cross-cutting settings
        </span>
        <button
          className={styles.toolbarBtnPrimary}
          onClick={handleSave}
          style={{ padding: "0.375rem 1rem", fontSize: "var(--text-xs)" }}
        >
          Save
        </button>
      </div>

      <GlobalSettingsRoutingSection
        local={local}
        collapsedSections={collapsedSections}
        modelSelection={modelSelection}
        reasoningFamilies={reasoningFamilies}
        looper={looper}
        listeners={listeners}
        onToggleSection={toggleCollapse}
        onSetField={setField}
        onSetNestedField={setNestedField}
        onUpdateListener={updateListener}
        onAddListener={addListener}
        onRemoveListener={removeListener}
      />

      <GlobalSettingsSafetySection
        local={local}
        collapsedSections={collapsedSections}
        promptGuard={promptGuard}
        hallucination={hallucination}
        authz={authz}
        ratelimit={ratelimit}
        onToggleSection={toggleCollapse}
        onSetField={setField}
        onSetNestedField={setNestedField}
        onSetDeepField={setDeepField}
      />

      <GlobalSettingsObservabilitySection
        local={local}
        collapsedSections={collapsedSections}
        tracing={tracing}
        metrics={metrics}
        onToggleSection={toggleCollapse}
        onSetField={setField}
      />

      <GlobalSettingsEndpointsSection
        collapsedSections={collapsedSections}
        vllmEndpoints={vllmEndpoints}
        providerProfiles={providerProfiles}
        onToggleSection={toggleCollapse}
      />

      <DslPreviewPanel dslText={dslPreview} />
    </div>
  );
};

export { GlobalSettingsEditor };
