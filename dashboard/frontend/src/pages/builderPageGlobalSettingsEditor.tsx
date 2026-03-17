import React, { useCallback, useEffect, useMemo, useState } from "react";

import type { DSLFieldObject, DSLFieldValue } from "@/types/dsl";
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
  generateGlobalOverridePreview,
} from "./builderPageSharedDslEditors";

const GlobalSettingsEditor: React.FC<{
  fields: DSLFieldObject;
  onUpdate: (fields: DSLFieldObject) => void;
  endpoints: Array<{
    backendType: string;
    name: string;
    fields: DSLFieldObject;
  }>;
  onSelectEndpoint: () => void;
}> = ({ fields, onUpdate, endpoints: allEndpoints }) => {
  const [local, setLocal] = useState<DSLFieldObject>(() =>
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

  const setField = useCallback((key: string, value: DSLFieldValue) => {
    setLocal((previous) => ({ ...previous, [key]: value }));
  }, []);

  const setNestedField = useCallback(
    (parentKey: string, childKey: string, value: DSLFieldValue) => {
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
    (p1: string, p2: string, p3: string, value: DSLFieldValue) => {
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

  const globalPreview = useMemo(
    () => generateGlobalOverridePreview(local),
    [local],
  );

  const promptGuard = getObj(local, "prompt_guard");
  const hallucination = getObj(local, "hallucination_mitigation");
  const observability = getObj(local, "observability");
  const tracing = getObj(observability, "tracing");
  const metrics = getObj(observability, "metrics");
  const authz = getObj(local, "authz");
  const ratelimit = getObj(local, "ratelimit");
  const modelSelection = getObj(local, "model_selection");
  const reasoningFamilies = getObj(local, "reasoning_families");
  const looper = getObj(local, "looper");
  const listeners = getListeners(local, "listeners");

  const vllmEndpoints = allEndpoints.filter(
    (endpoint) => endpoint.backendType === "vllm_endpoint",
  );
  const providerProfiles = allEndpoints.filter(
    (endpoint) => endpoint.backendType === "provider_profile",
  );

  const serializeListeners = useCallback(
    (listeners: EditableListener[]): DSLFieldValue[] =>
      listeners.map((listener) => ({
        name: listener.name,
        address: listener.address,
        port: listener.port,
        timeout: listener.timeout,
      })),
    [],
  );

  const updateListener = useCallback(
    (index: number, field: keyof EditableListener, value: string | number) => {
      setLocal((previous) => {
        const current = getListeners(previous, "listeners");
        const next = current.map((listener, listenerIndex) =>
          listenerIndex === index ? { ...listener, [field]: value } : listener,
        );
        return { ...previous, listeners: serializeListeners(next) };
      });
    },
    [serializeListeners],
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
        listeners: serializeListeners([
          ...current,
          {
            name: `http-${nextPort}`,
            address: "0.0.0.0",
            port: nextPort,
            timeout: "300s",
          },
        ]),
      };
    });
  }, [serializeListeners]);

  const removeListener = useCallback((index: number) => {
    setLocal((previous) => {
      const current = getListeners(previous, "listeners");
      if (current.length <= 1) return previous;
      return {
        ...previous,
        listeners: serializeListeners(
          current.filter((_, listenerIndex) => listenerIndex !== index),
        ),
      };
    });
  }, [serializeListeners]);

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

      <DslPreviewPanel
        title="Canonical global override"
        dslText={globalPreview}
      />
    </div>
  );
};

export { GlobalSettingsEditor };
