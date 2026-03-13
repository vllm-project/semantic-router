import React from "react";

import type { ASTBackendDecl } from "@/types/dsl";

import styles from "./BuilderPage.module.css";
import { getBool, getNum, getObj, getStr } from "./builderPageGlobalSettingsSupport";

interface GlobalSettingsObservabilitySectionProps {
  local: Record<string, unknown>;
  collapsedSections: Record<string, boolean>;
  tracing: Record<string, unknown>;
  metrics: Record<string, unknown>;
  onToggleSection: (key: string) => void;
  onSetField: (key: string, value: unknown) => void;
}

const GlobalSettingsObservabilitySection: React.FC<
  GlobalSettingsObservabilitySectionProps
> = ({
  local,
  collapsedSections,
  tracing,
  metrics,
  onToggleSection,
  onSetField,
}) => {
  return (
    <div className={styles.gsSection}>
      <div
        className={styles.gsSectionHeader}
        onClick={() => onToggleSection("observability")}
      >
        <svg
          className={styles.gsSectionChevron}
          data-open={!collapsedSections["observability"]}
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M3 2l4 3-4 3" />
        </svg>
        <span className={styles.gsSectionTitle}>Observability</span>
      </div>
      {!collapsedSections["observability"] && (
        <div className={styles.gsSectionBody}>
          <div className={styles.gsSubSection}>
            <div className={styles.gsSubHeader}>
              <label className={styles.gsCheckbox}>
                <input
                  type="checkbox"
                  checked={getBool(tracing, "enabled")}
                  onChange={(event) => {
                    const observability = getObj(local, "observability");
                    const currentTracing = getObj(observability, "tracing");
                    if (event.target.checked) {
                      const sampling = getObj(currentTracing, "sampling");
                      const exporter = getObj(currentTracing, "exporter");
                      onSetField("observability", {
                        ...observability,
                        tracing: {
                          ...currentTracing,
                          enabled: true,
                          provider: getStr(
                            currentTracing,
                            "provider",
                            "opentelemetry",
                          ),
                          exporter: {
                            ...exporter,
                            endpoint: getStr(
                              exporter,
                              "endpoint",
                              "localhost:4317",
                            ),
                          },
                          sampling: {
                            ...sampling,
                            type: getStr(
                              sampling,
                              "type",
                              "probabilistic",
                            ),
                            rate: getNum(sampling, "rate", 0.1),
                          },
                        },
                      });
                    } else {
                      onSetField("observability", {
                        ...observability,
                        tracing: { ...currentTracing, enabled: false },
                      });
                    }
                  }}
                />
                <span className={styles.gsSubTitle}>Tracing</span>
              </label>
            </div>
            {getBool(tracing, "enabled") && (
              <div className={styles.gsSubBody}>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Provider</label>
                  <input
                    className={styles.fieldInput}
                    value={getStr(tracing, "provider", "opentelemetry")}
                    onChange={(event) => {
                      const observability = getObj(local, "observability");
                      const currentTracing = getObj(observability, "tracing");
                      onSetField("observability", {
                        ...observability,
                        tracing: {
                          ...currentTracing,
                          provider: event.target.value,
                        },
                      });
                    }}
                    placeholder="opentelemetry"
                  />
                </div>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Endpoint</label>
                  <input
                    className={styles.fieldInput}
                    value={getStr(getObj(tracing, "exporter"), "endpoint", "")}
                    onChange={(event) => {
                      const observability = getObj(local, "observability");
                      const currentTracing = getObj(observability, "tracing");
                      const exporter = getObj(currentTracing, "exporter");
                      onSetField("observability", {
                        ...observability,
                        tracing: {
                          ...currentTracing,
                          exporter: {
                            ...exporter,
                            endpoint: event.target.value,
                          },
                        },
                      });
                    }}
                    placeholder="localhost:4317"
                  />
                </div>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Sampling</label>
                  <div className={styles.gsInlineRow}>
                    <input
                      className={styles.fieldInput}
                      style={{ width: "8rem" }}
                      value={getStr(
                        getObj(tracing, "sampling"),
                        "type",
                        "probabilistic",
                      )}
                      onChange={(event) => {
                        const observability = getObj(local, "observability");
                        const currentTracing = getObj(observability, "tracing");
                        const sampling = getObj(currentTracing, "sampling");
                        onSetField("observability", {
                          ...observability,
                          tracing: {
                            ...currentTracing,
                            sampling: {
                              ...sampling,
                              type: event.target.value,
                            },
                          },
                        });
                      }}
                      placeholder="probabilistic"
                    />
                    <span className={styles.gsSmallLabel}>Rate:</span>
                    <input
                      className={styles.fieldInput}
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      style={{ width: "5rem" }}
                      value={getNum(getObj(tracing, "sampling"), "rate", 0.1)}
                      onChange={(event) => {
                        const observability = getObj(local, "observability");
                        const currentTracing = getObj(observability, "tracing");
                        const sampling = getObj(currentTracing, "sampling");
                        onSetField("observability", {
                          ...observability,
                          tracing: {
                            ...currentTracing,
                            sampling: {
                              ...sampling,
                              rate: parseFloat(event.target.value) || 0,
                            },
                          },
                        });
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className={styles.gsSubSection}>
            <div className={styles.gsSubHeader}>
              <label className={styles.gsCheckbox}>
                <input
                  type="checkbox"
                  checked={getBool(metrics, "enabled", true)}
                  onChange={(event) => {
                    const observability = getObj(local, "observability");
                    const currentMetrics = getObj(observability, "metrics");
                    onSetField("observability", {
                      ...observability,
                      metrics: {
                        ...currentMetrics,
                        enabled: event.target.checked,
                      },
                    });
                  }}
                />
                <span className={styles.gsSubTitle}>Metrics</span>
              </label>
            </div>
            {getBool(metrics, "enabled", true) && (
              <div className={styles.gsSubBody}>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Windowed Metrics</label>
                  <label className={styles.gsCheckbox}>
                    <input
                      type="checkbox"
                      checked={getBool(metrics, "windowed", false)}
                      onChange={(event) => {
                        const observability = getObj(local, "observability");
                        const currentMetrics = getObj(observability, "metrics");
                        onSetField("observability", {
                          ...observability,
                          metrics: {
                            ...currentMetrics,
                            windowed: event.target.checked,
                          },
                        });
                      }}
                    />
                    <span>Enable windowed aggregation</span>
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

interface GlobalSettingsEndpointsSectionProps {
  collapsedSections: Record<string, boolean>;
  vllmEndpoints: ASTBackendDecl[];
  providerProfiles: ASTBackendDecl[];
  onToggleSection: (key: string) => void;
}

const GlobalSettingsEndpointsSection: React.FC<
  GlobalSettingsEndpointsSectionProps
> = ({
  collapsedSections,
  vllmEndpoints,
  providerProfiles,
  onToggleSection,
}) => {
  return (
    <div className={styles.gsSection}>
      <div
        className={styles.gsSectionHeader}
        onClick={() => onToggleSection("endpoints")}
      >
        <svg
          className={styles.gsSectionChevron}
          data-open={!collapsedSections["endpoints"]}
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M3 2l4 3-4 3" />
        </svg>
        <span className={styles.gsSectionTitle}>Endpoints</span>
        <span className={styles.gsCountBadge}>
          {vllmEndpoints.length + providerProfiles.length}
        </span>
      </div>
      {!collapsedSections["endpoints"] && (
        <div className={styles.gsSectionBody}>
          {vllmEndpoints.length === 0 && providerProfiles.length === 0 ? (
            <div className={styles.gsEmptyHint}>
              No endpoints defined. Add endpoints via{" "}
              <strong>Backends &rarr; vllm_endpoint</strong> or{" "}
              <strong>provider_profile</strong>.
            </div>
          ) : (
            <div className={styles.gsEndpointTable}>
              <div className={styles.gsEndpointHeader}>
                <span>Name</span>
                <span>Address</span>
                <span>Type</span>
                <span>Weight</span>
              </div>
              {vllmEndpoints.map((endpoint) => (
                <div key={endpoint.name} className={styles.gsEndpointRow}>
                  <span className={styles.gsEndpointName}>{endpoint.name}</span>
                  <span className={styles.gsEndpointAddr}>
                    {getStr(endpoint.fields, "address", "—")}:
                    {getStr(endpoint.fields, "port", "—")}
                  </span>
                  <span className={styles.gsEndpointType}>
                    {getStr(endpoint.fields, "type", "vllm")}
                  </span>
                  <span>{getStr(endpoint.fields, "weight", "1")}</span>
                </div>
              ))}
              {providerProfiles.map((profile) => (
                <div key={profile.name} className={styles.gsEndpointRow}>
                  <span className={styles.gsEndpointName}>{profile.name}</span>
                  <span className={styles.gsEndpointAddr}>
                    {getStr(profile.fields, "base_url", "—")}
                  </span>
                  <span className={styles.gsEndpointType}>
                    {getStr(profile.fields, "type", "openai")}
                  </span>
                  <span>—</span>
                </div>
              ))}
            </div>
          )}
          <div className={styles.gsEndpointHint}>
            Endpoints are managed in <strong>Backends</strong>. This is a
            read-only summary.
          </div>
        </div>
      )}
    </div>
  );
};

export {
  GlobalSettingsEndpointsSection,
  GlobalSettingsObservabilitySection,
};
