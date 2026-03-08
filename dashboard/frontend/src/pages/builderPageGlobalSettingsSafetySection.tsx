import React from "react";

import styles from "./BuilderPage.module.css";
import { getBool, getNum, getObj, getStr } from "./builderPageGlobalSettingsSupport";

interface GlobalSettingsSafetySectionProps {
  local: Record<string, unknown>;
  collapsedSections: Record<string, boolean>;
  promptGuard: Record<string, unknown>;
  hallucination: Record<string, unknown>;
  authz: Record<string, unknown>;
  ratelimit: Record<string, unknown>;
  onToggleSection: (key: string) => void;
  onSetField: (key: string, value: unknown) => void;
  onSetNestedField: (
    parentKey: string,
    childKey: string,
    value: unknown,
  ) => void;
  onSetDeepField: (
    p1: string,
    p2: string,
    p3: string,
    value: unknown,
  ) => void;
}

const GlobalSettingsSafetySection: React.FC<
  GlobalSettingsSafetySectionProps
> = ({
  local,
  collapsedSections,
  promptGuard,
  hallucination,
  authz,
  ratelimit,
  onToggleSection,
  onSetField,
  onSetNestedField,
  onSetDeepField,
}) => {
  return (
    <div className={styles.gsSection}>
      <div
        className={styles.gsSectionHeader}
        onClick={() => onToggleSection("safety")}
      >
        <svg
          className={styles.gsSectionChevron}
          data-open={!collapsedSections["safety"]}
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M3 2l4 3-4 3" />
        </svg>
        <span className={styles.gsSectionTitle}>Safety</span>
      </div>
      {!collapsedSections["safety"] && (
        <div className={styles.gsSectionBody}>
          <div className={styles.gsSubSection}>
            <div className={styles.gsSubHeader}>
              <label className={styles.gsCheckbox}>
                <input
                  type="checkbox"
                  checked={getBool(promptGuard, "enabled")}
                  onChange={(event) => {
                    const current = getObj(local, "prompt_guard");
                    if (event.target.checked) {
                      onSetField("prompt_guard", {
                        ...current,
                        enabled: true,
                        threshold: getNum(current, "threshold", 0.7),
                        model_type: getStr(current, "model_type", "candle"),
                      });
                    } else {
                      onSetField("prompt_guard", { ...current, enabled: false });
                    }
                  }}
                />
                <span className={styles.gsSubTitle}>Prompt Guard</span>
              </label>
            </div>
            {getBool(promptGuard, "enabled") && (
              <div className={styles.gsSubBody}>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Threshold</label>
                  <input
                    className={styles.fieldInput}
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    style={{ width: "6rem" }}
                    value={getNum(promptGuard, "threshold", 0.7)}
                    onChange={(event) =>
                      onSetNestedField(
                        "prompt_guard",
                        "threshold",
                        parseFloat(event.target.value) || 0,
                      )
                    }
                  />
                </div>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Model</label>
                  <div className={styles.gsRadioGroup}>
                    {[
                      { label: "Candle (local)", value: "candle" },
                      { label: "vLLM (external)", value: "vllm" },
                    ].map((option) => (
                      <label key={option.value} className={styles.gsRadio}>
                        <input
                          type="radio"
                          name="gs-pg-model"
                          checked={
                            getStr(promptGuard, "model_type", "candle") ===
                            option.value
                          }
                          onChange={() =>
                            onSetNestedField(
                              "prompt_guard",
                              "model_type",
                              option.value,
                            )
                          }
                        />
                        <span>{option.label}</span>
                      </label>
                    ))}
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
                  checked={getBool(hallucination, "enabled")}
                  onChange={(event) => {
                    const current = getObj(local, "hallucination_mitigation");
                    if (event.target.checked) {
                      const factCheckModel = getObj(current, "fact_check_model");
                      const hallucinationModel = getObj(
                        current,
                        "hallucination_model",
                      );
                      onSetField("hallucination_mitigation", {
                        ...current,
                        enabled: true,
                        fact_check_model: {
                          ...factCheckModel,
                          threshold: getNum(factCheckModel, "threshold", 0.7),
                        },
                        hallucination_model: {
                          ...hallucinationModel,
                          threshold: getNum(
                            hallucinationModel,
                            "threshold",
                            0.5,
                          ),
                        },
                        use_nli: getBool(current, "use_nli", false),
                      });
                    } else {
                      onSetField("hallucination_mitigation", {
                        ...current,
                        enabled: false,
                      });
                    }
                  }}
                />
                <span className={styles.gsSubTitle}>
                  Hallucination Mitigation
                </span>
              </label>
            </div>
            {getBool(hallucination, "enabled") && (
              <div className={styles.gsSubBody}>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Fact-Check Threshold</label>
                  <input
                    className={styles.fieldInput}
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    style={{ width: "6rem" }}
                    value={getNum(
                      getObj(hallucination, "fact_check_model"),
                      "threshold",
                      0.7,
                    )}
                    onChange={(event) =>
                      onSetDeepField(
                        "hallucination_mitigation",
                        "fact_check_model",
                        "threshold",
                        parseFloat(event.target.value) || 0,
                      )
                    }
                  />
                </div>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>
                    Hallucination Threshold
                  </label>
                  <input
                    className={styles.fieldInput}
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    style={{ width: "6rem" }}
                    value={getNum(
                      getObj(hallucination, "hallucination_model"),
                      "threshold",
                      0.5,
                    )}
                    onChange={(event) =>
                      onSetDeepField(
                        "hallucination_mitigation",
                        "hallucination_model",
                        "threshold",
                        parseFloat(event.target.value) || 0,
                      )
                    }
                  />
                </div>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>NLI Model</label>
                  <label className={styles.gsCheckbox}>
                    <input
                      type="checkbox"
                      checked={getBool(hallucination, "use_nli", false)}
                      onChange={(event) =>
                        onSetNestedField(
                          "hallucination_mitigation",
                          "use_nli",
                          event.target.checked,
                        )
                      }
                    />
                    <span>Enhanced explanations</span>
                  </label>
                </div>
              </div>
            )}
          </div>

          <div className={styles.gsSubSection}>
            <div className={styles.gsSubHeader}>
              <span className={styles.gsSubTitle}>Authorization</span>
            </div>
            <div className={styles.gsSubBody}>
              <div className={styles.gsRow}>
                <label className={styles.gsLabel}>Fail Open</label>
                <label className={styles.gsCheckbox}>
                  <input
                    type="checkbox"
                    checked={getBool(authz, "fail_open")}
                    onChange={(event) =>
                      onSetNestedField(
                        "authz",
                        "fail_open",
                        event.target.checked,
                      )
                    }
                  />
                  <span>Allow on auth failure</span>
                </label>
              </div>
            </div>
          </div>

          <div className={styles.gsSubSection}>
            <div className={styles.gsSubHeader}>
              <span className={styles.gsSubTitle}>Rate Limit</span>
            </div>
            <div className={styles.gsSubBody}>
              <div className={styles.gsRow}>
                <label className={styles.gsLabel}>Fail Open</label>
                <label className={styles.gsCheckbox}>
                  <input
                    type="checkbox"
                    checked={getBool(ratelimit, "fail_open")}
                    onChange={(event) =>
                      onSetNestedField(
                        "ratelimit",
                        "fail_open",
                        event.target.checked,
                      )
                    }
                  />
                  <span>Allow on rate limit failure</span>
                </label>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export { GlobalSettingsSafetySection };
