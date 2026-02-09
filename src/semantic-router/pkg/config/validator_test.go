package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func boolPtr(b bool) *bool { return &b }

var _ = Describe("validateLatencyRules", func() {
	It("accepts nil and empty", func() {
		Expect(validateLatencyRules(nil)).To(Succeed())
		Expect(validateLatencyRules([]LatencyRule{})).To(Succeed())
	})

	It("accepts both percentiles set", func() {
		rules := []LatencyRule{{Name: "fast", TPOTPercentile: 10, TTFTPercentile: 50}}
		Expect(validateLatencyRules(rules)).To(Succeed())
	})

	It("accepts TPOT-only", func() {
		Expect(validateLatencyRules([]LatencyRule{{Name: "t", TPOTPercentile: 50}})).To(Succeed())
	})

	It("accepts TTFT-only", func() {
		Expect(validateLatencyRules([]LatencyRule{{Name: "t", TTFTPercentile: 50}})).To(Succeed())
	})

	It("accepts boundary values 1 and 100", func() {
		rules := []LatencyRule{{Name: "edge", TPOTPercentile: 1, TTFTPercentile: 100}}
		Expect(validateLatencyRules(rules)).To(Succeed())
	})

	It("rejects empty name", func() {
		err := validateLatencyRules([]LatencyRule{{Name: "", TPOTPercentile: 50}})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("name cannot be empty"))
	})

	It("rejects zero percentiles", func() {
		err := validateLatencyRules([]LatencyRule{{Name: "none"}})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("must specify at least one of"))
	})

	It("rejects TPOT > 100", func() {
		err := validateLatencyRules([]LatencyRule{{Name: "x", TPOTPercentile: 101}})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("tpot_percentile must be between 1 and 100"))
	})

	It("rejects TTFT > 100", func() {
		err := validateLatencyRules([]LatencyRule{{Name: "x", TTFTPercentile: 200}})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("ttft_percentile must be between 1 and 100"))
	})

	It("fails on second bad rule, not first", func() {
		rules := []LatencyRule{
			{Name: "good", TPOTPercentile: 50},
			{Name: "", TPOTPercentile: 50},
		}
		err := validateLatencyRules(rules)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("latency_rules[1]"))
	})
})

var _ = Describe("validateLoRAName", func() {
	loraConfig := func(model string, loras ...string) *RouterConfig {
		adapters := make([]LoRAAdapter, len(loras))
		for i, n := range loras {
			adapters[i] = LoRAAdapter{Name: n}
		}
		return &RouterConfig{
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{
					model: {LoRAs: adapters},
				},
			},
		}
	}

	It("accepts known adapter", func() {
		cfg := loraConfig("qwen3", "sql-expert", "code-review")
		Expect(validateLoRAName(cfg, "qwen3", "sql-expert")).To(Succeed())
		Expect(validateLoRAName(cfg, "qwen3", "code-review")).To(Succeed())
	})

	It("rejects model not in config", func() {
		cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{}}}
		err := validateLoRAName(cfg, "ghost", "adapter")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("is not defined in model_config"))
	})

	It("rejects model with no loras", func() {
		cfg := loraConfig("qwen3") // zero adapters
		err := validateLoRAName(cfg, "qwen3", "adapter")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("has no loras defined"))
	})

	It("rejects unknown name and lists available", func() {
		cfg := loraConfig("qwen3", "sql-expert", "code-review")
		err := validateLoRAName(cfg, "qwen3", "nope")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("sql-expert"))
		Expect(err.Error()).To(ContainSubstring("code-review"))
	})
})

var _ = Describe("validateConfigStructure", func() {
	It("skips everything in k8s mode", func() {
		cfg := &RouterConfig{
			ConfigSource: ConfigSourceKubernetes,
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{Name: "bad", ModelRefs: nil}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("accepts empty config", func() {
		Expect(validateConfigStructure(&RouterConfig{})).To(Succeed())
	})

	It("accepts valid decision", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "ok",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
				}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("rejects empty modelRefs", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{Name: "x", ModelRefs: []ModelRef{}}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("has no modelRefs defined"))
	})

	It("rejects blank model name", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "x",
					ModelRefs: []ModelRef{{
						Model:                 "",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)},
					}},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("model name cannot be empty"))
	})

	It("rejects nil use_reasoning", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name:      "x",
					ModelRefs: []ModelRef{{Model: "model-a"}},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("missing required field 'use_reasoning'"))
	})

	It("validates lora ref in decision", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "x",
					ModelRefs: []ModelRef{{
						Model:                 "qwen3",
						LoRAName:              "sql-expert",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
				}},
			},
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{
					"qwen3": {LoRAs: []LoRAAdapter{{Name: "sql-expert"}}},
				},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("rejects bad lora ref", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "x",
					ModelRefs: []ModelRef{{
						Model:                 "qwen3",
						LoRAName:              "nope",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
				}},
			},
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{
					"qwen3": {LoRAs: []LoRAAdapter{{Name: "sql-expert"}}},
				},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("is not defined in model"))
	})

	It("delegates to latency rule validation", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{{Name: "", TPOTPercentile: 50}},
				},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("name cannot be empty"))
	})
})
