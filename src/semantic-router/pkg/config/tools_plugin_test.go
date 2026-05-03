package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

var _ = Describe("ToolsPluginConfig", func() {
	Describe("Validate", registerToolsValidationSpecs)
	Describe("DynamicRetrieval helpers", registerToolsDynamicRetrievalAccessorSpecs)
	Describe("YAML round-trip", registerToolsDynamicRetrievalYAMLSpecs)
})

func registerToolsValidationSpecs() {
	It("accepts an empty disabled config", func() {
		cfg := &ToolsPluginConfig{}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("accepts a passthrough config without dynamic_retrieval", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
		}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("accepts a config with dynamic_retrieval disabled", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled: false,
				// Other fields intentionally invalid; should be ignored when disabled.
				Strategy:             "garbage",
				MinHistoryConfidence: -5.0,
			},
		}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("accepts a semantic_only dynamic_retrieval config", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:  true,
				Strategy: DynamicRetrievalStrategySemanticOnly,
			},
		}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("accepts a hybrid_history config with valid weights", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:              true,
				Strategy:             DynamicRetrievalStrategyHybridHistory,
				HistoryWindow:        10,
				MinHistoryConfidence: 0.5,
				Weights: &DynamicRetrievalWeights{
					Semantic:          1.0,
					History:           0.7,
					DecisionPrior:     0.2,
					RepetitionPenalty: 0.1,
				},
			},
		}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("rejects an unknown strategy name", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:  true,
				Strategy: "not_a_real_strategy",
			},
		}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("dynamic_retrieval.strategy must be one of"))
	})

	It("rejects history_window < 1 when strategy is hybrid_history", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:       true,
				Strategy:      DynamicRetrievalStrategyHybridHistory,
				HistoryWindow: 0,
			},
		}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("history_window must be >= 1"))
	})

	It("permits history_window=0 when strategy is semantic_only", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:       true,
				Strategy:      DynamicRetrievalStrategySemanticOnly,
				HistoryWindow: 0,
			},
		}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("rejects min_history_confidence below 0", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:              true,
				Strategy:             DynamicRetrievalStrategySemanticOnly,
				MinHistoryConfidence: -0.1,
			},
		}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("min_history_confidence must be between 0.0 and 1.0"))
	})

	It("rejects min_history_confidence above 1", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:              true,
				Strategy:             DynamicRetrievalStrategySemanticOnly,
				MinHistoryConfidence: 1.5,
			},
		}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("min_history_confidence must be between 0.0 and 1.0"))
	})

	It("rejects a negative weight", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:       true,
				Strategy:      DynamicRetrievalStrategyHybridHistory,
				HistoryWindow: 5,
				Weights: &DynamicRetrievalWeights{
					Semantic: -0.5,
				},
			},
		}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("weights.semantic must be non-negative"))
	})

	It("accepts zero weights as a way to disable individual signals", func() {
		cfg := &ToolsPluginConfig{
			Enabled: true,
			Mode:    ToolsPluginModePassthrough,
			DynamicRetrieval: &DynamicRetrievalConfig{
				Enabled:       true,
				Strategy:      DynamicRetrievalStrategyHybridHistory,
				HistoryWindow: 5,
				Weights: &DynamicRetrievalWeights{
					Semantic:          1.0,
					History:           0.0,
					DecisionPrior:     0.0,
					RepetitionPenalty: 0.0,
				},
			},
		}
		Expect(cfg.Validate()).To(Succeed())
	})
}

func registerToolsDynamicRetrievalAccessorSpecs() {
	It("returns false for DynamicRetrievalEnabled when receiver is nil", func() {
		var c *ToolsPluginConfig
		Expect(c.DynamicRetrievalEnabled()).To(BeFalse())
	})

	It("returns false for DynamicRetrievalEnabled when block is unset", func() {
		c := &ToolsPluginConfig{Enabled: true}
		Expect(c.DynamicRetrievalEnabled()).To(BeFalse())
	})

	It("returns false for DynamicRetrievalEnabled when block is disabled", func() {
		c := &ToolsPluginConfig{
			Enabled:          true,
			DynamicRetrieval: &DynamicRetrievalConfig{Enabled: false},
		}
		Expect(c.DynamicRetrievalEnabled()).To(BeFalse())
	})

	It("returns true for DynamicRetrievalEnabled when block is enabled", func() {
		c := &ToolsPluginConfig{
			Enabled:          true,
			DynamicRetrieval: &DynamicRetrievalConfig{Enabled: true},
		}
		Expect(c.DynamicRetrievalEnabled()).To(BeTrue())
	})

	It("defaults EffectiveStrategy to semantic_only for nil receiver", func() {
		var d *DynamicRetrievalConfig
		Expect(d.EffectiveStrategy()).To(Equal(DynamicRetrievalStrategySemanticOnly))
	})

	It("defaults EffectiveStrategy to semantic_only when Strategy is empty", func() {
		d := &DynamicRetrievalConfig{Enabled: true}
		Expect(d.EffectiveStrategy()).To(Equal(DynamicRetrievalStrategySemanticOnly))
	})

	It("returns the configured strategy when set", func() {
		d := &DynamicRetrievalConfig{
			Enabled:  true,
			Strategy: DynamicRetrievalStrategyHybridHistory,
		}
		Expect(d.EffectiveStrategy()).To(Equal(DynamicRetrievalStrategyHybridHistory))
	})
}

func registerToolsDynamicRetrievalYAMLSpecs() {
	It("unmarshals a full dynamic_retrieval block from YAML", func() {
		cfgYAML := `
enabled: true
mode: passthrough
dynamic_retrieval:
  enabled: true
  strategy: hybrid_history
  history_window: 8
  weights:
    semantic: 1.0
    history: 0.7
    decision_prior: 0.2
    repetition_penalty: 0.1
  min_history_confidence: 0.4
  fallback_on_low_confidence: true
`
		cfg := &ToolsPluginConfig{}
		Expect(yaml.Unmarshal([]byte(cfgYAML), cfg)).To(Succeed())

		Expect(cfg.DynamicRetrievalEnabled()).To(BeTrue())
		Expect(cfg.DynamicRetrieval.EffectiveStrategy()).To(Equal(DynamicRetrievalStrategyHybridHistory))
		Expect(cfg.DynamicRetrieval.HistoryWindow).To(Equal(8))
		Expect(cfg.DynamicRetrieval.MinHistoryConfidence).To(Equal(0.4))
		Expect(cfg.DynamicRetrieval.FallbackOnLowConfidence).To(BeTrue())
		Expect(cfg.DynamicRetrieval.Weights).NotTo(BeNil())
		Expect(cfg.DynamicRetrieval.Weights.Semantic).To(Equal(1.0))
		Expect(cfg.DynamicRetrieval.Weights.History).To(Equal(0.7))
		Expect(cfg.DynamicRetrieval.Weights.DecisionPrior).To(Equal(0.2))
		Expect(cfg.DynamicRetrieval.Weights.RepetitionPenalty).To(Equal(0.1))
		Expect(cfg.Validate()).To(Succeed())
	})

	It("treats an absent dynamic_retrieval block as nil", func() {
		cfgYAML := `
enabled: true
mode: passthrough
`
		cfg := &ToolsPluginConfig{}
		Expect(yaml.Unmarshal([]byte(cfgYAML), cfg)).To(Succeed())

		Expect(cfg.DynamicRetrieval).To(BeNil())
		Expect(cfg.DynamicRetrievalEnabled()).To(BeFalse())
		Expect(cfg.Validate()).To(Succeed())
	})

	It("surfaces validation errors from a bad YAML config", func() {
		cfgYAML := `
enabled: true
mode: passthrough
dynamic_retrieval:
  enabled: true
  strategy: bogus
`
		cfg := &ToolsPluginConfig{}
		Expect(yaml.Unmarshal([]byte(cfgYAML), cfg)).To(Succeed())

		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("dynamic_retrieval.strategy must be one of"))
	})
}
