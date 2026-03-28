package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("IsRAGEnabledForDecision", func() {
	It("returns false for decision without RAG plugin", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{Name: "no-rag", ModelRefs: []ModelRef{{Model: "m"}}},
				},
			},
		}
		Expect(cfg.IsRAGEnabledForDecision("no-rag")).To(BeFalse())
	})

	It("returns false when RAG plugin is explicitly disabled", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{
						Name:      "rag-disabled",
						ModelRefs: []ModelRef{{Model: "m"}},
						Plugins: []DecisionPlugin{
							{Type: "rag", Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": false,
								"backend": "milvus",
							})},
						},
					},
				},
			},
		}
		Expect(cfg.IsRAGEnabledForDecision("rag-disabled")).To(BeFalse())
	})

	It("returns true when RAG plugin is enabled", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{
						Name:      "rag-on",
						ModelRefs: []ModelRef{{Model: "m"}},
						Plugins: []DecisionPlugin{
							{Type: "rag", Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": true,
								"backend": "milvus",
							})},
						},
					},
				},
			},
		}
		Expect(cfg.IsRAGEnabledForDecision("rag-on")).To(BeTrue())
	})

	It("returns false for non-existent decision", func() {
		cfg := &RouterConfig{}
		Expect(cfg.IsRAGEnabledForDecision("nonexistent")).To(BeFalse())
	})
})

var _ = Describe("IsMemoryEnabledForDecision", func() {
	It("returns false when global memory is disabled and no per-decision config", func() {
		cfg := &RouterConfig{
			Memory: MemoryConfig{Enabled: false},
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{Name: "test", ModelRefs: []ModelRef{{Model: "m"}}},
				},
			},
		}
		Expect(cfg.IsMemoryEnabledForDecision("test")).To(BeFalse())
	})

	It("returns true when global memory is enabled and no per-decision override", func() {
		cfg := &RouterConfig{
			Memory: MemoryConfig{Enabled: true},
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{Name: "test", ModelRefs: []ModelRef{{Model: "m"}}},
				},
			},
		}
		Expect(cfg.IsMemoryEnabledForDecision("test")).To(BeTrue())
	})

	It("returns false when per-decision memory plugin explicitly disables memory", func() {
		cfg := memoryDecisionConfig(true, false)
		Expect(cfg.IsMemoryEnabledForDecision("mem-off")).To(BeFalse())
	})

	It("returns true when per-decision memory plugin enables memory", func() {
		cfg := memoryDecisionConfig(false, true)
		Expect(cfg.IsMemoryEnabledForDecision("mem-on")).To(BeTrue())
	})

	It("returns global default for non-existent decision", func() {
		cfg := &RouterConfig{Memory: MemoryConfig{Enabled: true}}
		Expect(cfg.IsMemoryEnabledForDecision("nonexistent")).To(BeTrue())
	})
})

var _ = Describe("HasPersonalizationPlugins", func() {
	It("returns false when neither RAG nor memory is enabled", func() {
		cfg := &RouterConfig{
			Memory: MemoryConfig{Enabled: false},
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{Name: "plain", ModelRefs: []ModelRef{{Model: "m"}}},
				},
			},
		}
		Expect(cfg.HasPersonalizationPlugins("plain")).To(BeFalse())
	})

	It("returns true when only RAG is enabled", func() {
		cfg := ragDecisionConfig("rag-only")
		Expect(cfg.HasPersonalizationPlugins("rag-only")).To(BeTrue())
	})

	It("returns true when only memory is enabled (global)", func() {
		cfg := &RouterConfig{
			Memory: MemoryConfig{Enabled: true},
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{Name: "mem-global", ModelRefs: []ModelRef{{Model: "m"}}},
				},
			},
		}
		Expect(cfg.HasPersonalizationPlugins("mem-global")).To(BeTrue())
	})

	It("returns true when both RAG and memory are enabled", func() {
		cfg := &RouterConfig{
			Memory: MemoryConfig{Enabled: true},
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{
						Name:      "both",
						ModelRefs: []ModelRef{{Model: "m"}},
						Plugins: []DecisionPlugin{
							{Type: "rag", Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": true,
								"backend": "external_api",
							})},
							{Type: "memory", Configuration: MustStructuredPayload(map[string]interface{}{
								"enabled": true,
							})},
						},
					},
				},
			},
		}
		Expect(cfg.HasPersonalizationPlugins("both")).To(BeTrue())
	})
})

func ragDecisionConfig(name string) *RouterConfig {
	return &RouterConfig{
		Memory: MemoryConfig{Enabled: false},
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{
				{
					Name:      name,
					ModelRefs: []ModelRef{{Model: "m"}},
					Plugins: []DecisionPlugin{
						{Type: "rag", Configuration: MustStructuredPayload(map[string]interface{}{
							"enabled": true,
							"backend": "milvus",
						})},
					},
				},
			},
		},
	}
}

func memoryDecisionConfig(globalEnabled, perDecisionEnabled bool) *RouterConfig {
	name := "mem-on"
	if !perDecisionEnabled {
		name = "mem-off"
	}
	return &RouterConfig{
		Memory: MemoryConfig{Enabled: globalEnabled},
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{
				{
					Name:      name,
					ModelRefs: []ModelRef{{Model: "m"}},
					Plugins: []DecisionPlugin{
						{Type: "memory", Configuration: MustStructuredPayload(map[string]interface{}{
							"enabled": perDecisionEnabled,
						})},
					},
				},
			},
		},
	}
}
