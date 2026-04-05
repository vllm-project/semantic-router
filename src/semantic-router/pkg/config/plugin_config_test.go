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

var _ = Describe("EffectiveRouterReplayConfigForDecision", registerEffectiveRouterReplayConfigForDecisionSpecs)

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

func registerEffectiveRouterReplayConfigForDecisionSpecs() {
	It("returns nil when replay is globally disabled and no decision override exists", func() {
		cfg := routerReplayDecisionConfig(false, "plain", nil)
		Expect(cfg.EffectiveRouterReplayConfigForDecision("plain")).To(BeNil())
	})

	It("inherits default replay settings when replay is globally enabled", func() {
		cfg := routerReplayDecisionConfig(true, "plain", nil)

		replayCfg := cfg.EffectiveRouterReplayConfigForDecision("plain")
		Expect(replayCfg).NotTo(BeNil())
		Expect(replayCfg.Enabled).To(BeTrue())
		Expect(replayCfg.MaxRecords).To(Equal(10000))
		Expect(replayCfg.CaptureRequestBody).To(BeTrue())
		Expect(replayCfg.CaptureResponseBody).To(BeTrue())
		Expect(replayCfg.MaxBodyBytes).To(Equal(4096))
	})

	Describe("decision plugin overrides", registerRouterReplayPluginOverrideSpecs)
}

func registerRouterReplayPluginOverrideSpecs() {
	It("allows a decision plugin to opt in even when global replay is disabled", func() {
		cfg := routerReplayDecisionConfig(false, "opt-in", map[string]interface{}{
			"enabled":               false,
			"capture_response_body": false,
		})
		Expect(cfg.EffectiveRouterReplayConfigForDecision("opt-in")).To(BeNil())

		cfg.Decisions[0].Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
			"enabled":               false,
			"capture_response_body": false,
		})
		Expect(cfg.EffectiveRouterReplayConfigForDecision("opt-in")).To(BeNil())

		cfg.Decisions[0].Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
			"enabled":               true,
			"capture_response_body": false,
		})
		replayCfg := cfg.EffectiveRouterReplayConfigForDecision("opt-in")
		Expect(replayCfg).NotTo(BeNil())
		Expect(replayCfg.Enabled).To(BeTrue())
		Expect(replayCfg.CaptureRequestBody).To(BeTrue())
		Expect(replayCfg.CaptureResponseBody).To(BeFalse())
	})

	It("lets a decision plugin disable replay when global replay is enabled", func() {
		cfg := routerReplayDecisionConfig(true, "opt-out", map[string]interface{}{
			"enabled": false,
		})
		Expect(cfg.EffectiveRouterReplayConfigForDecision("opt-out")).To(BeNil())
	})

	It("treats omitted enabled as inherit-from-global while applying other plugin overrides", func() {
		cfg := routerReplayDecisionConfig(true, "customized", map[string]interface{}{
			"capture_response_body": false,
			"max_records":           42,
		})

		replayCfg := cfg.EffectiveRouterReplayConfigForDecision("customized")
		Expect(replayCfg).NotTo(BeNil())
		Expect(replayCfg.Enabled).To(BeTrue())
		Expect(replayCfg.MaxRecords).To(Equal(42))
		Expect(replayCfg.CaptureRequestBody).To(BeTrue())
		Expect(replayCfg.CaptureResponseBody).To(BeFalse())
	})
}

func routerReplayDecisionConfig(globalEnabled bool, name string, pluginConfig map[string]interface{}) *RouterConfig {
	decision := Decision{
		Name:      name,
		ModelRefs: []ModelRef{{Model: "m"}},
	}
	if pluginConfig != nil {
		decision.Plugins = []DecisionPlugin{
			{
				Type:          "router_replay",
				Configuration: MustStructuredPayload(pluginConfig),
			},
		}
	}

	return &RouterConfig{
		RouterReplay: RouterReplayConfig{Enabled: globalEnabled},
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{decision},
		},
	}
}
