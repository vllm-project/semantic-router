package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// contextRouting is a routing scope whose one decision, d, has a context
// plugin of the given configuration, or none when it is nil.
func contextRouting(globalMemory bool, configuration *ContextPluginConfig) *RouterConfig {
	decision := Decision{Name: "d", ModelRefs: []ModelRef{{Model: "m"}}}
	if configuration != nil {
		decision.Plugins = []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(*configuration)}}
	}
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{decision}}}
	cfg.Memory.Enabled = globalMemory
	return cfg
}

var _ = Describe("a decision's context sources", func() {
	rag := ContextSource{Type: ContextSourceRAG, Backend: "milvus"}

	It("are none without a context plugin or global memory", func() {
		cfg := contextRouting(false, nil)
		Expect(cfg.IsRAGEnabledForDecision("d")).To(BeFalse())
		Expect(cfg.IsMemoryEnabledForDecision("d")).To(BeFalse())
		Expect(cfg.HasPersonalizationPlugins("d")).To(BeFalse())
		Expect(cfg.IsRAGEnabledForDecision("nonexistent")).To(BeFalse())
	})

	It("are the global memory for a decision without a context plugin", func() {
		cfg := contextRouting(true, nil)
		Expect(cfg.IsMemoryEnabledForDecision("d")).To(BeTrue())
		Expect(cfg.HasPersonalizationPlugins("d")).To(BeTrue())
		Expect(cfg.UsesMemory()).To(BeTrue())
	})

	It("are what an enabled context plugin lists", func() {
		cfg := contextRouting(true, &ContextPluginConfig{Enabled: true, Sources: []ContextSource{rag}})
		Expect(cfg.IsRAGEnabledForDecision("d")).To(BeTrue())
		Expect(cfg.IsMemoryEnabledForDecision("d")).To(BeFalse())
		Expect(cfg.HasPersonalizationPlugins("d")).To(BeTrue())
	})

	It("are none for a disabled context plugin, global memory included", func() {
		cfg := contextRouting(true, &ContextPluginConfig{Enabled: false, Sources: []ContextSource{rag}})
		Expect(cfg.IsRAGEnabledForDecision("d")).To(BeFalse())
		Expect(cfg.IsMemoryEnabledForDecision("d")).To(BeFalse())
		Expect(cfg.HasPersonalizationPlugins("d")).To(BeFalse())
	})

	It("make the memory store needed when a decision lists memory", func() {
		cfg := contextRouting(false, &ContextPluginConfig{Enabled: true, Sources: []ContextSource{{Type: ContextSourceMemory}}})
		Expect(cfg.UsesMemory()).To(BeTrue())
		Expect(contextRouting(false, &ContextPluginConfig{Enabled: true, Sources: []ContextSource{rag}}).UsesMemory()).To(BeFalse())
	})
})

var _ = Describe("EffectiveRouterReplayConfigForDecision", registerEffectiveRouterReplayConfigForDecisionSpecs)

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
