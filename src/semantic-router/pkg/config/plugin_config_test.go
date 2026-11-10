package config

import (
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// layeredRouting is a config whose one recipe holds decision, its plugins
// layered over global as the loader layers them. Plain tests call it too, so
// it panics instead of using Gomega's global Expect, which needs a fail
// handler that only the Ginkgo suite registers.
func layeredRouting(global []DecisionPlugin, decision Decision) *RouterConfig {
	recipe := RoutingRecipe{Name: DefaultRecipeName, Profile: RoutingProfile{Decisions: []Decision{decision}}}
	if err := layerRecipePlugins(global, &recipe); err != nil {
		panic(fmt.Sprintf("layeredRouting: layering the plugins of decision %q: %v", decision.Name, err))
	}
	cfg := &RouterConfig{GlobalPlugins: global, Recipes: []RoutingRecipe{recipe}}
	cfg.mirrorDefaultRecipe()
	return cfg
}

// contextRouting is a recipe whose one decision, d, has a context plugin of
// the given configuration, or none when it is nil; with globalMemory,
// global.plugins recall memories.
func contextRouting(globalMemory bool, configuration *ContextPluginConfig) *RouterConfig {
	decision := Decision{Name: "d", ModelRefs: []ModelRef{{Model: "m"}}}
	if configuration != nil {
		decision.Plugins = []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(*configuration)}}
	}
	var global []DecisionPlugin
	if globalMemory {
		global = []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(ContextPluginConfig{
			Enabled: true, Sources: []ContextSource{{Type: ContextSourceMemory}},
		})}}
	}
	return layeredRouting(global, decision)
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

var _ = Describe("EffectiveRouterReplayConfig", registerEffectiveRouterReplayConfigSpecs)

func registerEffectiveRouterReplayConfigSpecs() {
	It("returns nil without a router_replay plugin in global.plugins or the decision", func() {
		cfg := routerReplayDecisionConfig(false, "plain", nil)
		Expect(cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("plain"))).To(BeNil())
	})

	It("takes the default capture settings from a global.plugins router_replay", func() {
		cfg := routerReplayDecisionConfig(true, "plain", nil)

		replayCfg := cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("plain"))
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
	It("allows a decision plugin to opt in without a global.plugins router_replay", func() {
		cfg := routerReplayDecisionConfig(false, "opt-in", map[string]interface{}{
			"enabled":               false,
			"capture_response_body": false,
		})
		Expect(cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("opt-in"))).To(BeNil())

		cfg.Decisions[0].Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
			"enabled":               false,
			"capture_response_body": false,
		})
		Expect(cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("opt-in"))).To(BeNil())

		cfg.Decisions[0].Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
			"enabled":               true,
			"capture_response_body": false,
		})
		replayCfg := cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("opt-in"))
		Expect(replayCfg).NotTo(BeNil())
		Expect(replayCfg.Enabled).To(BeTrue())
		Expect(replayCfg.CaptureRequestBody).To(BeTrue())
		Expect(replayCfg.CaptureResponseBody).To(BeFalse())
	})

	It("lets a decision plugin disable replay that global.plugins enables", func() {
		cfg := routerReplayDecisionConfig(true, "opt-out", map[string]interface{}{
			"enabled": false,
		})
		Expect(cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("opt-out"))).To(BeNil())
	})

	It("layers a decision plugin over global.plugins field by field", func() {
		cfg := routerReplayDecisionConfig(true, "customized", map[string]interface{}{
			"capture_response_body": false,
			"max_records":           42,
		})

		replayCfg := cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("customized"))
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

	var global []DecisionPlugin
	if globalEnabled {
		global = []DecisionPlugin{{Type: DecisionPluginRouterReplay, Configuration: MustStructuredPayload(map[string]interface{}{"enabled": true})}}
	}
	return layeredRouting(global, decision)
}
