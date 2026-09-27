package config

import "testing"

func TestDecisionPluginActivationUsesTypedDefaults(t *testing.T) {
	enabled, disabled := true, false
	for _, tc := range []struct {
		payload interface{}
		want    bool
	}{
		{&SystemPromptPluginConfig{}, false},
		{&SystemPromptPluginConfig{SystemPrompt: "policy"}, true},
		{&SystemPromptPluginConfig{Enabled: &enabled}, true},
		{&SystemPromptPluginConfig{Enabled: &disabled}, false},
		{&MemoryPluginConfig{}, false},
		{&MemoryPluginConfig{Enabled: true}, true},
		{&FastResponsePluginConfig{Message: "hello"}, true},
		{&HeaderMutationPluginConfig{}, true},
		{&RequestParamsPluginConfig{}, true},
	} {
		if got := DecisionPluginEnabled(tc.payload); got != tc.want {
			t.Fatalf("%T enabled=%v want %v", tc.payload, got, tc.want)
		}
	}
}

func TestDecodeDecisionPluginKeepsCanonicalValidation(t *testing.T) {
	for _, plugin := range []DecisionPlugin{
		{Type: "image_gen", Configuration: MustStructuredPayload(map[string]interface{}{})},
		{Type: DecisionPluginRequestParams, Configuration: MustStructuredPayload(map[string]interface{}{"default_max_tokens": 0})},
	} {
		if _, err := DecodeDecisionPlugin(plugin); err == nil {
			t.Fatalf("accepted invalid plugin %+v", plugin)
		}
	}
}

// A new registry member must explicitly agree with the runtime's activation
// semantics instead of silently inheriting a reflected field-name convention.
func TestDecisionPluginActivationMatchesEveryRuntimeGetter(t *testing.T) {
	for _, entry := range DecisionPluginCatalog() {
		for _, raw := range []map[string]interface{}{{}, {"enabled": true}, {"enabled": false}} {
			t.Run(entry.Type, func(t *testing.T) {
				plugin := DecisionPlugin{Type: entry.Type, Configuration: MustStructuredPayload(raw)}
				payload := DecisionPluginSchemaSamples()[entry.Type]
				if err := plugin.Configuration.DecodeInto(payload); err != nil {
					t.Fatal(err)
				}
				decision := &Decision{Plugins: []DecisionPlugin{plugin}}
				var want bool
				switch payload.(type) {
				case *ResponseCachePluginConfig:
					want = decision.GetResponseCacheConfig().Enabled
				case *MemoryPluginConfig:
					want = decision.GetMemoryConfig().Enabled
				case *SystemPromptPluginConfig:
					want = decision.IsSystemPromptEnabled()
				case *HeaderMutationPluginConfig:
					want = decision.GetHeaderMutationConfig() != nil
				case *HallucinationPluginConfig:
					want = decision.GetHallucinationConfig().Enabled
				case *RouterReplayPluginConfig:
					want = decision.GetRouterReplayConfig().Enabled
				case *RAGPluginConfig:
					want = decision.GetRAGConfig().Enabled
				case *FastResponsePluginConfig:
					want = decision.GetFastResponseConfig() != nil
				case *ToolsPluginConfig:
					want = decision.GetToolsConfig().Enabled
				case *ToolSelectionPluginConfig:
					want = decision.GetToolSelectionConfig().Enabled
				case *RequestParamsPluginConfig:
					want = decision.GetRequestParamsConfig() != nil
				case *ResponseJailbreakPluginConfig:
					want = decision.GetResponseJailbreakConfig().Enabled
				case *ContextCompressionPluginConfig:
					want = decision.GetContextCompressionConfig().Enabled
				case *PromptCachePluginConfig:
					want = decision.GetPromptCacheConfig().Enabled
				case *ShadowDispatchPluginConfig:
					want = decision.GetShadowDispatchConfig().Enabled
				default:
					t.Fatalf("new plugin %q requires a runtime activation contract assertion", entry.Type)
				}
				if got := DecisionPluginEnabled(payload); got != want {
					t.Fatalf("introspection=%v runtime=%v input=%v", got, want, raw)
				}
			})
		}
	}
}
