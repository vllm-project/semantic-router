package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestUpdateResponseCache_SkipsWhenDecisionCacheDisabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "no-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					// No semantic-cache plugin
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "no-cache-decision",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.updateCalled, "should not store response when decision has no semantic-cache plugin")
}

func TestUpdateResponseCache_SkipsWhenDecisionCacheExplicitlyDisabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "disabled-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					Plugins: []config.DecisionPlugin{
						{
							Type:          "semantic-cache",
							Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false}),
						},
					},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "disabled-cache-decision",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.updateCalled, "should not store response when decision has semantic-cache disabled")
}

func TestUpdateResponseCache_StoresWhenDecisionCacheEnabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					Plugins: []config.DecisionPlugin{
						{
							Type:          "semantic-cache",
							Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
						},
					},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "cache-decision",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.True(t, mockCache.updateCalled, "should store response when decision has semantic-cache enabled")
}

func TestUpdateResponseCache_StoresWhenNoDecisionSelectedAndNoDecisionsConfigured(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "", // no decision matched
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.True(t, mockCache.updateCalled, "should store response when no decision is selected (global cache applies)")
}

func TestUpdateResponseCache_SkipsWhenNoDecisionSelectedButDecisionsConfigured(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "default-route",
					ModelRefs: []config.ModelRef{{Model: "test"}},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.updateCalled, "should not store response when decisions exist but no decision matched")
}

func TestParseResponseUsage_ExtractsUsageFields(t *testing.T) {
	usage := parseResponseUsage([]byte(`{
		"usage": {
			"prompt_tokens": 11,
			"completion_tokens": 7
		}
	}`), "test-model")

	assert.Equal(t, responseUsageMetrics{
		promptTokens:     11,
		completionTokens: 7,
	}, usage)
}

func TestParseResponseUsage_ReturnsZeroForInvalidUsageTypes(t *testing.T) {
	usage := parseResponseUsage([]byte(`{
		"usage": {
			"prompt_tokens": "11",
			"completion_tokens": 7
		}
	}`), "test-model")

	assert.Equal(t, responseUsageMetrics{}, usage)
}
