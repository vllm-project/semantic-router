package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyProviderPromptCachePluginBuildsRoutePolicy(t *testing.T) {
	decision := &config.Decision{
		Name: "cached-provider-route",
		Plugins: []config.DecisionPlugin{{
			Type: "provider_prompt_cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":   true,
				"system":    true,
				"tools":     true,
				"last_user": false,
				"ttl":       "1h",
			}),
		}},
	}
	ctx := &RequestContext{
		RequestID:           "req-provider-cache",
		VSRSelectedDecision: decision,
	}

	passthrough := applyProviderPromptCachePlugin(ctx, nil)

	if passthrough == nil || passthrough.AutoCacheControl == nil {
		t.Fatal("provider cache policy was not created")
	}
	if !passthrough.AutoCacheControl.System ||
		!passthrough.AutoCacheControl.Tools ||
		passthrough.AutoCacheControl.LastUser ||
		passthrough.AutoCacheControl.TTL != "1h" {
		t.Fatalf("unexpected provider cache policy: %#v", passthrough.AutoCacheControl)
	}
}

func TestApplyProviderPromptCachePluginHonorsAuthorizedBypass(t *testing.T) {
	decision := &config.Decision{
		Name: "controlled-provider-cache",
		Plugins: []config.DecisionPlugin{{
			Type: "provider_prompt_cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":                true,
				"system":                 true,
				"allow_request_controls": true,
			}),
		}},
	}
	ctx := &RequestContext{
		RequestID:           "req-provider-cache-bypass",
		Headers:             map[string]string{"x-vsr-provider-cache-control": "bypass"},
		VSRSelectedDecision: decision,
	}

	passthrough := applyProviderPromptCachePlugin(ctx, nil)

	if passthrough != nil {
		t.Fatalf("bypass must not create passthrough policy: %#v", passthrough)
	}
}
