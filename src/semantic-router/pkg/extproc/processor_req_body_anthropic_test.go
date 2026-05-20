package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleAnthropicRoutingStartsRouterReplay(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{Name: "simple-queries", ModelRefs: []config.ModelRef{{Model: "claude-sonnet-4.6"}}},
			},
		},
	}
	recorders := initializeReplayRecorders(cfg)
	router := &OpenAIRouter{
		Config:          cfg,
		ReplayRecorders: recorders,
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)

	request, err := parseOpenAIRequest([]byte(`{"model":"MoM","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("parseOpenAIRequest failed: %v", err)
	}

	ctx := &RequestContext{
		Headers:                  map[string]string{},
		RouterReplayPluginConfig: cfg.EffectiveRouterReplayConfigForDecision("simple-queries"),
		OriginalRequestBody:      []byte(`{"model":"MoM","messages":[{"role":"user","content":"hi"}]}`),
	}
	response, err := router.handleAnthropicRouting(request, "MoM", "claude-sonnet-4.6", "simple-queries", ctx)
	if err != nil {
		t.Fatalf("handleAnthropicRouting failed: %v", err)
	}
	if response == nil {
		t.Fatal("expected routing response")
	}
	if ctx.RouterReplayID == "" {
		t.Fatal("expected router replay to start on anthropic routing path")
	}
}
