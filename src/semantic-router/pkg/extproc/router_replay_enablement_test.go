package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestInitializeReplayRecordersUsesGlobalReplayDefault(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{Name: "inherits-global", ModelRefs: []config.ModelRef{{Model: "m"}}},
				{
					Name:      "opt-out",
					ModelRefs: []config.ModelRef{{Model: "m"}},
					Plugins: []config.DecisionPlugin{
						{Type: config.DecisionPluginRouterReplay, Configuration: config.MustStructuredPayload(map[string]interface{}{
							"enabled": false,
						})},
					},
				},
			},
		},
	}

	recorders := initializeReplayRecorders(cfg)
	if _, ok := recorders["inherits-global"]; !ok {
		t.Fatalf("expected global replay to create a recorder for decisions without an explicit plugin")
	}
	if _, ok := recorders["opt-out"]; ok {
		t.Fatalf("expected per-decision enabled=false to disable router replay")
	}
}

func TestApplyDecisionResultToContextUsesEffectiveRouterReplayConfig(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{Name: "inherits-global", ModelRefs: []config.ModelRef{{Model: "m"}}},
			},
		},
	}
	router := &OpenAIRouter{Config: cfg}
	ctx := &RequestContext{}

	router.applyDecisionResultToContext(&decision.DecisionResult{
		Decision: &cfg.Decisions[0],
	}, ctx)

	if ctx.RouterReplayPluginConfig == nil {
		t.Fatalf("expected effective replay config to be attached to the request context")
	}
	if !ctx.RouterReplayPluginConfig.Enabled {
		t.Fatalf("expected attached replay config to be enabled")
	}
	if ctx.RouterReplayPluginConfig.MaxRecords != 10000 {
		t.Fatalf("expected default max_records=10000, got %d", ctx.RouterReplayPluginConfig.MaxRecords)
	}
	if !ctx.RouterReplayPluginConfig.CaptureRequestBody || !ctx.RouterReplayPluginConfig.CaptureResponseBody {
		t.Fatalf("expected effective replay config to capture both request and response bodies by default")
	}
}

func TestBuildReplayPostgresConfigUsesUnifiedTableName(t *testing.T) {
	pgConfig := buildReplayPostgresConfig(&config.RouterReplayPostgresConfig{
		Host:      "localhost",
		Database:  "router",
		User:      "postgres",
		TableName: "router_replay",
	})

	if pgConfig.TableName != "router_replay" {
		t.Fatalf("expected unified postgres table name, got %q", pgConfig.TableName)
	}
}
