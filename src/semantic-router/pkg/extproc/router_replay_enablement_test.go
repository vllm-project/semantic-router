package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func TestInitializeReplayRecordersUsesGlobalReplayDefaultWithinEntrypoint(t *testing.T) {
	cfg, recipe := replayEntrypointConfig(t,
		config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		[]config.Decision{
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
	)

	recorders := initializeReplayRecorders(cfg)
	if _, ok := recorders[config.RoutingDecisionKey(recipe.RuntimeScope(), "inherits-global")]; !ok {
		t.Fatalf("expected global replay to create a recorder for decisions without an explicit plugin")
	}
	if _, ok := recorders[config.RoutingDecisionKey(recipe.RuntimeScope(), "opt-out")]; ok {
		t.Fatalf("expected per-decision enabled=false to disable router replay")
	}
}

func TestInitializeReplayRecordersIsolatesEntrypointBindings(t *testing.T) {
	cfg, _ := replayEntrypointConfig(t,
		config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		[]config.Decision{{Name: "shared-route"}},
	)
	second := cfg.Entrypoints[0]
	second.ID = "entrypoint-replay-second"
	second.Name = "router/replay-second"
	second.ModelNames = []string{"router/replay-second"}
	second.Rules = append([]config.EntrypointRule(nil), second.Rules...)
	second.Rules[0].ID = "rule-replay-second"
	cfg.Entrypoints = append(cfg.Entrypoints, second)
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("prepare second replay Entrypoint: %v", err)
	}
	firstRecipe, firstOK := cfg.RecipeForRequestModel("router/replay")
	secondRecipe, secondOK := cfg.RecipeForRequestModel("router/replay-second")
	if !firstOK || !secondOK {
		t.Fatal("resolve replay Entrypoints")
	}

	recorders := initializeReplayRecorders(cfg)
	for _, key := range []string{
		config.RoutingDecisionKey(firstRecipe.RuntimeScope(), "shared-route"),
		config.RoutingDecisionKey(secondRecipe.RuntimeScope(), "shared-route"),
	} {
		if _, ok := recorders[key]; !ok {
			t.Fatalf("expected isolated replay recorder %q, got keys %+v", key, recorders)
		}
	}
}

func TestNamedRecipeRecorderNeverFallsBackToBareDecisionKey(t *testing.T) {
	bareRecorder := new(routerreplay.Recorder)
	router := &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"shared-route": bareRecorder,
		},
	}
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})

	if recorder := router.resolveReplayRecorder(ctx, "shared-route"); recorder != nil {
		t.Fatal("named recipe must not reuse a bare/default-recipe recorder")
	}
}

func TestApplyDecisionResultToContextUsesEffectiveRouterReplayConfig(t *testing.T) {
	cfg, recipe := replayEntrypointConfig(t,
		config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		[]config.Decision{{Name: "inherits-global"}},
	)
	router := &OpenAIRouter{Config: cfg}
	ctx := &RequestContext{}

	router.applyDecisionResultToContext(&decision.DecisionResult{
		Decision: &recipe.Profile.Decisions[0],
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

func TestResolveReplayStoreBackend(t *testing.T) {
	tests := []struct {
		name string
		cfg  config.RouterReplayConfig
		want string
	}{
		{
			name: "explicit backend wins",
			cfg: config.RouterReplayConfig{
				StoreBackend: " redis ",
				Postgres:     &config.RouterReplayPostgresConfig{},
			},
			want: "redis",
		},
		{
			name: "postgres config implies postgres",
			cfg: config.RouterReplayConfig{
				Postgres: &config.RouterReplayPostgresConfig{},
			},
			want: "postgres",
		},
		{
			name: "redis config implies redis",
			cfg: config.RouterReplayConfig{
				Redis: &config.RouterReplayRedisConfig{},
			},
			want: "redis",
		},
		{
			name: "milvus config implies milvus",
			cfg: config.RouterReplayConfig{
				Milvus: &config.RouterReplayMilvusConfig{},
			},
			want: "milvus",
		},
		{
			name: "qdrant config implies qdrant",
			cfg: config.RouterReplayConfig{
				Qdrant: &config.RouterReplayQdrantConfig{},
			},
			want: "qdrant",
		},
		{
			name: "empty config defaults to memory",
			cfg:  config.RouterReplayConfig{},
			want: "memory",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveReplayStoreBackend(tt.cfg); got != tt.want {
				t.Fatalf("expected backend %q, got %q", tt.want, got)
			}
		})
	}
}

func TestCreateReplayRuntimeFailsClosedWhenEnabledSharedStoreIsInvalid(t *testing.T) {
	cfg, _ := replayEntrypointConfig(t,
		config.RouterReplayConfig{Enabled: true, StoreBackend: "redis"},
		[]config.Decision{{Name: "replay-required"}},
	)

	_, _, _, err := createReplayRuntime(cfg)
	if err == nil || !strings.Contains(err.Error(), "redis config required") {
		t.Fatalf("createReplayRuntime() error = %v, want enabled-store failure", err)
	}
}

func TestCreateReplayRuntimeAllowsExplicitDecisionOptInWithSafeDefaults(t *testing.T) {
	cfg, recipe := replayEntrypointConfig(t,
		config.RouterReplayConfig{Enabled: false, StoreBackend: "memory"},
		[]config.Decision{
			{
				Name: "replay-opt-in",
				Plugins: []config.DecisionPlugin{
					{Type: config.DecisionPluginRouterReplay, Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled": true,
					})},
				},
			},
		},
	)

	recorders, shared, sharedStore, err := createReplayRuntime(cfg)
	if err != nil {
		t.Fatalf("createReplayRuntime() returned error for memory opt-in: %v", err)
	}
	if shared != nil || sharedStore {
		t.Fatalf("memory replay must remain decision-scoped, shared=%v sharedStore=%v", shared, sharedStore)
	}
	if recorders[config.RoutingDecisionKey(recipe.RuntimeScope(), "replay-opt-in")] == nil {
		t.Fatalf("expected explicit decision opt-in to create an in-memory recorder")
	}
}

func replayEntrypointConfig(
	t *testing.T,
	replay config.RouterReplayConfig,
	decisions []config.Decision,
) (*config.RouterConfig, *config.RoutingRecipe) {
	t.Helper()
	assignments := make(map[string]config.RoutingAssignmentSet, len(decisions))
	for index := range decisions {
		decision := &decisions[index]
		decision.ID = "decision-" + decision.Name
		decision.ModelRefs = nil
		assignments[decision.ID] = config.RoutingAssignmentSet{Models: []config.RoutingModelAssignment{{
			ModelID: "model-replay", ModelRevision: 1, ModelName: "backend/replay", Weight: "1",
		}}}
	}
	cfg := &config.RouterConfig{
		RouterReplay: replay,
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"backend/replay": {ResourceID: "model-replay", ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: "recipe-replay", Revision: 1, Name: "replay", Profile: config.RoutingProfile{Decisions: decisions},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "entrypoint-replay", Revision: 1, Name: "router/replay", ModelNames: []string{"router/replay"},
			Rules: []config.EntrypointRule{{
				ID: "rule-replay", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: "recipe-replay", RecipeRevision: 1, Recipe: "replay", Assignments: assignments,
				},
			}},
		}},
	}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("prepare replay Entrypoint: %v", err)
	}
	recipe, ok := cfg.RecipeForRequestModel("router/replay")
	if !ok {
		t.Fatal("resolve replay Entrypoint")
	}
	return cfg, recipe
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
